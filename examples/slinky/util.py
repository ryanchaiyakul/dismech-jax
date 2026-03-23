from __future__ import annotations
from abc import abstractmethod

import jax
import jax.numpy as jnp
import equinox as eqx
import numpy as np
import optax

from jaxtyping import Float, jaxtyped, TypeCheckError
from beartype import beartype

import dismech_jax


class TestCase(eqx.Module):
    bc: dismech_jax.BatchedLinearBC
    qs: jax.Array

    @classmethod
    def from_npz(cls, filename: str) -> TestCase:
        data = jnp.load(filename)
        loaded_bc = dismech_jax.BatchedLinearBC(
            idx_b=data["idx_b"], xb_c=data["xb_c"], xb_m=data["xb_m"]
        )
        return cls(bc=loaded_bc, qs=data["qs"])

    # TODO: create a TestCase npz?


class TripletModel(eqx.Module):
    """NN base class."""

    def __init__(self, der_K: jax.Array, key: jax.Array): ...

    @abstractmethod
    def __call__(self, del_strain: Float[jax.Array, "5"]) -> Float[jax.Array, ""]: ...

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        # Enforce jaxtyped runtime check for all subclasses
        parent_annotations = TripletModel.__call__.__annotations__
        if "__call__" in cls.__dict__:
            child_call = cls.__dict__["__call__"]
            child_call.__annotations__ = parent_annotations.copy()
            cls.__call__ = jaxtyped(typechecker=beartype)(child_call)


def validate_model(cls: type) -> None:
    """Validates a provided class as a suitable TripletModel. Throws a ValueError if improper."""
    if not issubclass(cls, TripletModel):
        raise ValueError(f"validate_model: {cls} is not an subclass of {TripletModel}")
    try:
        obj = cls(jnp.ones(5), jax.random.PRNGKey(42))
    except TypeError:
        raise ValueError(f"validate_model: {cls} cannot be initialized with a PRNGKey")
    try:
        obj(jnp.empty(5))
    except TypeCheckError as e:
        raise ValueError(f"validate_model: obj.__call__ must return a scalar:\n {e}")
    except Exception as e:
        raise ValueError(
            f"validate_model: obj.__call__ encountered an unknown exception: \n {e}"
        )


def get_base_rod():
    geom = dismech_jax.Geometry(0.5, 5e-3)
    mat = dismech_jax.Material(1273.52, 1e7)
    temp, aux = dismech_jax.Rod.from_geometry(geom, mat, N=3)

    # Replace F_ext i.e. gravity with full slinky
    MASS = 0.647  # From dismech-python
    # MASS = 0.09
    F_new = jnp.array(
        [
            0.0,
            0.0,
            MASS / 3 * -9.81,
            0.0,
            0.0,
            0.0,
            MASS / 3 * -9.81,
            0.0,
            0.0,
            0.0,
            MASS / 3 * -9.81,
        ]
    )

    base = eqx.tree_at(lambda r: r.E_ext, temp, dismech_jax.Gravity(F_new))
    der = base.get_DER(geom, mat)
    return base, aux, der

#  train and validation datafiles are now input, n_epochs is input, train and validation loss history is output
def train_model(
    cls: type,
    key: jax.Array = jax.random.PRNGKey(42),
    train_file: str = "train.npz",
    valid_file: str = "valid.npz",
    n_epochs: int = 100,
    lr: float = 1e-2,
) -> tuple:
    base, aux, der = get_base_rod()

    # Load from binary
    train = TestCase.from_npz(train_file)
    valid = TestCase.from_npz(valid_file)

    # Partially batched BCs
    train_rods = base.with_bc(train.bc)
    valid_rods = base.with_bc(valid.bc)

    train_lambdas = jnp.linspace(0.0, 1.0, train.qs.shape[0])
    valid_lambdas = jnp.linspace(0.0, 1.0, valid.qs.shape[0])

    model = cls(der_K=jnp.array([2.0, 2.0, 0.02, 0.02, 0.01]), key=key)
    ####################
    # Cosine decay schedule
    schedule = optax.cosine_decay_schedule(
        init_value=lr,
        decay_steps=n_epochs + 1,
        alpha=0.1,   # final LR = 0.1 * lr
    )

    optimizer = optax.adam(schedule)
    opt_state = optimizer.init(model)

    def loss(model: eqx.Module, rods: dismech_jax.Rod, lambdas: jax.Array, aux: jax.Array, truth: jax.Array):
        pred = rods.solve(
            model, lambdas, aux, max_dlambda=5e-3, iters=5, ls_steps=10
        )
        diff = pred - truth
        return jnp.mean(jnp.square(diff))
    
    @eqx.filter_jit
    def run_training(model, opt_state, num_steps: int, val_interval: int = 10):
        def train_step(carry, step_idx):
            m, s = carry

            loss_val, grads = eqx.filter_value_and_grad(loss)(
                m, train_rods, train_lambdas, aux, train.qs
            )
            updates, next_s = optimizer.update(grads, s, m)
            next_m = eqx.apply_updates(m, updates)

            is_val_step = (step_idx % val_interval) == 0
            valid_loss = jax.lax.cond(
                is_val_step,
                lambda: loss(next_m, valid_rods, valid_lambdas, aux, valid.qs),
                lambda: -1.0,
            )

            current_lr = schedule(step_idx)

            jax.debug.callback(
                lambda s, lr_now, t, v: print(
                    f"Step {s:<4} | LR: {lr_now:<10.3e} | Train: {t:<12.5e} | Valid: {v:<12.5e}"
                ) if v != -1.0 else None,
                step_idx,
                current_lr,
                loss_val,
                valid_loss,
            )

            return (next_m, next_s), (loss_val, valid_loss)

        (final_model, final_state), (train_history, valid_history) = jax.lax.scan(
            train_step, (model, opt_state), jnp.arange(num_steps)
        )
        return final_model, final_state, train_history, valid_history

    model, opt_state, train_history, valid_history = run_training(
        model, opt_state, n_epochs + 1
    )

    return model, train_history, valid_history
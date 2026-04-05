from __future__ import annotations
import warnings
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
    idx_b: jax.Array
    xb_c: jax.Array
    xb_m: jax.Array
    qs: jax.Array
    lambdas: jax.Array | None = None

    @classmethod
    def from_npz(cls, filename: str) -> "TestCase":
        data = np.load(filename)

        idx_b = jnp.asarray(data["idx_b"])
        xb_c = jnp.asarray(data["xb_c"])
        xb_m = jnp.asarray(data["xb_m"])
        qs = jnp.asarray(data["qs"])
        lambdas = jnp.asarray(data["lambdas"]) if "lambdas" in data else None

        if qs.ndim != 3:
            raise ValueError(
                f"{filename}: expected qs to have shape (n_traj, n_lambda, n_dof), "
                f"got shape {qs.shape}"
            )

        if xb_c.ndim != 2:
            print("xb_c: ", xb_c)
            xb_c = jnp.broadcast_to(xb_c, (qs.shape[0], xb_c.shape[0]))
            warnings.warn(
                f"{filename}: expected xb_c to have shape (n_traj, n_b), "
                f"got shape {xb_c.shape}. Using same xb_c for all trajectories."
            )

        if xb_m.ndim != 3:
            print("xb_m: ", xb_m)
            xb_m = jnp.broadcast_to(xb_m, (qs.shape[0], xb_m.shape[0]))
            warnings.warn(
                f"{filename}: expected xb_m to have shape (n_traj, n_b), "
                f"got shape {xb_m.shape}. Using same xb_m for all trajectories."
            )

        n_traj, n_lambda, _ = qs.shape

        if xb_c.shape[0] != n_traj:
            raise ValueError(
                f"{filename}: xb_c.shape[0] must match qs.shape[0]. "
                f"Got xb_c.shape={xb_c.shape}, qs.shape={qs.shape}"
            )

        if xb_m.shape[0] != n_traj:
            raise ValueError(
                f"{filename}: xb_m.shape[0] must match qs.shape[0]. "
                f"Got xb_m.shape={xb_m.shape}, qs.shape={qs.shape}"
            )

        if idx_b.ndim == 1:
            if idx_b.shape[0] != xb_c.shape[1]:
                raise ValueError(
                    f"{filename}: shared idx_b has incompatible size. "
                    f"Got idx_b.shape={idx_b.shape}, xb_c.shape={xb_c.shape}"
                )
        elif idx_b.ndim == 2:
            if idx_b.shape != xb_c.shape:
                raise ValueError(
                    f"{filename}: per-trajectory idx_b must have shape (n_traj, n_b). "
                    f"Got idx_b.shape={idx_b.shape}, xb_c.shape={xb_c.shape}"
                )
        else:
            raise ValueError(
                f"{filename}: idx_b must have shape (n_b,) or (n_traj, n_b). "
                f"Got shape {idx_b.shape}"
            )

        if lambdas is not None:
            if lambdas.ndim != 1 or lambdas.shape[0] != n_lambda:
                raise ValueError(
                    f"{filename}: lambdas must have shape (n_lambda,), "
                    f"got shape {lambdas.shape} with qs.shape={qs.shape}"
                )

        return cls(idx_b=idx_b, xb_c=xb_c, xb_m=xb_m, qs=qs, lambdas=lambdas)


class TripletModel(eqx.Module):
    """NN base class."""

    def __init__(self, der_K: jax.Array, key: jax.Array): ...

    @abstractmethod
    def __call__(self, del_strain: Float[jax.Array, "5"]) -> Float[jax.Array, ""]: ...

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        parent_annotations = TripletModel.__call__.__annotations__
        if "__call__" in cls.__dict__:
            child_call = cls.__dict__["__call__"]
            child_call.__annotations__ = parent_annotations.copy()
            cls.__call__ = jaxtyped(typechecker=beartype)(child_call)


def validate_model(cls: type, der_K: jax.Array) -> None:
    """Validates a provided class as a suitable TripletModel."""
    if not issubclass(cls, TripletModel):
        raise ValueError(f"validate_model: {cls} is not a subclass of {TripletModel}")

    try:
        obj = cls(der_K=jnp.asarray(der_K), key=jax.random.PRNGKey(42))
    except Exception as e:
        raise ValueError(
            f"validate_model: {cls} could not be initialized with der_K shape "
            f"{jnp.asarray(der_K).shape}:\n{e}"
        )

    try:
        out = obj(jnp.empty(5))
    except TypeCheckError as e:
        raise ValueError(f"validate_model: obj.__call__ must return a scalar:\n{e}")
    except Exception as e:
        raise ValueError(
            f"validate_model: obj.__call__ encountered an unknown exception:\n{e}"
        )

    if jnp.shape(out) != ():
        raise ValueError(
            f"validate_model: obj.__call__ must return a scalar, got shape {jnp.shape(out)}"
        )


def get_base_rod():
    geom = dismech_jax.Geometry(0.5, 5e-3)
    mat = dismech_jax.Material(1273.52, 1e7)
    temp, aux = dismech_jax.Rod.from_geometry(geom, mat, N=3)

    MASS = 0.647
    F_new = jnp.array(
        [
            0.0,
            0.0,
            MASS / 4 * -9.81,
            0.0,
            0.0,
            0.0,
            MASS / 2 * -9.81,
            0.0,
            0.0,
            0.0,
            MASS / 4 * -9.81,
        ]
    )

    base = eqx.tree_at(lambda r: r.E_ext, temp, dismech_jax.Gravity(F_new))
    der = base.get_DER(geom, mat)
    return base, aux, der


def _get_dataset_lambdas(dataset: TestCase) -> jax.Array:
    if dataset.lambdas is not None:
        return dataset.lambdas
    return jnp.linspace(0.0, 1.0, dataset.qs.shape[1])


def _get_idx_b_all(dataset: TestCase) -> jax.Array:
    """
    Returns idx_b with shape (n_traj, n_b), whether the file stored it
    as shared (n_b,) or per-trajectory (n_traj, n_b).
    """
    n_traj = dataset.qs.shape[0]

    if dataset.idx_b.ndim == 1:
        return jnp.broadcast_to(
            dataset.idx_b[None, :], (n_traj, dataset.idx_b.shape[0])
        )

    return dataset.idx_b


def spectral_spread_penalty(K: jax.Array, eps: float = 1e-8, max_log_spread: float = None) -> jax.Array:
    """
    Penalize the log spectral spread:
        (log(lambda_max) - log(lambda_min))^2
    """
    K = 0.5 * (K + K.T)  # numerical safety
    eigs = jnp.linalg.eigvalsh(K)  # ascending order
    lam_min = jnp.maximum(eigs[0], eps)
    lam_max = jnp.maximum(eigs[-1], eps)
    log_spread = jnp.log(lam_max) - jnp.log(lam_min)
    if max_log_spread is None:
        return log_spread**2
    excess = jnp.maximum(log_spread - max_log_spread, 0.0)
    return excess**2


def _trajectory_spectral_reg(
    model: eqx.Module,
    rod: dismech_jax.Rod,
    pred_qs: jax.Array,
    pred_auxs,
) -> jax.Array:
    del_strain_hist = rod.get_del_strain_history(pred_qs, pred_auxs)
    del_strain_flat = del_strain_hist.reshape(-1, del_strain_hist.shape[-1])

    penalties = jax.vmap(
        lambda ds: spectral_spread_penalty(model.get_K_matrix(ds), max_log_spread=2.0)
    )(del_strain_flat)

    return jnp.mean(penalties)


def _trajectory_losses(
    model: eqx.Module,
    base: dismech_jax.Rod,
    aux,
    idx_b: jax.Array,
    xb_c: jax.Array,
    xb_m: jax.Array,
    lambdas: jax.Array,
    truth_qs: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    bc = dismech_jax.BatchedLinearBC(idx_b=idx_b, xb_c=xb_c, xb_m=xb_m)
    rod = base.with_bc(bc)

    # --------------------------
    # 1. differentiable solve for data loss
    # --------------------------
    pred_qs = rod.solve(
        model,
        lambdas,
        aux,
        max_dlambda=5e-3,
        iters=5,
        ls_steps=10,
    )

    data_loss = jnp.mean(jnp.square(pred_qs - truth_qs))

    # --------------------------
    # 2. detached solve_with_aux for spectral regularizer
    # --------------------------
    pred_qs_aux, pred_auxs = rod.solve_with_aux(
        jax.lax.stop_gradient(model),
        lambdas,
        jax.lax.stop_gradient(aux),
        max_dlambda=5e-3,
        iters=5,
        ls_steps=10,
    )

    pred_qs_aux = jax.lax.stop_gradient(pred_qs_aux)
    pred_auxs = jax.lax.stop_gradient(pred_auxs)

    spec_loss = _trajectory_spectral_reg(model, rod, pred_qs_aux, pred_auxs)

    return data_loss, spec_loss

def _dataset_loss_terms(
    model: eqx.Module,
    base: dismech_jax.Rod,
    aux,
    dataset: TestCase,
) -> tuple[jax.Array, jax.Array]:
    lambdas = _get_dataset_lambdas(dataset)
    idx_b_all = _get_idx_b_all(dataset)

    data_losses, spec_losses = jax.vmap(
        lambda idx_b, xb_c, xb_m, truth_qs: _trajectory_losses(
            model=model,
            base=base,
            aux=aux,
            idx_b=idx_b,
            xb_c=xb_c,
            xb_m=xb_m,
            lambdas=lambdas,
            truth_qs=truth_qs,
        )
    )(idx_b_all, dataset.xb_c, dataset.xb_m, dataset.qs)

    return jnp.mean(data_losses), jnp.mean(spec_losses)


def _dataset_loss(
    model: eqx.Module,
    base: dismech_jax.Rod,
    aux,
    dataset: TestCase,
    alpha_spec: float = 0.0,
) -> jax.Array:
    """
    Total dataset loss = data loss + alpha_spec * spectral regularization.
    """
    data_loss, spec_loss = _dataset_loss_terms(model, base, aux, dataset)
    return data_loss + alpha_spec * spec_loss


def train_model(
    cls: type,
    key: jax.Array = jax.random.PRNGKey(42),
    train_file: str = "train.npz",
    valid_file: str = "valid.npz",
    n_epochs: int = 100,
    lr: float = 1e-2,
    init_K=jnp.array([2.0, 0.01, 0.02]),  # stretching, coupled, bending
    alpha_spec: float = 0.0,
) -> tuple:
    validate_model(cls, init_K)

    base, aux, der = get_base_rod()

    train = TestCase.from_npz(train_file)
    valid = TestCase.from_npz(valid_file)

    model = cls(der_K=init_K, key=key, l_k=base.triplets.l_k[0,0])
    init_K = model.get_K_entries(jnp.zeros(5))

    schedule = optax.cosine_decay_schedule(
        init_value=lr,
        decay_steps=n_epochs + 1,
        alpha=0.1,
    )

    optimizer = optax.adam(schedule)
    opt_state = optimizer.init(model)

    @eqx.filter_jit
    def run_training(model, opt_state, num_steps: int, val_interval: int = 10):
        def train_step(carry, step_idx):
            m, s = carry

            total_train_loss, grads = eqx.filter_value_and_grad(_dataset_loss)(
                m, base, aux, train, alpha_spec
            )

            train_data_loss, train_spec_loss = _dataset_loss_terms(
                m, base, aux, train
            )

            updates, next_s = optimizer.update(grads, s, m)
            next_m = eqx.apply_updates(m, updates)

            is_val_step = (step_idx % val_interval) == 0

            valid_total_loss = jax.lax.cond(
                is_val_step,
                lambda: _dataset_loss(next_m, base, aux, valid, alpha_spec),
                lambda: -1.0,
            )

            valid_data_loss = jax.lax.cond(
                is_val_step,
                lambda: _dataset_loss_terms(next_m, base, aux, valid)[0],
                lambda: -1.0,
            )

            valid_spec_loss = jax.lax.cond(
                is_val_step,
                lambda: _dataset_loss_terms(next_m, base, aux, valid)[1],
                lambda: -1.0,
            )

            current_lr = schedule(step_idx)

            jax.debug.callback(
                lambda s, lr_now, ttot, tdata, tspec, vtot, vdata, vspec, K: print(
                    f"Step {s:<4} | LR: {lr_now:<10.3e} | "
                    f"TrainTot: {ttot:<12.5e} | TrainData: {tdata:<12.5e} | "
                    f"TrainSpec: {tspec:<12.5e} | "
                    f"ValidTot: {vtot:<12.5e} | ValidData: {vdata:<12.5e} | "
                    f"ValidSpec: {vspec:<12.5e} | K: {K}"
                ) if vtot != -1.0 else None,
                step_idx,
                current_lr,
                total_train_loss,
                train_data_loss,
                train_spec_loss,
                valid_total_loss,
                valid_data_loss,
                valid_spec_loss,
                next_m.get_K_entries(jnp.zeros(5)),
            )

            return (next_m, next_s), (
                total_train_loss,
                train_data_loss,
                train_spec_loss,
                valid_total_loss,
                valid_data_loss,
                valid_spec_loss,
            )

        (final_model, final_state), histories = jax.lax.scan(
            train_step, (model, opt_state), jnp.arange(num_steps)
        )

        (
            train_total_history,
            train_data_history,
            train_spec_history,
            valid_total_history,
            valid_data_history,
            valid_spec_history,
        ) = histories

        return (
            final_model,
            final_state,
            train_total_history,
            train_data_history,
            train_spec_history,
            valid_total_history,
            valid_data_history,
            valid_spec_history,
        )

    (
        model,
        opt_state,
        train_total_history,
        train_data_history,
        train_spec_history,
        valid_total_history,
        valid_data_history,
        valid_spec_history,
    ) = run_training(model, opt_state, n_epochs + 1)

    return (
        model,
        init_K,
        train_total_history,
        train_data_history,
        train_spec_history,
        valid_total_history,
        valid_data_history,
        valid_spec_history,
    )
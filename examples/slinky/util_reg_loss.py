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

        # expected shape is (n_traj, n_b)
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
    if not issubclass(cls, TripletModel):
        raise ValueError(f"validate_model: {cls} is not a subclass of {TripletModel}")

    try:
        obj = cls(der_K=jnp.asarray(der_K), key=jax.random.PRNGKey(42), l_k=1.0)
    except TypeError:
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
    n_traj = dataset.qs.shape[0]
    if dataset.idx_b.ndim == 1:
        return jnp.broadcast_to(
            dataset.idx_b[None, :], (n_traj, dataset.idx_b.shape[0])
        )
    return dataset.idx_b


def spectral_spread_penalty(
    K: jax.Array,
    eps: float = 1e-8,
    max_log_spread: float | None = 2.0,
) -> jax.Array:
    K = 0.5 * (K + K.T)
    eigs = jnp.linalg.eigvalsh(K)
    lam_min = jnp.maximum(eigs[0], eps)
    lam_max = jnp.maximum(eigs[-1], eps)
    log_spread = jnp.log(lam_max) - jnp.log(lam_min)

    if max_log_spread is None:
        return log_spread**2

    excess = jnp.maximum(log_spread - max_log_spread, 0.0)
    return excess**2

def _alpha_spec_schedule(
    step_idx: jax.Array,
    alpha_spec_max: float,
    start_frac: float = 0.2,
    ramp_frac: float = 0.3,
    num_steps: int = 100,
) -> jax.Array:
    """
    Piecewise-linear schedule for spectral regularization.

    - no regularization until start_frac * num_steps
    - linearly ramp to alpha_spec_max over ramp_frac * num_steps
    - then keep constant
    """
    start_step = start_frac * num_steps
    ramp_steps = jnp.maximum(ramp_frac * num_steps, 1.0)

    progress = (step_idx - start_step) / ramp_steps
    progress = jnp.clip(progress, 0.0, 1.0)

    return alpha_spec_max * progress

def _build_aux_history_for_qs(aux0, qs: jax.Array):
    """
    Build auxiliary frame history from a sequence of configurations qs
    by repeatedly applying aux.update(q).

    Returns a TripletState-like pytree with leading axis n_steps.
    """
    def step(aux_prev, q):
        aux_next = jax.vmap(lambda a: a.update(q))(aux_prev)
        return aux_next, aux_next

    _, aux_hist = jax.lax.scan(step, aux0, qs)
    return aux_hist


def _precompute_dataset_strain_bank(
    base: dismech_jax.Rod,
    aux0,
    dataset: TestCase,
) -> jax.Array:
    """
    Precompute all del_strains from dataset trajectories once.

    Returns shape:
        (n_samples, 5)
    where n_samples = n_traj * n_lambda * n_triplets
    """
    def one_traj(qs_traj):
        aux_hist = _build_aux_history_for_qs(aux0, qs_traj)
        del_strain_hist = base.get_del_strain_history(qs_traj, aux_hist)
        return del_strain_hist.reshape(-1, del_strain_hist.shape[-1])

    strain_blocks = jax.vmap(one_traj)(dataset.qs)
    return strain_blocks.reshape(-1, strain_blocks.shape[-1])


def _spectral_loss_from_bank(
    model: eqx.Module,
    strain_bank: jax.Array,
) -> jax.Array:
    penalties = jax.vmap(
        lambda ds: spectral_spread_penalty(model.get_K_matrix(ds))
    )(strain_bank)
    return jnp.mean(penalties)


def _trajectory_data_loss(
    model: eqx.Module,
    base: dismech_jax.Rod,
    aux,
    idx_b: jax.Array,
    xb_c: jax.Array,
    xb_m: jax.Array,
    lambdas: jax.Array,
    truth_qs: jax.Array,
) -> jax.Array:
    bc = dismech_jax.BatchedLinearBC(idx_b=idx_b, xb_c=xb_c, xb_m=xb_m)
    rod = base.with_bc(bc)

    pred_qs = rod.solve(
        model,
        lambdas,
        aux,
        max_dlambda=5e-3,
        iters=5,
        ls_steps=10,
    )

    return jnp.mean(jnp.square(pred_qs - truth_qs))


def _dataset_data_loss(
    model: eqx.Module,
    base: dismech_jax.Rod,
    aux,
    dataset: TestCase,
) -> jax.Array:
    lambdas = _get_dataset_lambdas(dataset)
    idx_b_all = _get_idx_b_all(dataset)

    losses = jax.vmap(
        lambda idx_b, xb_c, xb_m, truth_qs: _trajectory_data_loss(
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

    return jnp.mean(losses)


def _loss_with_aux(
    model: eqx.Module,
    base: dismech_jax.Rod,
    aux,
    dataset: TestCase,
    strain_bank: jax.Array,
    alpha_spec: float,
) -> tuple[jax.Array, tuple[jax.Array, jax.Array]]:
    data_loss = _dataset_data_loss(model, base, aux, dataset)
    spec_loss = _spectral_loss_from_bank(model, strain_bank)
    total_loss = data_loss + alpha_spec * spec_loss
    return total_loss, (data_loss, spec_loss)


def train_model(
    cls: type,
    key: jax.Array = jax.random.PRNGKey(42),
    train_file: str = "train.npz",
    valid_file: str = "valid.npz",
    n_epochs: int = 100,
    lr: float = 1e-2,
    init_K=jnp.array([2.0, 0.01, 0.02]),
    alpha_spec: float = 0.0,
    spec_start_frac: float = 0.2,
    spec_ramp_frac: float = 0.3,
) -> tuple:
    base, aux, der = get_base_rod()
    validate_model(cls, init_K)

    train = TestCase.from_npz(train_file)
    valid = TestCase.from_npz(valid_file)

    # model expects l_k in your current setup
    try:
        model = cls(der_K=init_K, key=key, l_k=base.triplets.l_k[0, 0])
    except TypeError:
        model = cls(der_K=init_K, key=key)

    init_K = model.get_K_entries(jnp.zeros(5))

    # -------- precompute spectral regularizer inputs ONCE --------
    train_strain_bank = _precompute_dataset_strain_bank(base, aux, train)
    valid_strain_bank = _precompute_dataset_strain_bank(base, aux, valid)

    
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

            alpha_now = _alpha_spec_schedule(
                step_idx=step_idx,
                alpha_spec_max=alpha_spec,
                start_frac=spec_start_frac,
                ramp_frac=spec_ramp_frac,
                num_steps=num_steps,
            )

            (train_total_loss, (train_data_loss, train_spec_loss)), grads = eqx.filter_value_and_grad(
                _loss_with_aux,
                has_aux=True,
            )(m, base, aux, train, train_strain_bank, alpha_now)

            updates, next_s = optimizer.update(grads, s, m)
            next_m = eqx.apply_updates(m, updates)

            is_val_step = (step_idx % val_interval) == 0

            valid_total_loss, valid_data_loss, valid_spec_loss = jax.lax.cond(
                is_val_step,
                lambda: (
                    lambda out: (out[0], out[1][0], out[1][1])
                )(_loss_with_aux(next_m, base, aux, valid, valid_strain_bank, alpha_now)),
                lambda: (-1.0, -1.0, -1.0),
            )

            current_lr = schedule(step_idx)

            jax.debug.callback(
                lambda s, lr_now, a_now, ttot, tdata, tspec, vtot, vdata, vspec, K: print(
                    f"Step {s:<4} | LR: {lr_now:<10.3e} | "
                    f"alpha_spec: {a_now:<10.3e} | "
                    f"TrainTot: {ttot:<12.5e} | TrainData: {tdata:<12.5e} | "
                    f"TrainSpec: {tspec:<12.5e} | "
                    f"ValidTot: {vtot:<12.5e} | ValidData: {vdata:<12.5e} | "
                    f"ValidSpec: {vspec:<12.5e} | K: {K}"
                ) if vtot != -1.0 else None,
                step_idx,
                current_lr,
                alpha_now,
                train_total_loss,
                train_data_loss,
                train_spec_loss,
                valid_total_loss,
                valid_data_loss,
                valid_spec_loss,
                next_m.get_K_entries(jnp.zeros(5)),
            )

            return (next_m, next_s), (
                train_total_loss,
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
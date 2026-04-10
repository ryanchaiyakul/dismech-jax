import jax
import jax.numpy as jnp
import equinox as eqx
import optax
import numpy as np
import dismech_jax as djx

from Energy_NN_architectures import ModelParams
# =========================================================
# Dataset (ONLY direct BC)
# =========================================================
class Dataset(eqx.Module):
    qs: jax.Array        # (n_traj, T, dof)
    xb: jax.Array        # (n_traj, T, n_b)
    idx_b: jax.Array     # (n_b,) or (n_traj, n_b)
    lambdas: jax.Array   # (T,)

    @staticmethod
    def load(path):
        data = np.load(path)
        return Dataset(
            qs=jnp.asarray(data["qs"]),
            xb=jnp.asarray(data["xb"]),
            idx_b=jnp.asarray(data["idx_b"]),
            lambdas=jnp.asarray(data["lambdas"]),
        )


# =========================================================
# Base slinky rod (fixed)
# =========================================================
def get_slinky():
    geom = djx.Geometry(0.5, 5e-3)
    mat = djx.Material(1273.52, 1e7)

    rod, aux = djx.Rod.from_geometry(geom, mat, N=3)

    mass = 0.647
    f = jnp.array([
        0, 0, mass/4 * -9.81,
        0, 0, 0,
        mass/2 * -9.81,
        0, 0, 0,
        mass/4 * -9.81,
    ])

    rod = eqx.tree_at(lambda r: r.E_ext, rod, djx.Gravity(f))
    return rod, aux

# =========================================================
# Predict
# =========================================================
def predict(model, base, aux, idx_b, xb, lambdas):
    bc = djx.BatchedDirectBC(idx_b=idx_b, xb=xb, lambdas=lambdas)
    rod = base.with_bc(bc)
    pred = rod.solve(model, lambdas, aux, max_dlambda=5e-3, iters=5, ls_steps=10)
    return pred

# =========================================================
# Loss (MSE over trajectories)
# =========================================================
def traj_loss(model, base, aux, idx_b, xb, lambdas, qs_true):
    bc = djx.DirectBC(idx_b=idx_b, xb=xb, lambdas=lambdas)
    rod = base.with_bc(bc)

    qs_pred = rod.solve(model, lambdas, aux,
                        max_dlambda=5e-3, iters=5, ls_steps=10)

    return jnp.mean((qs_pred - qs_true) ** 2)


def dataset_loss(model, base, aux, data: Dataset):
    n_traj = data.qs.shape[0]

    # handle shared vs per-trajectory idx_b
    if data.idx_b.ndim == 1:
        idx_all = jnp.broadcast_to(data.idx_b, (n_traj, data.idx_b.shape[0]))
    else:
        idx_all = data.idx_b

    losses = jax.vmap(
        lambda ib, xb, qs: traj_loss(model, base, aux, ib, xb, data.lambdas, qs)
    )(idx_all, data.xb, data.qs)

    return jnp.mean(losses)


# =========================================================
# Training
# =========================================================
def train_model(
    model_cls,
    params: ModelParams,
    train_file,
    valid_file,
    n_epochs=100,
    lr=1e-2,
):
    # --- setup ---
    base, aux = get_slinky()
    train = Dataset.load(train_file)
    valid = Dataset.load(valid_file)

    model = model_cls(params)

    schedule = optax.cosine_decay_schedule(
        init_value=lr,
        decay_steps=n_epochs + 1,
        alpha=0.1,
    )

    opt = optax.adam(schedule)
    opt_state = opt.init(model)

    # --- training step ---
    @eqx.filter_jit
    def step(model, opt_state):
        loss, grads = eqx.filter_value_and_grad(
            lambda m: dataset_loss(m, base, aux, train)
        )(model)

        updates, opt_state = opt.update(grads, opt_state)
        model = eqx.apply_updates(model, updates)

        return model, opt_state, loss

    # --- training loop ---
    train_hist = []
    valid_hist = []

    for i in range(n_epochs):
        model, opt_state, train_loss = step(model, opt_state)

        if i % 10 == 0:
            val_loss = dataset_loss(model, base, aux, valid)
            print(f"Epoch {i:03d} | Train: {train_loss:.3e} | Valid: {val_loss:.3e}")
        else:
            val_loss = -1.0
        train_hist.append(train_loss)
        valid_hist.append(val_loss)

    return model, train_hist, valid_hist
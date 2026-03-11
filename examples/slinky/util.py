from __future__ import annotations
from abc import abstractmethod

import jax
import jax.numpy as jnp
import equinox as eqx
import numpy as np
import optax

from jaxtyping import Float, jaxtyped, TypeCheckError
from beartype import beartype
import plotly.graph_objects as go

import dismech_jax


class TestCase(eqx.Module):
    bc: dismech_jax.BC
    qs: jax.Array

    @classmethod
    def from_npz(cls, filename: str) -> TestCase:
        data = jnp.load(filename)
        loaded_bc = dismech_jax.BC(
            idx_b=data["idx_b"], xb_c=data["xb_c"], xb_m=data["xb_m"]
        )
        return cls(bc=loaded_bc, qs=data["qs"])


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


def train_model(cls: type, key: jax.Array = jax.random.PRNGKey(42)) -> None:
    train = TestCase.from_npz("train.npz")
    valid = TestCase.from_npz("valid.npz")

    lambdas = jnp.linspace(0.0, 1.0, 11)
    geom = dismech_jax.Geometry(0.2, 5e-3)
    mat = dismech_jax.Material(1273.52, 1e7)
    temp, aux = dismech_jax.Rod.from_geometry(geom, mat, N=3)

    # Replace F_ext i.e. gravity with full slinky
    MASS = 0.0907407814  # From dismech-python
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
    base = eqx.tree_at(lambda r: r.F_ext, temp, F_new)

    train_rods = base.with_bc(train.bc)
    valid_rods = base.with_bc(valid.bc)

    # TODO: more principled init
    model = cls(der_K=jnp.array([2.0, 2.0, 0.02, 0.02, 0.01]), key=key)
    optimizer = optax.adam(1e-2)
    opt_state = optimizer.init(model)

    def loss(model: eqx.Module, rods: dismech_jax.Rod, truth: jax.Array):
        pred = rods.batch_solve(model, lambdas, aux, max_dt=5e-3, iters=5, ls_steps=10)
        return jnp.mean(jnp.square(pred - truth))

    def eval_loss(model: eqx.Module, rods: dismech_jax.Rod, truth: jax.Array):
        pred = rods.batch_solve(model, lambdas, aux, max_dt=5e-3, iters=5, ls_steps=10)
        return jnp.mean(jnp.square(pred - truth))

    @eqx.filter_jit
    def run_training(model, opt_state, num_steps: int, val_interval: int = 10):
        def train_step(carry, step_idx):
            m, s = carry
            loss_val, grads = eqx.filter_value_and_grad(loss)(m, train_rods, train.qs)
            updates, next_s = optimizer.update(grads, s, m)
            next_m = eqx.apply_updates(m, updates)
            is_val_step = (step_idx % val_interval) == 0
            valid_loss = jax.lax.cond(
                is_val_step,
                lambda: eval_loss(next_m, valid_rods, valid.qs),
                lambda: -1.0,
            )
            jax.debug.callback(
                lambda s, t, v: print(
                    f"Step {s:<4} | Train: {t:<12.5e} | Valid: {v:<12.5e}"
                )
                if v != -1.0
                else None,
                step_idx,
                loss_val,
                valid_loss,
            )

            return (next_m, next_s), (loss_val, valid_loss)

        (final_model, final_state), (train_history, valid_history) = jax.lax.scan(
            train_step, (model, opt_state), jnp.arange(num_steps)
        )
        return final_model, final_state, train_history, valid_history

    model, opt_state, train_history, valid_history = run_training(model, opt_state, 201)

    return model


def animate(qs):
    frames = []
    all_coords = np.vstack([qs[:, 0:3], qs[:, 4:7], qs[:, 8:11]])

    mins = all_coords.min(axis=0)
    maxs = all_coords.max(axis=0)
    center = (mins + maxs) / 2

    # Get max range for cube domain
    max_range = np.max(maxs - mins)
    buffer = max_range * 0.1

    # Fixed limits for all frames
    plot_limit = (max_range / 2) + buffer
    x_range = [center[0] - plot_limit, center[0] + plot_limit]
    y_range = [center[1] - plot_limit, center[1] + plot_limit]
    z_range = [center[2] - plot_limit, center[2] + plot_limit]

    # Build frames
    for t in range(len(qs)):
        row = qs[t]
        q_points = [row[0:3], row[4:7], row[8:11]]
        frames.append(
            go.Frame(
                data=[
                    go.Scatter3d(
                        x=[q_points[0][0], q_points[1][0], q_points[2][0]],
                        y=[q_points[0][1], q_points[1][1], q_points[2][1]],
                        z=[q_points[0][2], q_points[1][2], q_points[2][2]],
                        mode="lines+markers",
                        line=dict(color="black", width=7),
                    ),
                ],
                name=str(t),
            )
        )

    fig = go.Figure(
        data=frames[0].data,
        layout=go.Layout(
            scene=dict(
                xaxis=dict(range=x_range, autorange=False),
                yaxis=dict(range=y_range, autorange=False),
                zaxis=dict(range=z_range, autorange=False),
                aspectmode="cube",  # Forces 1:1:1 scale visuals
            ),
            updatemenus=[
                {
                    "buttons": [
                        {
                            "args": [
                                None,
                                {
                                    "frame": {"duration": 50, "redraw": True},
                                    "fromcurrent": True,
                                },
                            ],
                            "label": "Play",
                            "method": "animate",
                        },
                        {
                            "args": [
                                [None],
                                {
                                    "frame": {"duration": 0, "redraw": True},
                                    "mode": "immediate",
                                },
                            ],
                            "label": "Pause",
                            "method": "animate",
                        },
                    ],
                    "type": "buttons",
                    "showactive": False,
                }
            ],
            sliders=[
                {
                    "steps": [
                        {
                            "args": [
                                [f.name],
                                {
                                    "frame": {"duration": 0, "redraw": True},
                                    "mode": "immediate",
                                },
                            ],
                            "label": f.name,
                            "method": "animate",
                        }
                        for f in frames
                    ]
                }
            ],
        ),
        frames=frames,
    )
    return fig

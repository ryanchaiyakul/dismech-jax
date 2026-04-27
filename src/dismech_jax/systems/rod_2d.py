from __future__ import annotations

import jax
import jax.numpy as jnp
import equinox as eqx
import diffrax

from .bc import AbstractBC
from .energy import AbstractEnergy, Gravity

from ..models import DER2D
from ..stencils import Triplet2D
from ..states import TripletState
from ..params import Geometry, Material
from .system import System
from ..solver import solve


class Rod2D(System[None]):
    triplets: Triplet2D
    mass: jax.Array
    q0: jax.Array
    E_ext: AbstractEnergy
    bc: AbstractBC

    @classmethod
    def from_geometry(
        cls,
        geom: Geometry,
        material: Material,
        N: int = 30,
        bc: AbstractBC = AbstractBC(),
        origin: jax.Array = jnp.array([0.0, 0.0]),
        gravity: jax.Array = jnp.array([0.0, -9.81]),
    ) -> Rod2D:
        if N < 3:
            raise ValueError("Cannot create a rod with less than 3 nodes.")
        if geom.length < 1e-6:
            raise ValueError("Cannot create a rod less than 1 um.")

        q0 = jnp.zeros(2 * N)
        xs = jnp.linspace(0, geom.length, N) + origin[0]
        q0 = q0.at[0::2].set(xs)
        q0 = q0.at[1::2].set(origin[1])
        batch_q = cls._global_q_to_batch_q(q0)

        l_ks = jnp.diff(xs)
        mass = cls._get_mass(geom, material, l_ks)

        N_triplets = batch_q.shape[0]
        batch_l_ks = jax.vmap(lambda i: jax.lax.dynamic_slice(l_ks, (i,), (2,)))(
            jnp.arange(N_triplets)
        )

        triplets = jax.vmap(lambda q, l_k: Triplet2D.init(q, None, l_k=l_k))(
            batch_q, batch_l_ks
        )

        mass_reshaped = mass.reshape(-1, 2)
        F_reshaped = mass_reshaped * gravity
        F_ext = F_reshaped.ravel()

        rod = cls(
            triplets=triplets,
            E_ext=Gravity(F_ext),
            bc=bc,
            q0=q0,
            mass=mass,
        )
        return rod

    def with_bc(self, bc: AbstractBC) -> Rod2D:
        """Get a new Rod PyTree with `self.bc` replaced with passed `bc`.

        Args:
            bc (AbstractBC): Subclassed boundary condition object.

        Returns:
            Rod: rod object.
        """
        return eqx.tree_at(lambda r: r.bc, self, bc)

    def get_DER(self, geom: Geometry, material: Material) -> DER2D:
        """Get a DER energy model for `self.solve(model, ...)`.

        Args:
            geom (Geometry): Geometry object.
            material (Material): Material object.

        Returns:
            DER: Energy model.
        """
        # Assumes nodes are evenly spaced
        return DER2D.from_legacy(self.triplets.l_k[0, 0], geom, material)

    def get_q(self, _lambda: jax.Array, q0: jax.Array) -> jax.Array:
        """Get `q0` with applied with boundary condition at `_lambda`.

        Args:
            _lambda (jax.Array): Lambda.
            q0 (jax.Array): Initial state.

        Returns:
            jax.Array: State with applied boundary condition.
        """
        return self.bc.apply(q0, _lambda)

    def get_E(
        self, _lambda: jax.Array, q: jax.Array, model: eqx.Module, aux: None = None
    ) -> jax.Array:
        """Get scalar energy of state `q` at `_lambda` after applied boundary condition.

        Args:
            _lambda (jax.Array): Lambda.
            q (jax.Array): Initial state.
            model (eqx.Module): Energy model.
            aux (None): None.

        Returns:
            jax.Array: Scalar energy.
        """
        batch_qs = self._global_q_to_batch_q(q)
        E_int = jnp.sum(
            jax.vmap(lambda t, q_loc: t.get_energy(q_loc, model, None))(
                self.triplets, batch_qs
            )
        )
        return E_int + self.E_ext(q, _lambda)

    def get_F(
        self, _lambda: jax.Array, q: jax.Array, model: eqx.Module, aux: None = None
    ) -> jax.Array:
        """Get vector force of state `q` at `_lambda` after applied boundary condition.

        Args:
            _lambda (jax.Array): Lambda.
            q (jax.Array): Initial state.
            model (eqx.Module): Energy model.
            aux (None): None.

        Returns:
            jax.Array: Vector force.
        """
        mask = self.bc.mask(q)
        return mask * jax.grad(self.get_E, 1)(_lambda, q, model, aux)

    def get_H(
        self, _lambda: jax.Array, q: jax.Array, model: eqx.Module, aux: None = None
    ) -> jax.Array:
        """Get square Hessian of state `q` at `_lambda` after applied boundary condition.

        Args:
            _lambda (jax.Array): Lambda.
            q (jax.Array): Initial state.
            model (eqx.Module): Energy model.
            aux (None): None.

        Returns:
            jax.Array: square Hessian.
        """
        mask = self.bc.mask(q)
        batch_qs = self._global_q_to_batch_q(q)
        H_batched = jax.vmap(
            lambda t, q_loc, _aux: jax.hessian(t.get_energy)(q_loc, model, _aux)
        )(self.triplets, batch_qs, aux)

        # Get global indices
        base_idx = jnp.arange(H_batched.shape[0]) * 2  # TODO: parameterize overlap
        offset = jnp.arange(H_batched.shape[1])
        global_idx = base_idx[:, None] + offset[None, :]

        # Row/Column
        rows = global_idx[:, :, None]
        cols = global_idx[:, None, :]

        # Assemble H
        H = jnp.zeros((q.shape[0], q.shape[0]))
        H = H.at[rows, cols].add(H_batched)
        H = H * mask[:, None] * mask[None, :]
        diag_idx = jnp.arange(H.shape[0])
        return H.at[diag_idx, diag_idx].add(1.0 - mask)

    def get_ode_term(self) -> diffrax.ODETerm:
        """Get `diffrax.ODETerm` to solve a ODE.

        Returns:
            diffrax.ODETerm: ODETerm object.
        """

        if self.is_batched(self.in_axes):
            raise NotImplementedError(
                f"get_ode_term: {self} contains a batched BC or E_ext. This is not supported yet!"
            )

        @eqx.filter_jit
        def rhs(_lambda, y, args):
            model, aux = args
            _lambda = jnp.asarray(_lambda)

            # split [q, v]
            n_dofs = self.q0.shape[0]
            q, v = y[:n_dofs], y[n_dofs:]

            # Get fixed DOF
            q_fixed, v_fixed = jax.jvp(
                lambda l: self.bc.apply(q, l), (_lambda,), (jnp.ones_like(_lambda),)
            )

            # update [q, v]
            v = v * self.bc.mask(q) + v_fixed * (1.0 - self.bc.mask(q))
            a = -self.get_F(_lambda, q_fixed, model, aux) / self.mass
            return jnp.concatenate([v, a])

        return diffrax.ODETerm(rhs)

    @eqx.filter_jit
    def solve(
        self,
        model: eqx.Module,
        lambdas: jax.Array,
        aux: TripletState,
        iters: int = 10,
        ls_steps: int = 10,
        c1: float = 1e-4,
        max_dlambda: float = 1e-1,
    ) -> jax.Array:
        """Helper solve function which batches BC and external energy if necessary.

        Args:
            model (eqx.Module): energy model.
            lambdas (jax.Array): lambdas `(N,)`.
            aux (TripletState): Initial triplet state.
            iters (int, optional): Number of newton-raphson iterations. Defaults to 10.
            ls_steps (int, optional): Number of alphas evaluated. Defaults to 10.
            c1 (float, optional): Armijo coefficient. Defaults to 1e-4.
            max_dlambda (float, optional): Maximum lambda step size. Defaults to 1e-1.

        Returns:
            jax.Array: Solved state `(N, # of DOFs)` or `(B, N, # of DOFs)`.
        """
        args = (model, lambdas, self.q0, aux, self, iters, ls_steps, c1, max_dlambda)
        if not self.is_batched(self.in_axes):
            return solve(*args)
        return eqx.filter_vmap(
            solve,
            in_axes=(None, None, None, None, self.in_axes, None, None, None, None),
        )(*args)

    @staticmethod
    def _global_q_to_batch_q(q: jax.Array) -> jax.Array:
        N_triplets = q.shape[0] // 2 - 2
        starts = jnp.arange(N_triplets) * 2
        return jax.vmap(lambda s: jax.lax.dynamic_slice(q, (s,), (6,)))(starts)

    @staticmethod
    def _get_mass(geom: Geometry, material: Material, l_ks: jax.Array) -> jax.Array:
        N = l_ks.shape[0] + 1  # Number of nodes
        mass = jnp.zeros(N * 2)
        A = geom.axs if geom.axs else jnp.pi * geom.r0**2

        # Node contributions
        weights = 0.5 * l_ks[0]
        v_ref_len = jnp.ones(N) * 2 * weights
        v_ref_len = v_ref_len.at[0].set(weights)
        v_ref_len = v_ref_len.at[-1].set(weights)
        dm_nodes = v_ref_len * A * material.density

        node_start_indices = jnp.arange(N) * 4
        for i in range(2):  # Fill x and y
            mass = mass.at[node_start_indices + i].set(dm_nodes)

        return mass

    @property
    def in_axes(self) -> Rod2D:
        return type(self)(
            triplets=None,  # type: ignore
            mass=None,  # type: ignore
            q0=None,  # type: ignore
            E_ext=self.E_ext.in_axes,  # type: ignore
            bc=self.bc.in_axes,  # type: ignore
        )

    @staticmethod
    def is_batched(axes) -> bool:
        """Returns True if any leaf in the PyTree is not None."""
        leaves = jax.tree_util.tree_leaves(axes)
        return any(leaf is not None for leaf in leaves)

import jax
import jax.numpy as jnp
import equinox as eqx
from jaxtyping import Float

from util import TripletModel


# ===================================================================================== #
# Helpers
# ===================================================================================== #
def inv_softplus(y: jax.Array) -> jax.Array:
    return jnp.log(jnp.expm1(y))


def get_reduced_strain_features(del_strain: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array]:
    del_strain = jnp.ravel(del_strain)
    if del_strain.shape[0] < 4:
        raise ValueError(
            f"Expected del_strain to have at least 4 entries, got {del_strain.shape}"
        )
    e0 = del_strain[0]
    e1 = del_strain[1]
    eb = del_strain[3]
    return e0, e1, eb


def get_nn_input_from_strain(del_strain: jax.Array) -> jax.Array:
    e0, e1, eb = get_reduced_strain_features(del_strain)
    return jnp.array([e0**2 + e1**2, eb**2])


def _small_linear(in_features: int, out_features: int, key: jax.Array) -> eqx.nn.Linear:
    layer = eqx.nn.Linear(in_features, out_features, key=key)
    return eqx.tree_at(lambda l: l.weight, layer, layer.weight * 1e-2)


def _init_diag_raw(der_K: jax.Array) -> jax.Array:
    der_K = jnp.ravel(der_K)
    if der_K.shape != (2,):
        raise ValueError(f"Expected der_K shape (2,), got {der_K.shape}")
    return inv_softplus(jnp.maximum(der_K, 1e-8))


def _init_cholesky_raw(der_K: jax.Array) -> jax.Array:
    der_K = jnp.ravel(der_K)
    if der_K.shape != (3,):
        raise ValueError(f"Expected der_K shape (3,), got {der_K.shape}")

    k_ss0, k_sb0, k_bb0 = der_K
    eps = 1e-6

    if (k_ss0 < 0) or (k_bb0 < 0) or (k_ss0 * k_bb0 - 2.0 * k_sb0**2 < 0):
        raise ValueError(
            "Initial [k_ss, k_sb, k_bb] must satisfy PSD condition: "
            "k_ss >= 0, k_bb >= 0, and k_ss*k_bb - 2*k_sb^2 >= 0."
        )

    l11 = jnp.sqrt(jnp.maximum(k_ss0, eps))
    l21 = jnp.sqrt(2.0) * k_sb0 / l11
    rem = k_bb0 - l21**2
    l22 = jnp.sqrt(jnp.maximum(rem, eps))

    return jnp.array([
        inv_softplus(l11 - eps),
        l21,
        inv_softplus(l22 - eps),
    ])


def _vec_to_L(p: jax.Array) -> jax.Array:
    eps = 1e-6
    p = jnp.ravel(p)
    return jnp.array([
        [jax.nn.softplus(p[0]) + eps, 0.0],
        [p[1],                        jax.nn.softplus(p[2]) + eps],
    ])


# ===================================================================================== #
# Scalar nets
# ===================================================================================== #
class ScalarMLP(eqx.Module):
    layers: tuple[eqx.nn.Linear, ...]
    positive_output: bool

    def __init__(
        self,
        in_features: int,
        hidden: tuple[int, ...],
        key: jax.Array,
        *,
        positive_output: bool,
    ):
        sizes = (in_features, *hidden, 1)
        keys = jax.random.split(key, len(sizes) - 1)
        self.layers = tuple(
            _small_linear(sizes[i], sizes[i + 1], keys[i]) for i in range(len(sizes) - 1)
        )
        self.positive_output = positive_output

    def __call__(self, x: jax.Array) -> jax.Array:
        x = jnp.ravel(x)
        for layer in self.layers[:-1]:
            x = jax.nn.softplus(layer(x))
        y = self.layers[-1](x)[0]
        return jax.nn.softplus(y) if self.positive_output else y


class ScalarICNN(eqx.Module):
    x_layers: tuple[eqx.nn.Linear, ...]
    z_layers: tuple[eqx.nn.Linear, ...]
    final_x: eqx.nn.Linear
    final_z: eqx.nn.Linear
    positive_output: bool

    def __init__(
        self,
        in_features: int,
        hidden: tuple[int, ...],
        key: jax.Array,
        *,
        positive_output: bool,
    ):
        if len(hidden) == 0:
            raise ValueError("hidden must contain at least one hidden layer.")

        n_x = len(hidden)
        n_z = len(hidden) - 1
        n_total = n_x + n_z + 2
        keys = jax.random.split(key, n_total)

        k = 0
        x_layers = []
        for h in hidden:
            x_layers.append(_small_linear(in_features, h, keys[k]))
            k += 1

        z_layers = []
        for h_in, h_out in zip(hidden[:-1], hidden[1:]):
            z_layers.append(_small_linear(h_in, h_out, keys[k]))
            k += 1

        self.x_layers = tuple(x_layers)
        self.z_layers = tuple(z_layers)
        self.final_x = _small_linear(in_features, 1, keys[k]); k += 1
        self.final_z = _small_linear(hidden[-1], 1, keys[k])
        self.positive_output = positive_output

    @staticmethod
    def _positive_linear(layer: eqx.nn.Linear, x: jax.Array) -> jax.Array:
        w_pos = jax.nn.softplus(layer.weight)
        return w_pos @ x + layer.bias

    def __call__(self, x: jax.Array) -> jax.Array:
        x = jnp.ravel(x)
        z = jax.nn.softplus(self.x_layers[0](x))
        for x_layer, z_layer in zip(self.x_layers[1:], self.z_layers):
            z = jax.nn.softplus(self._positive_linear(z_layer, z) + x_layer(x))
        y = self._positive_linear(self.final_z, z) + self.final_x(x)
        y = y[0]
        return jax.nn.softplus(y) if self.positive_output else y


class VectorNet(eqx.Module):
    heads: tuple[eqx.Module, ...]

    def __init__(
        self,
        net_type: str,
        in_features: int,
        hidden: tuple[int, ...],
        out_features: int,
        key: jax.Array,
        *,
        positive_output: bool,
    ):
        keys = jax.random.split(key, out_features)
        head_cls = ScalarMLP if net_type == "MLP" else ScalarICNN
        self.heads = tuple(
            head_cls(in_features, hidden, k, positive_output=positive_output) for k in keys
        )

    def __call__(self, x: jax.Array) -> jax.Array:
        x = jnp.ravel(x)
        return jnp.array([head(x) for head in self.heads])


# ===================================================================================== #
# Shared mixins
# ===================================================================================== #
class _DiagonalBase(eqx.Module):
    K0_raw: jax.Array

    def _init_diag(self, der_K: jax.Array):
        self.K0_raw = _init_diag_raw(der_K)

    def get_K0(self) -> jax.Array:
        return jax.nn.softplus(self.K0_raw)

    def _diag_energy_from_entries(self, k_s: jax.Array, k_b: jax.Array, del_strain: jax.Array) -> jax.Array:
        e0, e1, eb = get_reduced_strain_features(del_strain)
        return 0.5 * k_s * (e0**2 + e1**2) + 0.5 * k_b * eb**2

    def _diag_matrix(self, k_s: jax.Array, k_b: jax.Array) -> jax.Array:
        return jnp.array([
            [k_s, 0.0, 0.0],
            [0.0, k_s, 0.0],
            [0.0, 0.0, k_b],
        ])


class _CholeskyBase(eqx.Module):
    K0_raw: jax.Array

    def _init_cholesky(self, der_K: jax.Array):
        self.K0_raw = _init_cholesky_raw(der_K)

    def get_B0(self) -> jax.Array:
        L0 = _vec_to_L(self.K0_raw)
        return L0 @ L0.T

    @staticmethod
    def _B_to_entries(B: jax.Array) -> jax.Array:
        k_ss = B[0, 0]
        k_sb = B[0, 1] / jnp.sqrt(2.0)
        k_bb = B[1, 1]
        return jnp.array([k_ss, k_sb, k_bb])

    @staticmethod
    def _entries_to_matrix(k_ss: jax.Array, k_sb: jax.Array, k_bb: jax.Array) -> jax.Array:
        return jnp.array([
            [k_ss, 0.0,  k_sb],
            [0.0,  k_ss, k_sb],
            [k_sb, k_sb, k_bb],
        ])

    def _chol_energy_from_entries(self, k_ss: jax.Array, k_sb: jax.Array, k_bb: jax.Array, del_strain: jax.Array) -> jax.Array:
        e0, e1, eb = get_reduced_strain_features(del_strain)
        return (
            0.5 * k_ss * (e0**2 + e1**2)
            + k_sb * (e0 + e1) * eb
            + 0.5 * k_bb * eb**2
        )


# ===================================================================================== #
# 1) DiagonalPlusEnergyNN
# ===================================================================================== #
class DiagonalPlusEnergyNN(TripletModel, _DiagonalBase):
    mlp: ScalarMLP
    icnn: ScalarICNN
    which_case: str
    zero_reference: bool

    def __init__(self, der_K: jax.Array, key: jax.Array, hidden: tuple[int, ...] = (10,),
                 which_case: str = "baseline", zero_reference: bool = True):
        k1, k2 = jax.random.split(key, 2)
        self._init_diag(der_K)
        self.mlp = ScalarMLP(2, hidden, k1, positive_output=True)
        self.icnn = ScalarICNN(2, hidden, k2, positive_output=True)
        self.which_case = which_case
        self.zero_reference = zero_reference

    def baseline_energy(self, del_strain: jax.Array) -> jax.Array:
        k_s, k_b = self.get_K0()
        return self._diag_energy_from_entries(k_s, k_b, del_strain)

    def correction_energy(self, del_strain: jax.Array) -> jax.Array:
        x = get_nn_input_from_strain(del_strain)
        net = self.mlp if self.which_case == "MLP" else self.icnn
        out = net(x)
        if self.zero_reference:
            out = out - net(jnp.zeros_like(x))
        return out

    def __call__(self, del_strain: Float[jax.Array, "..."]) -> Float[jax.Array, ""]:
        if self.which_case == "baseline":
            return self.baseline_energy(del_strain)
        if self.which_case in ("MLP", "ICNN"):
            return self.baseline_energy(del_strain) + self.correction_energy(del_strain)
        raise ValueError("which_case must be 'baseline', 'MLP', or 'ICNN'.")


# ===================================================================================== #
# 2) CholeskyPlusEnergyNN
# ===================================================================================== #
class CholeskyPlusEnergyNN(TripletModel, _CholeskyBase):
    mlp: ScalarMLP
    icnn: ScalarICNN
    which_case: str
    zero_reference: bool

    def __init__(self, der_K: jax.Array, key: jax.Array, hidden: tuple[int, ...] = (10,),
                 which_case: str = "baseline", zero_reference: bool = True):
        k1, k2 = jax.random.split(key, 2)
        self._init_cholesky(der_K)
        self.mlp = ScalarMLP(2, hidden, k1, positive_output=True)
        self.icnn = ScalarICNN(2, hidden, k2, positive_output=True)
        self.which_case = which_case
        self.zero_reference = zero_reference

    def get_K_entries(self) -> jax.Array:
        return self._B_to_entries(self.get_B0())

    def get_K_matrix(self) -> jax.Array:
        k_ss, k_sb, k_bb = self.get_K_entries()
        return self._entries_to_matrix(k_ss, k_sb, k_bb)

    def baseline_energy(self, del_strain: jax.Array) -> jax.Array:
        k_ss, k_sb, k_bb = self.get_K_entries()
        return self._chol_energy_from_entries(k_ss, k_sb, k_bb, del_strain)

    def correction_energy(self, del_strain: jax.Array) -> jax.Array:
        x = get_nn_input_from_strain(del_strain)
        net = self.mlp if self.which_case == "MLP" else self.icnn
        out = net(x)
        if self.zero_reference:
            out = out - net(jnp.zeros_like(x))
        return out

    def __call__(self, del_strain: Float[jax.Array, "..."]) -> Float[jax.Array, ""]:
        if self.which_case == "baseline":
            return self.baseline_energy(del_strain)
        if self.which_case in ("MLP", "ICNN"):
            return self.baseline_energy(del_strain) + self.correction_energy(del_strain)
        raise ValueError("which_case must be 'baseline', 'MLP', or 'ICNN'.")


# ===================================================================================== #
# 3) DiagonalPlusStiffnessNN  (PSD)
# ===================================================================================== #
class DiagonalPlusStiffnessNN(TripletModel, _DiagonalBase):
    mlp: VectorNet
    icnn: VectorNet
    which_case: str

    def __init__(self, der_K: jax.Array, key: jax.Array, hidden: tuple[int, ...] = (10,),
                 which_case: str = "MLP"):
        k1, k2 = jax.random.split(key, 2)
        self._init_diag(der_K)
        self.mlp = VectorNet("MLP", 2, hidden, 2, k1, positive_output=False)
        self.icnn = VectorNet("ICNN", 2, hidden, 2, k2, positive_output=False)
        self.which_case = which_case

    def get_K_correction(self, del_strain: jax.Array) -> jax.Array:
        x = get_nn_input_from_strain(del_strain)
        raw = self.mlp(x) if self.which_case == "MLP" else self.icnn(x)
        return jax.nn.softplus(raw)

    def get_K_total(self, del_strain: jax.Array) -> jax.Array:
        return self.get_K0() + self.get_K_correction(del_strain)

    def get_K_matrix(self, del_strain: jax.Array) -> jax.Array:
        k_s, k_b = self.get_K_total(del_strain)
        return self._diag_matrix(k_s, k_b)

    def __call__(self, del_strain: Float[jax.Array, "..."]) -> Float[jax.Array, ""]:
        k_s, k_b = self.get_K_total(del_strain)
        return self._diag_energy_from_entries(k_s, k_b, del_strain)


# ===================================================================================== #
# 4) CholeskyPlusStiffnessNN  (PSD)
# ===================================================================================== #
class CholeskyPlusStiffnessNN(TripletModel, _CholeskyBase):
    mlp: VectorNet
    icnn: VectorNet
    which_case: str

    def __init__(self, der_K: jax.Array, key: jax.Array, hidden: tuple[int, ...] = (10,),
                 which_case: str = "MLP"):
        k1, k2 = jax.random.split(key, 2)
        self._init_cholesky(der_K)
        self.mlp = VectorNet("MLP", 2, hidden, 3, k1, positive_output=False)
        self.icnn = VectorNet("ICNN", 2, hidden, 3, k2, positive_output=False)
        self.which_case = which_case

    def get_Bnn(self, del_strain: jax.Array) -> jax.Array:
        x = get_nn_input_from_strain(del_strain)
        p = self.mlp(x) if self.which_case == "MLP" else self.icnn(x)
        L = _vec_to_L(p)
        return L @ L.T

    def get_B_total(self, del_strain: jax.Array) -> jax.Array:
        return self.get_B0() + self.get_Bnn(del_strain)

    def get_K_entries(self, del_strain: jax.Array) -> jax.Array:
        return self._B_to_entries(self.get_B_total(del_strain))

    def get_K_matrix(self, del_strain: jax.Array) -> jax.Array:
        k_ss, k_sb, k_bb = self.get_K_entries(del_strain)
        return self._entries_to_matrix(k_ss, k_sb, k_bb)

    def __call__(self, del_strain: Float[jax.Array, "..."]) -> Float[jax.Array, ""]:
        k_ss, k_sb, k_bb = self.get_K_entries(del_strain)
        return self._chol_energy_from_entries(k_ss, k_sb, k_bb, del_strain)


# ===================================================================================== #
# 5) CholeskyPlusStiffnessSignedNN  (signed raw-parameter correction, PSD-guaranteed)
# ===================================================================================== #
class CholeskyPlusStiffnessSignedNN(TripletModel, _CholeskyBase):
    mlp: VectorNet
    icnn: VectorNet
    which_case: str

    def __init__(self, der_K: jax.Array, key: jax.Array, hidden: tuple[int, ...] = (10,),
                 which_case: str = "MLP"):
        k1, k2 = jax.random.split(key, 2)
        self._init_cholesky(der_K)
        self.mlp = VectorNet("MLP", 2, hidden, 3, k1, positive_output=False)
        self.icnn = VectorNet("ICNN", 2, hidden, 3, k2, positive_output=False)
        self.which_case = which_case

    def get_B_total(self, del_strain: jax.Array) -> jax.Array:
        x = get_nn_input_from_strain(del_strain)
        dp = self.mlp(x) if self.which_case == "MLP" else self.icnn(x)
        L = _vec_to_L(self.K0_raw + dp)
        return L @ L.T

    def get_K_entries(self, del_strain: jax.Array) -> jax.Array:
        return self._B_to_entries(self.get_B_total(del_strain))

    def get_K_matrix(self, del_strain: jax.Array) -> jax.Array:
        k_ss, k_sb, k_bb = self.get_K_entries(del_strain)
        return self._entries_to_matrix(k_ss, k_sb, k_bb)

    def __call__(self, del_strain: Float[jax.Array, "..."]) -> Float[jax.Array, ""]:
        k_ss, k_sb, k_bb = self.get_K_entries(del_strain)
        return self._chol_energy_from_entries(k_ss, k_sb, k_bb, del_strain)
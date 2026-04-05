import jax
import jax.numpy as jnp
import equinox as eqx

class MLP_Energy(eqx.Module):
    ws: tuple[jax.Array, ...]
    bs: tuple[jax.Array, ...]

    k0_raw: jax.Array
    which_case: str = eqx.field(static=True)

    def __init__(
        self,
        key,
        K_initial=1.0,
        which_case=None,
        hidden_sizes=(10, 10),
        weight_scale=0.01,
    ):
        """
        hidden_sizes:
            hidden layer widths.
            Examples:
                (10,)          -> 1 hidden layer
                (10, 10)       -> 2 hidden layers
                (32, 16, 16)   -> 3 hidden layers
        """
        if len(hidden_sizes) < 1:
            raise ValueError("hidden_sizes must contain at least one hidden layer.")

        hidden_sizes = tuple(hidden_sizes)

        # full layer sizes: input=1 -> hidden_sizes -> output=1
        layer_sizes = (1,) + hidden_sizes + (1,)
        keys = jax.random.split(key, len(layer_sizes) - 1)

        ws = []
        bs = []

        for k, (din, dout) in zip(keys, zip(layer_sizes[:-1], layer_sizes[1:])):
            ws.append(weight_scale * jax.random.normal(k, (dout, din)))
            bs.append(jnp.zeros((dout,)))

        self.ws = tuple(ws)
        self.bs = tuple(bs)

        self.k0_raw = jnp.array(K_initial)
        self.which_case = which_case

    def mlp_energy(self, eps):
        # eps expected shape scalar / (1,) / (1,1)-ish
        x = jnp.atleast_1d(eps).reshape(1)

        # hidden layers
        z = x
        for w, b in zip(self.ws[:-1], self.bs[:-1]):
            z = jax.nn.softplus(w @ z + b)

        # output layer
        out = self.ws[-1] @ z + self.bs[-1]

        # keep correction nonnegative to parallel ICNN behavior
        return jax.nn.softplus(out).squeeze()

    def baseline_energy(self, eps):
        eps_scalar = jnp.squeeze(eps)
        k0 = jax.nn.softplus(self.k0_raw)
        return k0 * eps_scalar**2

    def __call__(self, eps):

        if self.which_case == "baseline":
            return self.baseline_energy(eps)
        elif self.which_case == "mlp":
            return self.mlp_energy(eps)
        else:
            return self.baseline_energy(eps) + self.mlp_energy(eps)


class ICNN_Energy(eqx.Module):
    # Input-to-hidden/output weights
    wxs: tuple[jax.Array, ...]
    # Hidden-to-hidden/output raw weights (will be softplus'd to enforce >= 0)
    wzs_raw: tuple[jax.Array, ...]
    # Biases for hidden/output layers
    bs: tuple[jax.Array, ...]

    k0_raw: jax.Array
    which_case: str = eqx.field(static=True)

    def __init__(
        self,
        key,
        K_initial=1.0,
        which_case=None,
        hidden_sizes=(10, 10),
        weight_scale=0.01,
    ):
        """
        hidden_sizes:
            Sequence of hidden layer widths.
            Example:
                hidden_sizes=(10,)      -> 1 hidden layer
                hidden_sizes=(10, 10)   -> 2 hidden layers
                hidden_sizes=(16, 16, 8)-> 3 hidden layers
        """
        if len(hidden_sizes) < 1:
            raise ValueError("hidden_sizes must contain at least one hidden layer.")

        hidden_sizes = tuple(hidden_sizes)

        # Number of keys needed:
        #   one wx per layer incl. output  => len(hidden_sizes) + 1
        #   one bias per layer incl. output => len(hidden_sizes) + 1
        #   one wz for every layer after first hidden => len(hidden_sizes)
        n_wx = len(hidden_sizes) + 1
        n_b = len(hidden_sizes) + 1
        n_wz = len(hidden_sizes)
        keys = jax.random.split(key, n_wx + n_b + n_wz)

        wx_keys = keys[:n_wx]
        wz_keys = keys[n_wx:n_wx + n_wz]
        b_keys = keys[n_wx + n_wz:]  # not used, but kept for symmetry

        wxs = []
        wzs_raw = []
        bs = []

        in_dim = 1  # eps is scalar strain here

        # First hidden layer: z1 = act(Wx1 x + b1)
        first_hidden = hidden_sizes[0]
        wxs.append(weight_scale * jax.random.normal(wx_keys[0], (first_hidden, in_dim)))
        bs.append(jnp.zeros((first_hidden,)))

        # Remaining hidden layers:
        # z_i = act(Wz_i z_{i-1} + Wx_i x + b_i), with Wz_i >= 0
        for i in range(1, len(hidden_sizes)):
            prev_h = hidden_sizes[i - 1]
            curr_h = hidden_sizes[i]

            wzs_raw.append(
                weight_scale * jax.random.normal(wz_keys[i - 1], (curr_h, prev_h))
            )
            wxs.append(
                weight_scale * jax.random.normal(wx_keys[i], (curr_h, in_dim))
            )
            bs.append(jnp.zeros((curr_h,)))

        # Output layer:
        # out = Wz_out z_last + Wx_out x + b_out, with Wz_out >= 0
        last_hidden = hidden_sizes[-1]
        wzs_raw.append(
            weight_scale * jax.random.normal(wz_keys[-1], (1, last_hidden))
        )
        wxs.append(weight_scale * jax.random.normal(wx_keys[-1], (1, in_dim)))
        bs.append(jnp.zeros((1,)))

        self.wxs = tuple(wxs)
        self.wzs_raw = tuple(wzs_raw)
        self.bs = tuple(bs)

        self.k0_raw = jnp.array(K_initial)
        self.which_case = which_case

    def icnn_energy(self, eps):
        """Scalar-input ICNN energy."""
        x = jnp.atleast_1d(eps).reshape(1)

        # First hidden layer
        z = jax.nn.softplus(self.wxs[0] @ x + self.bs[0])

        # Remaining hidden layers
        # wzs_raw[0] corresponds to second hidden layer connection from z1 -> z2
        for i in range(1, len(self.bs) - 1):
            wz = jax.nn.softplus(self.wzs_raw[i - 1])
            z = jax.nn.softplus(wz @ z + self.wxs[i] @ x + self.bs[i])

        # Output layer
        wz_out = jax.nn.softplus(self.wzs_raw[-1])
        out = wz_out @ z + self.wxs[-1] @ x + self.bs[-1]

        return jax.nn.softplus(out).squeeze()

    def baseline_energy(self, eps):
        eps_scalar = jnp.squeeze(eps)
        k0 = jax.nn.softplus(self.k0_raw)
        return 0.5 * k0 * eps_scalar**2

    def __call__(self, eps):
        if self.which_case == "baseline":
            return self.baseline_energy(eps)
        elif self.which_case == "icnn":
            return self.icnn_energy(eps)
        else:
            return self.baseline_energy(eps) + self.icnn_energy(eps)

# =========================================================
# Stiffness as NN
# =========================================================
class MLP_Stiffness(eqx.Module):
    ws: tuple[jax.Array, ...]
    bs: tuple[jax.Array, ...]

    k0_raw: jax.Array
    which_case: str = eqx.field(static=True)

    def __init__(
        self,
        key,
        K_initial=1.0,
        which_case=None,
        hidden_sizes=(10,),
        weight_scale=0.01,
    ):
        """
        hidden_sizes:
            Example:
                (10,)          -> 1 hidden layer
                (10, 10)       -> 2 hidden layers
                (32, 16, 16)   -> 3 hidden layers
        """
        if len(hidden_sizes) < 1:
            raise ValueError("Need at least one hidden layer")

        layer_sizes = (1,) + tuple(hidden_sizes) + (1,)
        keys = jax.random.split(key, len(layer_sizes) - 1)

        ws = []
        bs = []

        for k, (din, dout) in zip(keys, zip(layer_sizes[:-1], layer_sizes[1:])):
            ws.append(weight_scale * jax.random.normal(k, (dout, din)))
            bs.append(jnp.zeros((dout,)))

        self.ws = tuple(ws)
        self.bs = tuple(bs)

        self.k0_raw = jnp.array(K_initial)
        self.which_case = which_case

    # -------------------------
    # MLP stiffness correction
    # -------------------------
    def K_mlp(self, eps):
        x = jnp.atleast_1d(eps).reshape(1)

        z = x
        for w, b in zip(self.ws[:-1], self.bs[:-1]):
            z = jax.nn.softplus(w @ z + b)

        k_nn = jax.nn.softplus(self.ws[-1] @ z + self.bs[-1]).squeeze()
        return k_nn

    # -------------------------
    # Baseline stiffness
    # -------------------------
    def K_baseline(self):
        return jax.nn.softplus(self.k0_raw)

    # -------------------------
    # Combine
    # -------------------------
    def get_K(self, eps):
        if self.which_case == "only_mlp":
            return self.K_mlp(eps)
        elif self.which_case == "only_baseline":
            return self.K_baseline()
        else:
            return self.K_mlp(eps) + self.K_baseline()

    # -------------------------
    # Energy
    # -------------------------
    def __call__(self, eps):
        eps_scalar = jnp.squeeze(eps)
        return self.get_K(eps) * eps_scalar**2


# =========================================================
# signed correction MLP stiffness
class Signed_MLP_Stiffness(eqx.Module):
    ws: tuple[jax.Array, ...]
    bs: tuple[jax.Array, ...]

    k0_raw: jax.Array
    which_case: str = eqx.field(static=True)

    def __init__(
        self,
        key,
        K_initial=1.0,
        which_case=None,
        hidden_sizes=(10,),
        weight_scale=0.01,
    ):
        """
        hidden_sizes:
            Example:
                (10,)          -> 1 hidden layer
                (10, 10)       -> 2 hidden layers
                (32, 16, 16)   -> 3 hidden layers
        """
        if len(hidden_sizes) < 1:
            raise ValueError("Need at least one hidden layer")

        layer_sizes = (1,) + tuple(hidden_sizes) + (1,)
        keys = jax.random.split(key, len(layer_sizes) - 1)

        ws = []
        bs = []

        for k, (din, dout) in zip(keys, zip(layer_sizes[:-1], layer_sizes[1:])):
            ws.append(weight_scale * jax.random.normal(k, (dout, din)))
            bs.append(jnp.zeros((dout,)))

        self.ws = tuple(ws)
        self.bs = tuple(bs)

        self.k0_raw = jnp.array(K_initial)
        self.which_case = which_case

    # -------------------------
    # MLP stiffness correction
    # -------------------------
    def K_mlp(self, eps):
        x = jnp.atleast_1d(eps).reshape(1)

        z = x
        for w, b in zip(self.ws[:-1], self.bs[:-1]):
            z = jax.nn.softplus(w @ z + b)

        k_nn = (self.ws[-1] @ z + self.bs[-1]).squeeze()
        return k_nn

    # -------------------------
    # Baseline stiffness
    # -------------------------
    def K_baseline(self):
        return jax.nn.softplus(self.k0_raw)

    # -------------------------
    # Combine
    # -------------------------
    def get_K(self, eps):
        if self.which_case == "only_mlp":
            return jax.nn.softplus(self.K_mlp(eps))
        elif self.which_case == "only_baseline":
            return self.K_baseline()
        else:
            return jax.nn.softplus(self.K_mlp(eps) + self.K_baseline())

    # -------------------------
    # Energy
    # -------------------------
    def __call__(self, eps):
        eps_scalar = jnp.squeeze(eps)
        return self.get_K(eps) * eps_scalar**2


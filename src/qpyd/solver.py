import functools
import jax
import jax.numpy as jnp


def fermi(x: jax.Array, kb_T: float) -> jax.Array:
    """
    Calculates the Fermi-Dirac distribution.

    Parameters
    ----------
    x : jax.Array
        Energy difference.
    kb_T : float
        Thermal energy.

    Returns
    -------
    jax.Array
        Occupancy probability bounded between 0.0 and 1.0.
    """
    return jax.nn.sigmoid(- (1.0 / kb_T) * x)


def bose(x: jax.Array, kb_T: float) -> jax.Array:
    """
    Calculates the Bose-Einstein distribution.

    Parameters
    ----------
    x : jax.Array
        Energy difference.
    kb_T : float
        Thermal energy.

    Returns
    -------
    jax.Array
        Occupancy probability tensor.
    """
    beta = 1.0 / kb_T
    denom = jnp.abs(jnp.expm1(beta * x))
    return jnp.where(denom == 0, 0.0, 1.0 / denom)


@functools.partial(jax.jit, static_argnames=('sector_bounds',))
def solve_chunk(
    Vext_chunk: jax.Array, 
    muL_chunk: jax.Array, 
    muR_chunk: jax.Array, 
    dE: jax.Array, 
    dNe: jax.Array, 
    rate_L: jax.Array, 
    rate_R: jax.Array, 
    rate_rel: jax.Array, 
    kb_T: float, 
    sector_bounds: tuple[int, ...]
) -> tuple[jax.Array, jax.Array]:
    """
    Calculates the steady-state probabilities and current for a batch of points.

    Parameters
    ----------
    Vext_chunk : jax.Array
        External voltage grid chunk.
    muL_chunk : jax.Array
        Left lead chemical potential chunk.
    muR_chunk : jax.Array
        Right lead chemical potential chunk.
    dE : jax.Array
        Energy difference matrix.
    dNe : jax.Array
        Particle number difference matrix.
    rate_L : jax.Array
        Left lead transition rate matrix.
    rate_R : jax.Array
        Right lead transition rate matrix.
    rate_rel : jax.Array
        Relaxation rate matrix.
    kb_T : float
        Thermal energy.
    sector_bounds : tuple
        Indices dividing charge sectors.

    Returns
    -------
    tuple of jax.Array
        Steady-state probabilities and current values.
    """
    def single_point(Vext, muL, muR):
        arg_L = dE - dNe * Vext - dNe * muL
        arg_R = dE - dNe * Vext - dNe * muR

        fL = jax.nn.sigmoid(- (1.0 / kb_T) * arg_L)
        fR = jax.nn.sigmoid(- (1.0 / kb_T) * arg_R)

        rate_L_eff = rate_L * fL
        rate_R_eff = rate_R * fR
        rate_total = rate_L_eff + rate_R_eff + rate_rel

        # Add stabilizer to prevent singular matrices
        rate_total += 1e-12

        # Enforce probability conservation on the diagonal
        rate_total = rate_total.at[jnp.diag_indices_from(rate_total)].set(0.0)
        col_sums = jnp.sum(rate_total, axis=0)
        rate_total = rate_total.at[jnp.diag_indices_from(rate_total)].set(-col_sums)

        # Solve square matrix (sum of probabilities = 1)
        N = rate_total.shape[0]
        rate_total_square = rate_total.at[-1, :].set(1.0)
        b = jnp.zeros(N).at[-1].set(1.0)
        P = jnp.linalg.solve(rate_total_square, b)

        curr = 0.0
        for s in range(len(sector_bounds) - 2):
            low_start, low_end = sector_bounds[s], sector_bounds[s+1]
            high_start, high_end = sector_bounds[s+1], sector_bounds[s+2]

            rate_high_low = rate_L_eff[high_start:high_end, low_start:low_end]
            rate_low_high = rate_L_eff[low_start:low_end, high_start:high_end]

            P_low = P[low_start:low_end]
            P_high = P[high_start:high_end]

            inflow = jnp.sum(rate_high_low @ P_low)
            outflow = jnp.sum(rate_low_high @ P_high)

            curr += (inflow - outflow)

        return P, -curr

    return jax.vmap(single_point)(Vext_chunk, muL_chunk, muR_chunk)
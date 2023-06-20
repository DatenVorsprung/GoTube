# transformation between polar and cartesian coordinates

import numpy as np
import jax.numpy as jnp
from jax import jit
import dynamics

# initialize random polar coordinates with dimension dim
_rng = np.random.RandomState(12937)


def uniform(start, end, dim, fixed_seed):
    if fixed_seed:
        global _rng
        return _rng.uniform(start, end, dim)
    else:
        return np.random.uniform(start, end, dim)


def init_random_phi(dim, samples=1, num_gpus=1, fixed_seed=False):
    phi = uniform(0, jnp.pi, samples * (dim - 2), fixed_seed)
    phi = jnp.append(phi, uniform(0, 2 * jnp.pi, samples, fixed_seed))
    phi = jnp.reshape(phi, (num_gpus, samples // num_gpus, dim - 1), order="F")

    return phi


@jit
def polar2cart(rad, phi):
    return rad * polar2cart_no_rad(phi)


def polar2cart_euclidean_metric(rad, phis, A0inv):
    return rad * jnp.matmul(A0inv, polar2cart_no_rad(phis))


def polar2cart_no_rad(phi):
    return dynamics.polar2cart_no_rad(phi)

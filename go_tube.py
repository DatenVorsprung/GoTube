# Algorithms of GoTube paper for safety region, probability and stoch. optimization

import jax.numpy as jnp
from jax import vmap, pmap
import polar_coordinates as pol
from jax.numpy.linalg import svd
import jax.scipy.special as sc
import time
from performance_log import log_stat
from timer import Timer
from scipy.stats import genextreme, kstest
import gc


# using expected difference quotient of center lipschitz constant
def get_safety_region_radius(model, dist, dist_best, lip, lip_mean_diff):
    safety_radius = -lip + jnp.sqrt(lip ** 2 + 4 * lip_mean_diff * (model.mu * dist_best - dist))
    safety_radius = safety_radius / (2 * lip_mean_diff)
    safety_radius = safety_radius * (dist > 0) * (lip_mean_diff > 0)

    safety_radius = jnp.minimum(safety_radius, 2 * model.rad_t0)

    return safety_radius


def compute_maximum_singular_value(A1, A0inv, F):
    F_metric = jnp.matmul(A1, F)
    F_metric = jnp.matmul(F_metric, A0inv)
    _, sf, _ = svd(F_metric)
    max_sf = jnp.max(sf)

    return max_sf


def get_angle_of_cap(model, radius):
    radius = jnp.minimum(radius, 2 * model.rad_t0)
    return 2 * jnp.arcsin(0.5 * radius / model.rad_t0)


def get_probability_of_cap(model, radius):
    with Timer('get angle of cap'):
        angle = get_angle_of_cap(model, radius)
    with Timer('get probability of cap'):
        a = 0.5 * (model.model.dim - 1)
        b = 0.5
        x = jnp.sin(angle) ** 2
        betainc_angle = 0.5 * sc.betainc(a, b, x)

        # formula is only for the smaller cap with angle <= pi/2, sinus is symmetric => thus use 1-area otherwise
        betainc_angle = jnp.where(angle > 0.5 * jnp.pi, 1 - betainc_angle, betainc_angle)

    return betainc_angle


def get_probability_not_in_cap(model, radius):
    return 1 - get_probability_of_cap(model, radius)


def get_probability_none_in_cap(model, radius_points):
    return jnp.prod(get_probability_not_in_cap(model, radius_points))


#  probability calculation using http://docsdrive.com/pdfs/ansinet/ajms/2011/66-70.pdf (equation 1
#  page 68) and the normalized incomplete Beta-Function in scipy (
#  https://scipy.github.io/devdocs/generated/scipy.special.betainc.html#scipy.special.betainc) - Only use the
#  random sampled points for probability construction
#  use also the discarded points and create balls around them
def get_probability(model, radius_points):
    return jnp.sqrt(1-model.gamma) * (1 - get_probability_none_in_cap(model, radius_points))


def compute_delta_lipschitz(y_jax, fy_jax, axis, gamma):
    gamma_hat = 1 - jnp.sqrt(1 - gamma)
    diff_quotients = get_diff_quotient_pairwise(y_jax, fy_jax, axis)
    sample_size = diff_quotients.size
    number_of_elements_for_maximum = round(sample_size ** (1 / 4))  # m in Lemma 1, Theorem 2 and throughout paper

    sample_size_dividable = sample_size - sample_size % number_of_elements_for_maximum

    diff_quotients_samples = diff_quotients[:sample_size_dividable].reshape(-1, number_of_elements_for_maximum)
    max_quotients = jnp.nanmax(diff_quotients_samples, axis=1)
    number_of_maxima = max_quotients.size  # n in Lemma 1
    alpha = min(gamma_hat, 0.5)
    epsilon = jnp.sqrt(jnp.log(1 / alpha) / (2 * number_of_maxima))

    c, loc, scale = genextreme.fit(max_quotients)
    rv_genextreme = genextreme(c, loc, scale)

    D_minus = kstest(max_quotients, rv_genextreme.cdf, alternative='less').statistic

    max_quantile = 0.9999

    # # with generalized extreme value distribution
    prob_quantile = min(1 - gamma_hat, max_quantile - epsilon - D_minus)
    delta_lipschitz = rv_genextreme.ppf([prob_quantile + epsilon + D_minus])  # transformation of Eq. (S14)
    prob_bound_lipschitz = (1 - gamma_hat) * prob_quantile

    # without generalized extreme value distribution
    # max_quotients = jnp.sort(max_quotients)
    # prob_quantile = min(1 - gamma_hat, max_quantile - epsilon)
    # delta_lipschitz = max_quotients[int(jnp.floor((prob_quantile + epsilon) * number_of_maxima))]  # transformation of Eq. (S14)
    # prob_bound_lipschitz = (1 - gamma_hat) * prob_quantile

    return delta_lipschitz, prob_bound_lipschitz


def get_diff_quotient_pairwise(x, fx, axis):
    # reshape to get samples as first index and remove gpu dimension
    x = jnp.reshape(x, (-1, x.shape[2]))
    fx = jnp.reshape(fx, fx.size)

    samples = int(jnp.floor(x.shape[0] / 2))
    x1 = x[::2][:samples]
    x2 = x[1::2][:samples]
    fx1 = fx[::2][:samples]
    fx2 = fx[1::2][:samples]
    distance = jnp.linalg.norm(x1 - x2, axis=axis)
    diff_quotients = abs(fx1 - fx2) / distance * (distance > 0)
    return diff_quotients


def get_diff_quotient(x, fx, y_jax, fy_jax, axis):
    distance = jnp.linalg.norm(x - y_jax, axis=axis)
    diff_quotients = abs(fx - fy_jax) / distance * (distance > 0)
    return diff_quotients


def get_diff_quotient_vmap(x_jax, fx_jax, y_jax, fy_jax, axis):
    return vmap(get_diff_quotient, in_axes=(0, 0, None, None, None))(x_jax, fx_jax, y_jax, fy_jax, axis)


def optimize(model, initial_points, points=None, gradients=None):
    start_time = time.time()

    prob = None

    if points is None or gradients is None:
        previous_samples = 0
        phis = pol.init_random_phi(model.model.dim, model.batch, model.num_gpus, model.fixed_seed)
        points, gradients, neg_dists, initial_points = model.aug_integrator_neg_dist(phis)
        dists = -neg_dists
        del neg_dists
        del phis
        gc.collect()
    else:
        previous_samples = points.shape[1]
        with Timer('integrate random points and gradients - one step'):
            points, gradients, dists = model.one_step_aug_integrator_dist(
                points, gradients
            )

    first_iteration = True

    while prob is None or prob < 1 - model.gamma:

        if not first_iteration:
            with Timer('sample phis'):
                phis = pol.init_random_phi(model.model.dim, model.batch, model.num_gpus, model.fixed_seed)
            with Timer('compute first integration step and dist'):
                new_points, new_gradients, new_neg_dists, new_initial_points = model.aug_integrator_neg_dist(phis)
                new_dists = -new_neg_dists
                del new_neg_dists
                del phis
                gc.collect()

            with Timer('concatenate new points to tensors'):
                points = jnp.concatenate((points, new_points), axis=1)
                gradients = jnp.concatenate((gradients, new_gradients), axis=1)
                dists = jnp.concatenate((dists, new_dists), axis=1)
                initial_points = jnp.concatenate((initial_points, new_initial_points), axis=1)
                del new_points
                del new_gradients
                del new_dists
                del new_initial_points
                gc.collect()

        with Timer('compute best dist'):
            dist_best = dists.max()

        with Timer('compute lipschitz'):
            # compute maximum singular values of all new gradient matrices
            lipschitz = pmap(vmap(compute_maximum_singular_value, in_axes=(None, None, 0)), in_axes=(None, None, 0))(model.A1, model.A0inv, gradients)

        with Timer('compute expected local lipschitz'):
            sample_size = points.shape[0]

            # compute expected value of delta lipschitz
            dimension_axis = 1

            gamma_hat = 1 - jnp.sqrt(1 - model.gamma)

            delta_lipschitz, prob_bound_lipschitz = compute_delta_lipschitz(
                initial_points[:sample_size],
                lipschitz[:sample_size],
                dimension_axis,
                gamma_hat
            )

        with Timer('get safety region radii'):
            safety_region_radii = get_safety_region_radius(
                model, dists, dist_best, lipschitz, delta_lipschitz
            )

        with Timer('compute probability'):
            prob = get_probability(model, safety_region_radii)

            del delta_lipschitz
            del lipschitz
            del safety_region_radii
            gc.collect()

        print(f"Current probability coverage is {100.0 * prob:0.3f}% using {model.num_gpus * points.shape[1]} points")

        first_iteration = False

    new_samples = model.num_gpus * points.shape[1] - previous_samples

    print("Probability reached given value!")
    print(
        f"Visited {new_samples} new points in {time.time() - start_time:0.2f} seconds."
    )

    dist_with_safety_mu = model.mu * dist_best

    if model.profile:
        # If profiling is enabled, log some statistics about the optimization process
        stat_dict = {
            "loop_time": time.time() - start_time,
            "new_points": int(new_samples),
            "total_points": int(previous_samples + new_samples),
            "prob": float(prob),
            "dist_best": float(dist_best),
            "radius": float(dist_with_safety_mu),
        }
        log_stat(stat_dict)

    return dist_with_safety_mu, prob, initial_points, points, gradients

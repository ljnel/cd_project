import itertools

import numpy as np
from scipy import integrate
from scipy.special import comb, factorial2
from scipy.special import gamma as gamma_func


def iterate_sub_indices(gamma):
    """Helper to iterate through multi-indices k <= gamma"""
    ranges = [range(g + 1) for g in gamma]
    for k in itertools.product(*ranges):
        yield np.array(k)


def gaussian_moment(gamma, mu, sigma):
    """
    Calculates E[x^gamma] where x ~ N(mu, sigma^2 * I).
    This function directly implements the formula:
    E[x^gamma] = sum_{k<=gamma, k_j even} [ binom(gamma,k) * mu^(gamma-k) * E[y^k] ]
    where E[y^k] for y ~ N(0, sigma^2*I) is prod(sigma^k_j * (k_j-1)!!)
    """
    total_moment = 0
    mu = np.asarray(mu)
    gamma = np.asarray(gamma)

    for k in iterate_sub_indices(gamma):
        # The moment of a centered Gaussian is non-zero only if all exponents are even
        if all(val % 2 == 0 for val in k):
            # Binomial coefficient term: C(gamma_1, k_1) * C(gamma_2, k_2) * ...
            binom_prod = np.prod([comb(g, ki) for g, ki in zip(gamma, k, strict=False)])

            # Term for the mean: mu_1^(gamma_1-k_1) * mu_2^(gamma_2-k_2) * ...
            mu_term = np.prod(mu ** (gamma - k))

            # Term for the centered moment E[y^k]
            sigma_term = 1.0
            for ki in k[k > 0]:
                sigma_term *= (sigma**ki) * factorial2(
                    ki - 1
                )  # (k-1)!! is the double factorial

            total_moment += binom_prod * mu_term * sigma_term

    return total_moment


def integral_monomial_centered_ball(k, epsilon, n_dim):
    """
    Calculates Integral_{B(0,eps)} y^k dy.
    The integral is non-zero only if all components of k are even.
    If k=2m, the formula is:
    eps^(n+|2m|) * [ prod_{j=1..n} Gamma(m_j + 1/2) ] / Gamma(n/2 + |m| + 1)
    """
    k = np.asarray(k)
    # Integral is 0 if any exponent is odd due to symmetry
    if not all(val % 2 == 0 for val in k):
        return 0.0

    m = k / 2
    abs_m = np.sum(m)
    abs_2m = np.sum(k)

    numerator = np.prod([gamma_func(mj + 0.5) for mj in m])
    denominator = gamma_func(n_dim / 2 + abs_m + 1)

    return (epsilon ** (n_dim + abs_2m)) * (numerator / denominator)


def ball_moment(gamma, center, epsilon):
    """
    Calculates Integral_{B(center,eps)} x^gamma dx using multinomial expansion.
    The formula is:
    Integral = sum_{k<=gamma} [ binom(gamma,k) * center^(gamma-k) * Integral_{B(0,eps)} y^k dy ]
    """
    n_dim = len(center)
    center = np.asarray(center)
    gamma = np.asarray(gamma)

    total_integral = 0
    for k in iterate_sub_indices(gamma):
        binom_prod = np.prod([comb(g, ki) for g, ki in zip(gamma, k, strict=False)])
        center_term = np.prod(center ** (gamma - k))
        integral_term = integral_monomial_centered_ball(k, epsilon, n_dim)

        total_integral += binom_prod * center_term * integral_term
    return total_integral


def ball_moment_numeric(gamma, p, epsilon):
    def monomial_integrand(gamma):
        """Defines the integrand for a given monomial"""

        def integrand(*args):
            val = 1.0
            for i in range(len(args)):
                val *= args[i] ** gamma[i]
            return val

        return integrand

    # (x-xi)**2 + (y-yi)**2 = eps**2
    # y = y_i +- sqrt(eps**2 - (x-xi)**2)
    def limits_y(p):
        return [p[1] - epsilon, p[1] + epsilon]

    def limits_x_given_y(p):
        return lambda y: [
            float(p[0] - np.sqrt(max(1e-10, epsilon**2 - (y - p[1]) ** 2))),
            float(p[0] + np.sqrt(max(1e-10, epsilon**2 - (y - p[1]) ** 2))),
        ]

    # note that the order here is crucial. see nquad documentation.
    if len(p) == 2:
        ranges = [
            limits_x_given_y(p),  # calcualte x as a function of y.
            limits_y(p),  # choose y in the right window
        ]
        integral, _ = integrate.nquad(monomial_integrand(gamma), ranges)  # type: ignore
    else:
        ranges = [p[0] - epsilon, p[0] + epsilon]
        integral, _ = integrate.quad(monomial_integrand(gamma), *ranges)  # type: ignore

    return integral


def gaussian_moment_numeric(gamma, p, epsilon, range_mult=5):
    def norm(x):
        n = len(x)
        z = epsilon**n * (2 * np.pi) ** (n / 2)  # normalization
        return np.exp(-0.5 * np.linalg.norm(x - p) ** 2 / epsilon**2) / z

    def monomial_integrand(gamma):
        """Defines the integrand for a given monomial"""

        def integrand(*args):
            val = 1.0
            for i in range(len(args)):
                val *= args[i] ** gamma[i]
            return val * norm(args)

        return integrand

    # note that the order here is crucial. see nquad documentation.
    ranges = [
        [p[i] - range_mult * epsilon, p[i] + range_mult * epsilon]
        for i in range(len(p))
    ]

    integral, _ = integrate.nquad(monomial_integrand(gamma), ranges)  # type: ignore
    return integral


def create_smooth_ball(points, degree, epsilon, analytic=True):
    import itertools

    from cd.algs.bases.poly_basis import total_degree_index_set as get_monomial_basis

    n_dim = points.shape[1]
    monomial_basis = get_monomial_basis(n_dim, degree)
    basis_size = len(monomial_basis)

    M_ball_analytic = np.zeros((basis_size, basis_size))

    ball_volume = (np.pi ** (n_dim / 2) / gamma_func(n_dim / 2 + 1)) * (epsilon**n_dim)
    for i, j in itertools.combinations_with_replacement(range(basis_size), 2):
        alpha = monomial_basis[i]
        beta = monomial_basis[j]
        gamma = tuple(a + b for a, b in zip(alpha, beta, strict=False))

        if analytic:
            total_ball_moment = sum(ball_moment(gamma, p, epsilon) for p in points)
        else:
            total_ball_moment = sum(
                ball_moment_numeric(gamma, p, epsilon) for p in points
            )
        total_ball_moment /= ball_volume
        M_ball_analytic[i, j] = total_ball_moment
        M_ball_analytic[j, i] = total_ball_moment
    return M_ball_analytic


def create_smooth_gauss(points, degree, epsilon, analytic=True):
    from cd.algs.bases.poly_basis import total_degree_index_set as get_monomial_basis

    n_dim = points.shape[1]
    monomial_basis = get_monomial_basis(n_dim, degree)
    basis_size = len(monomial_basis)

    M_gauss_analytic = np.zeros((basis_size, basis_size))
    for i, j in itertools.combinations_with_replacement(range(basis_size), 2):
        alpha = monomial_basis[i]
        beta = monomial_basis[j]
        gamma = tuple(a + b for a, b in zip(alpha, beta, strict=False))

        if analytic:
            total_gauss_moment = sum(gaussian_moment(gamma, p, epsilon) for p in points)
        else:
            total_gauss_moment = sum(
                gaussian_moment_numeric(gamma, p, epsilon) for p in points
            )

        M_gauss_analytic[i, j] = total_gauss_moment
        M_gauss_analytic[j, i] = total_gauss_moment
    return M_gauss_analytic

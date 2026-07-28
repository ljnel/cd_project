import numpy as np

from cd.algs.cd_smoothing import (
    ball_moment,
    ball_moment_numeric,
    gaussian_moment,
    gaussian_moment_numeric,
)

DIM = 1


def test_integral_ball():
    """This is not normalized, it should equal the volume of the ball"""
    p = np.zeros(DIM, dtype=float)
    epsilon = 1.0
    gamma = np.zeros(DIM, dtype=int)
    area_simple = np.pi * epsilon**2 if DIM == 2 else epsilon * 2
    area_numeric = ball_moment_numeric(gamma, p, epsilon)
    np.testing.assert_allclose(area_simple, area_numeric, rtol=1e-5)  # type: ignore

    area_analytic = ball_moment(gamma, p, epsilon)
    np.testing.assert_allclose(area_simple, area_analytic, rtol=1e-5)  # type: ignore


def test_integral_gauss():
    """This is normalized, it should equal 1.0"""
    p = np.zeros(DIM, dtype=float)
    epsilon = 1.0
    gamma = np.zeros(DIM, dtype=int)

    integral_numeric = gaussian_moment_numeric(gamma, p, epsilon, range_mult=10)
    np.testing.assert_allclose(1.0, integral_numeric, rtol=1e-5)  # type: ignore

    integral_analytic = gaussian_moment(gamma, p, epsilon)
    np.testing.assert_allclose(1.0, integral_analytic, rtol=1e-5)  # type: ignore


def test_all_same():
    """Test that the numeric and analytic versions give the same result"""

    # for Gaussian
    p = np.zeros(DIM, dtype=float)
    epsilon = 1.0
    gamma = np.zeros(DIM, dtype=int)

    integral_numeric = gaussian_moment_numeric(gamma, p, epsilon)
    integral_analytic = gaussian_moment(gamma, p, epsilon)
    np.testing.assert_allclose(
        integral_numeric, integral_analytic, rtol=1e-3  # type: ignore
    )

    # for Ball
    integral_numeric = ball_moment_numeric(gamma, p, epsilon)
    integral_analytic = ball_moment(gamma, p, epsilon)
    np.testing.assert_allclose(
        integral_numeric, integral_analytic, rtol=1e-3  # type: ignore
    )


if __name__ == "__main__":
    test_integral_ball()
    test_integral_gauss()
    test_all_same()
    print("All tests passed.")

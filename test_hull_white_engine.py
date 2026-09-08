import unittest

import numpy as np

import hull_white_engine as hw


class HullWhiteTests(unittest.TestCase):
    def setUp(self):
        self.maturities = np.array([0.25, 0.5, 1, 2, 3, 5, 7, 10, 20, 30.0])
        self.rate = 0.04
        self.curve = hw.CurveData(
            self.maturities,
            np.full_like(self.maturities, self.rate),
            curve_kind="zero",
            quote_convention="continuous",
        )

    def test_flat_zero_curve(self):
        engine = hw.HullWhiteCurveEngine(self.curve, smooth_curve=False)
        self.assertAlmostEqual(engine.initial_short_rate(), self.rate, places=12)
        self.assertLess(engine.initial_curve_repricing_error(), 1e-12)

    def test_flat_par_bootstrap(self):
        mats = np.array([1/12, 2/12, 3/12, 4/12, 0.5, 1, 2, 3, 5, 7, 10, 20, 30.0])
        curve = hw.CurveData(
            mats,
            np.full_like(mats, self.rate),
            curve_kind="par",
            quote_convention="semiannual",
        )
        engine = hw.HullWhiteCurveEngine(curve, smooth_curve=False)
        expected = (1 + self.rate / 2) ** (-2 * engine.maturities)
        np.testing.assert_allclose(engine.discount_factors, expected, atol=1e-12, rtol=0)

    def test_theta_flat_curve(self):
        a, sigma = 0.15, 0.01
        engine = hw.HullWhiteCurveEngine(self.curve, a, sigma, smooth_curve=False)
        t = 1.0
        expected = a * self.rate + sigma**2/(2*a)*(1-np.exp(-2*a*t))
        self.assertAlmostEqual(float(engine.hull_white_theta(t)), expected, places=10)

    def test_simulation_matches_model_moments(self):
        engine = hw.HullWhiteCurveEngine(self.curve, 0.15, 0.01, smooth_curve=False)
        config = hw.SimulationConfig(n_paths=40000, horizon_years=1, dt=1/12, seed=7)
        _, paths = engine.simulate_short_rate_paths(config)
        _, means, stds = engine.analytic_short_rate_moments(config)
        self.assertLess(abs(paths[:, -1].mean() - means[-1]), 2e-4)
        self.assertLess(abs(paths[:, -1].std(ddof=1) - stds[-1]), 2e-4)

    def test_parallel_shock(self):
        shocked = hw.apply_scenario_shock(self.curve, 50)
        np.testing.assert_allclose(shocked.yields, self.curve.yields + 0.005)


if __name__ == "__main__":
    unittest.main(verbosity=2)

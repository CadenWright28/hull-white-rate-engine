from __future__ import annotations

import unittest

import numpy as np

import hull_white_engine as hw


class HullWhiteEngineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.maturities = np.array([0.25, 0.5, 1, 2, 3, 5, 7, 10, 20, 30.0])
        self.flat_rate = 0.04
        self.zero_curve = hw.CurveData(
            self.maturities,
            np.full_like(self.maturities, self.flat_rate),
            curve_date="test",
            source="synthetic zero",
            curve_kind="zero",
            quote_convention="continuous",
        )

    def test_initial_curve_repricing(self) -> None:
        engine = hw.HullWhiteCurveEngine(self.zero_curve, 0.15, 0.01)
        self.assertLess(engine.initial_curve_repricing_error(), 1e-12)
        self.assertAlmostEqual(engine.initial_short_rate(), self.flat_rate, places=12)

    def test_flat_par_bootstrap(self) -> None:
        maturities = np.array([
            1 / 12, 1.5 / 12, 2 / 12, 3 / 12, 4 / 12,
            0.5, 1, 2, 3, 5, 7, 10, 20, 30.0,
        ])
        curve = hw.CurveData(
            maturities,
            np.full_like(maturities, self.flat_rate),
            curve_kind="par",
            quote_convention="bond_equivalent_semiannual",
        )
        engine = hw.HullWhiteCurveEngine(curve, 0.15, 0.01)
        expected = (1.0 + self.flat_rate / 2.0) ** (-2.0 * engine.maturities)
        np.testing.assert_allclose(engine.discount_factors, expected, atol=1e-12, rtol=0)

    def test_theta_identity_on_flat_curve(self) -> None:
        a, sigma = 0.15, 0.01
        engine = hw.HullWhiteCurveEngine(self.zero_curve, a, sigma)
        for time in (0.0, 0.5, 1.0, 5.0):
            expected = (
                a * self.flat_rate
                + sigma**2 / (2.0 * a) * (1.0 - np.exp(-2.0 * a * time))
            )
            self.assertAlmostEqual(float(engine.hull_white_theta(time, False)), expected, places=10)
            self.assertAlmostEqual(float(engine.hull_white_theta(time, True)), expected, places=10)

    def test_exact_transition_moments(self) -> None:
        engine = hw.HullWhiteCurveEngine(self.zero_curve, 0.15, 0.01)
        config = hw.SimulationConfig(50_000, 1.0, 1 / 12, seed=7)
        times, paths = engine.simulate_short_rate_paths(config)
        summary = engine.simulation_summary(times, paths)
        self.assertLess(abs(summary["terminal_mean_error"]), 1.5e-4)
        self.assertLess(abs(summary["terminal_std_error"]), 1.5e-4)

    def test_regularized_curve_is_smooth_and_close(self) -> None:
        maturities = np.array([
            1 / 12, 1.5 / 12, 2 / 12, 3 / 12, 4 / 12,
            0.5, 1, 2, 3, 5, 7, 10, 20, 30.0,
        ])
        yields = np.array([
            0.0376, 0.0383, 0.0390, 0.0396, 0.0402, 0.0405,
            0.0420, 0.0425, 0.0430, 0.0445, 0.0465, 0.0470,
            0.0518, 0.0518,
        ])
        curve = hw.CurveData(
            maturities, yields, curve_kind="par",
            quote_convention="bond_equivalent_semiannual",
        )
        engine = hw.HullWhiteCurveEngine(curve, 0.15, 0.01)
        diagnostics = engine.curve_diagnostics()
        self.assertLessEqual(
            diagnostics["drift_curve_max_discount_error"],
            engine.stability.drift_curve_max_discount_deviation + 1e-12,
        )
        self.assertLess(
            diagnostics["max_abs_theta_jump_smoothed"],
            diagnostics["max_abs_theta_jump_raw"],
        )
        self.assertTrue(np.isfinite(engine.fitting_phi(engine.build_curve_grid(1500))).all())

    def test_negative_zero_yields_are_supported(self) -> None:
        maturities = np.array([0.25, 0.5, 1, 2, 3, 5, 7, 10.0])
        curve = hw.CurveData(
            maturities,
            np.full_like(maturities, -0.0025),
            curve_kind="zero",
            quote_convention="continuous",
        )
        engine = hw.HullWhiteCurveEngine(curve, 0.15, 0.01)
        self.assertTrue(np.all(engine.discount_factors > 0))
        self.assertGreater(engine.discount_factors[-1], 1.0)

    def test_scenario_preserves_metadata(self) -> None:
        shocked = hw.apply_scenario_shock(self.zero_curve, "Parallel +50bp")
        self.assertEqual(shocked.curve_kind, self.zero_curve.curve_kind)
        self.assertEqual(shocked.quote_convention, self.zero_curve.quote_convention)
        np.testing.assert_allclose(shocked.yields, self.zero_curve.yields + 0.005)

    def test_inverted_par_curve_is_valid(self) -> None:
        maturities = np.array([
            1 / 12, 1.5 / 12, 2 / 12, 3 / 12, 4 / 12,
            0.5, 1, 2, 3, 5, 7, 10, 20, 30.0,
        ])
        yields = np.array([
            0.0440, 0.0438, 0.0435, 0.0430, 0.0425,
            0.0415, 0.0395, 0.0375, 0.0365, 0.0370, 0.0380,
            0.0400, 0.0440, 0.0460,
        ])
        curve = hw.CurveData(
            maturities, yields, curve_kind="par",
            quote_convention="bond_equivalent_semiannual",
        )
        engine = hw.HullWhiteCurveEngine(curve, 0.15, 0.01)
        self.assertTrue(np.all(engine.discount_factors > 0))
        self.assertTrue(np.all(np.diff(engine.discount_factors) <= 1e-10))


if __name__ == "__main__":
    unittest.main(verbosity=2)

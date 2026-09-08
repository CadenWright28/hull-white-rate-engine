from __future__ import annotations

from typing import Mapping

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import PchipInterpolator, UnivariateSpline

from core import (
    CurveData,
    HullWhiteParams,
    SimulationConfig,
    StabilityConfig,
    _discount_factors_from_zero_quotes,
    _validate_raw_curve_arrays,
    bootstrap_zero_curve_from_par,
    fmt_pct,
    show_plot_nonblocking,
)


class HullWhiteCurveEngine:
    """One-factor Hull-White model under dr=(theta(t)-a*r)dt+sigma*dW."""

    def __init__(
        self,
        curve_data: CurveData,
        mean_reversion: float = 0.15,
        sigma: float = 0.01,
        stability_config: StabilityConfig | None = None,
    ) -> None:
        self.curve_data = curve_data
        self.input_maturities = np.asarray(curve_data.maturities, dtype=float)
        self.input_yields = np.asarray(curve_data.yields, dtype=float)
        self.curve_date = curve_data.curve_date
        self.curve_source = curve_data.source
        self.curve_kind = str(curve_data.curve_kind).lower()
        self.quote_convention = str(curve_data.quote_convention).lower()
        self.params = HullWhiteParams(float(mean_reversion), float(sigma))
        self.stability = stability_config or StabilityConfig()
        self._validate()
        self._build_discount_curve()
        self._build_model_curve()

    def _validate(self) -> None:
        _validate_raw_curve_arrays(self.input_maturities, self.input_yields)
        if len(self.input_maturities) < self.stability.min_curve_points:
            raise ValueError(f"Provide at least {self.stability.min_curve_points} market points.")
        if np.any(np.abs(self.input_yields) > self.stability.yield_warning_abs_level):
            raise ValueError("Input yield magnitude is outside the configured range.")
        if self.curve_kind not in {"par", "zero"}:
            raise ValueError("curve_kind must be 'par' or 'zero'.")
        if self.params.mean_reversion <= 0 or not np.isfinite(self.params.mean_reversion):
            raise ValueError("Mean reversion must be positive and finite.")
        if self.params.sigma < 0 or not np.isfinite(self.params.sigma):
            raise ValueError("Sigma must be nonnegative and finite.")

    def _build_discount_curve(self) -> None:
        if self.curve_kind == "par":
            self.maturities, self.zero_curve_points, self.discount_factors = (
                bootstrap_zero_curve_from_par(self.input_maturities, self.input_yields)
            )
            self.curve_conversion = "approximate semiannual par bootstrap"
        else:
            self.maturities = self.input_maturities.copy()
            self.zero_curve_points = self.input_yields.copy()
            self.discount_factors = _discount_factors_from_zero_quotes(
                self.maturities, self.zero_curve_points, self.quote_convention
            )
            self.curve_conversion = f"zero quotes ({self.quote_convention})"
        if np.any(self.discount_factors <= 0) or not np.isfinite(self.discount_factors).all():
            raise ValueError("Discount factors must be positive and finite.")
        self.max_curve_maturity = float(self.maturities[-1])
        times = np.r_[0.0, self.maturities]
        log_df = np.r_[0.0, np.log(self.discount_factors)]
        self.market_log_discount_interp = PchipInterpolator(times, log_df, extrapolate=False)
        self.log_discount_interp = self.market_log_discount_interp

    def _build_model_curve(self) -> None:
        times = np.r_[0.0, self.maturities]
        log_df = np.r_[0.0, np.log(self.discount_factors)]
        if self.stability.use_drift_curve_regularization:
            candidates = [0.0, *np.geomspace(
                max(self.stability.drift_smoothing_min_factor, 1e-18),
                max(self.stability.drift_smoothing_max_factor, self.stability.drift_smoothing_min_factor),
                max(2, self.stability.drift_smoothing_steps),
            )]
        else:
            candidates = [0.0]
        chosen = None
        for factor in candidates:
            spline = UnivariateSpline(times, log_df, k=3, s=float(factor), ext=2)
            offset = float(spline(0.0))
            fitted = np.exp(spline(self.maturities) - offset)
            error = float(np.max(np.abs(fitted - self.discount_factors)))
            if error <= self.stability.drift_curve_max_discount_deviation + 1e-15:
                chosen = (spline, float(factor), offset, error)
            elif chosen is not None:
                break
        if chosen is None:
            spline = UnivariateSpline(times, log_df, k=3, s=0.0, ext=2)
            offset = float(spline(0.0))
            error = float(np.max(np.abs(np.exp(spline(self.maturities) - offset) - self.discount_factors)))
            chosen = (spline, 0.0, offset, error)
        self.model_log_discount_spline, self.drift_smoothing_factor, self.model_log_discount_offset, self.drift_curve_max_discount_error = chosen
        self.model_discount_factors = np.exp(
            self.model_log_discount_spline(self.maturities) - self.model_log_discount_offset
        )

        grid = np.linspace(0.0, self.max_curve_maturity, self.stability.theta_grid_size)
        raw_forward = self.market_instantaneous_forward_rate(grid)
        model_forward = self.instantaneous_forward_rate(grid)
        a, sigma = self.params.mean_reversion, self.params.sigma
        convexity = sigma**2 / (2.0 * a) * (1.0 - np.exp(-2.0 * a * grid))
        raw_theta = self.market_forward_rate_derivative(grid) + a * raw_forward + convexity
        model_theta = self.forward_rate_derivative(grid) + a * model_forward + convexity
        self.theta_grid = grid
        self.theta_forward_raw = np.asarray(raw_forward)
        self.theta_forward_smoothed = np.asarray(model_forward)
        self.theta_raw = np.asarray(raw_theta)
        self.theta_smoothed = np.asarray(model_theta)
        self.theta_raw_interp = PchipInterpolator(grid, self.theta_raw, extrapolate=False)

    def _checked_times(self, times: float | np.ndarray) -> np.ndarray:
        arr = np.asarray(times, dtype=float)
        if not np.isfinite(arr).all():
            raise ValueError("Evaluation times must be finite.")
        if np.any(arr < -1e-12) or np.any(arr > self.max_curve_maturity + 1e-12):
            raise ValueError(f"Evaluation times must be in [0, {self.max_curve_maturity:.3f}].")
        return np.clip(arr, 0.0, self.max_curve_maturity)

    def market_discount_factor(self, times: float | np.ndarray) -> np.ndarray:
        t = self._checked_times(times)
        return np.exp(self.market_log_discount_interp(t))

    def discount_factor(self, times: float | np.ndarray) -> np.ndarray:
        t = self._checked_times(times)
        return np.exp(self.model_log_discount_spline(t) - self.model_log_discount_offset)

    def zero_yield(self, times: float | np.ndarray) -> np.ndarray:
        t = self._checked_times(times)
        df = self.discount_factor(t)
        return np.where(t > 1e-12, -np.log(df) / np.maximum(t, 1e-12), self.initial_short_rate())

    def market_instantaneous_forward_rate(self, times: float | np.ndarray) -> np.ndarray:
        t = self._checked_times(times)
        return -self.market_log_discount_interp.derivative(1)(t)

    def market_forward_rate_derivative(self, times: float | np.ndarray) -> np.ndarray:
        t = self._checked_times(times)
        return -self.market_log_discount_interp.derivative(2)(t)

    def instantaneous_forward_rate(self, times: float | np.ndarray) -> np.ndarray:
        t = self._checked_times(times)
        return -self.model_log_discount_spline.derivative(1)(t)

    def forward_rate_derivative(self, times: float | np.ndarray) -> np.ndarray:
        t = self._checked_times(times)
        return -self.model_log_discount_spline.derivative(2)(t)

    def hull_white_theta(self, times: float | np.ndarray, smoothed: bool = True) -> np.ndarray:
        t = self._checked_times(times)
        a, sigma = self.params.mean_reversion, self.params.sigma
        if smoothed:
            f = self.instantaneous_forward_rate(t)
            dfdt = self.forward_rate_derivative(t)
        else:
            f = self.market_instantaneous_forward_rate(t)
            dfdt = self.market_forward_rate_derivative(t)
        return dfdt + a * f + sigma**2 / (2.0 * a) * (1.0 - np.exp(-2.0 * a * t))

    def fitting_phi(self, times: float | np.ndarray) -> np.ndarray:
        t = self._checked_times(times)
        a, sigma = self.params.mean_reversion, self.params.sigma
        return self.instantaneous_forward_rate(t) + 0.5 * (
            sigma * (1.0 - np.exp(-a * t)) / a
        ) ** 2

    def initial_short_rate(self) -> float:
        return float(self.instantaneous_forward_rate(0.0))

    def build_curve_grid(self, num: int | None = None) -> np.ndarray:
        return np.linspace(0.0, self.max_curve_maturity, num or self.stability.diagnostics_grid_size)

    def B(self, t: float | np.ndarray, T: float | np.ndarray) -> np.ndarray:
        a = self.params.mean_reversion
        tau = np.maximum(np.asarray(T) - np.asarray(t), 0.0)
        return (1.0 - np.exp(-a * tau)) / a

    def A(self, t: float | np.ndarray, T: float | np.ndarray) -> np.ndarray:
        t_arr = np.asarray(t, dtype=float)
        T_arr = np.asarray(T, dtype=float)
        P0T = self.discount_factor(T_arr)
        P0t = self.discount_factor(t_arr)
        B = self.B(t_arr, T_arr)
        a, sigma = self.params.mean_reversion, self.params.sigma
        variance = sigma**2 / (4.0 * a) * (1.0 - np.exp(-2.0 * a * t_arr)) * B**2
        return P0T / P0t * np.exp(B * self.instantaneous_forward_rate(t_arr) - variance)

    def zero_coupon_bond_price(self, t: float, T: float, r_t: np.ndarray | float) -> np.ndarray:
        return self.A(t, T) * np.exp(-self.B(t, T) * np.asarray(r_t))

    def zero_coupon_yield(self, t: float, T: float, r_t: np.ndarray | float) -> np.ndarray:
        tau = max(float(T - t), 1e-12)
        return -np.log(np.maximum(self.zero_coupon_bond_price(t, T, r_t), 1e-300)) / tau

    def initial_curve_repricing_error(self) -> float:
        model = np.asarray(self.zero_coupon_bond_price(0.0, self.maturities, self.initial_short_rate()))
        return float(np.max(np.abs(model - self.model_discount_factors)))

    def curve_diagnostics(self) -> dict[str, float | bool]:
        grid = self.build_curve_grid()
        zero = self.zero_yield(grid)
        raw_f = self.market_instantaneous_forward_rate(grid)
        model_f = self.instantaneous_forward_rate(grid)
        raw_t = self.hull_white_theta(grid, False)
        model_t = self.hull_white_theta(grid, True)
        return {
            "min_zero": float(np.min(zero)), "max_zero": float(np.max(zero)),
            "min_forward_raw": float(np.min(raw_f)), "max_forward_raw": float(np.max(raw_f)),
            "min_forward_smoothed": float(np.min(model_f)), "max_forward_smoothed": float(np.max(model_f)),
            "min_theta_raw": float(np.min(raw_t)), "max_theta_raw": float(np.max(raw_t)),
            "min_theta_smoothed": float(np.min(model_t)), "max_theta_smoothed": float(np.max(model_t)),
            "max_abs_theta_jump_raw": float(np.max(np.abs(np.diff(raw_t)))),
            "max_abs_theta_jump_smoothed": float(np.max(np.abs(np.diff(model_t)))),
            "drift_smoothing_factor": self.drift_smoothing_factor,
            "drift_curve_max_discount_error": self.drift_curve_max_discount_error,
            "market_curve_fit_error": self.drift_curve_max_discount_error,
            "discounts_nonincreasing": bool(np.all(np.diff(self.model_discount_factors) <= 1e-10)),
            "max_repricing_error": self.initial_curve_repricing_error(),
        }

    def stability_warnings(self) -> list[str]:
        d = self.curve_diagnostics()
        warnings: list[str] = []
        if d["max_abs_theta_jump_smoothed"] > self.stability.theta_jump_warning_level:
            warnings.append("Regularized theta still contains a large adjacent jump.")
        if d["market_curve_fit_error"] > self.stability.drift_curve_max_discount_deviation + 1e-12:
            warnings.append("Regularized curve exceeds the market discount-factor tolerance.")
        if self.curve_kind == "par":
            warnings.append("Treasury CMT par yields use an approximate semiannual bootstrap.")
        return warnings

    def simulate_short_rate_paths(self, config: SimulationConfig) -> tuple[np.ndarray, np.ndarray]:
        times = np.linspace(0.0, config.horizon_years, config.n_steps + 1)
        dt = config.actual_dt
        a, sigma = self.params.mean_reversion, self.params.sigma
        decay = np.exp(-a * dt)
        state_std = sigma * np.sqrt((1.0 - np.exp(-2.0 * a * dt)) / (2.0 * a))
        rng = np.random.default_rng(config.seed)
        x = np.zeros((config.n_steps + 1, config.n_paths))
        for step in range(config.n_steps):
            x[step + 1] = decay * x[step] + state_std * rng.standard_normal(config.n_paths)
        paths = x + self.fitting_phi(times)[:, None]
        return times, paths

    def analytic_short_rate_moments(self, times: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        a, sigma = self.params.mean_reversion, self.params.sigma
        mean = self.fitting_phi(times)
        variance = sigma**2 * (1.0 - np.exp(-2.0 * a * times)) / (2.0 * a)
        return mean, np.sqrt(np.maximum(variance, 0.0))

    def simulation_summary(self, times: np.ndarray, paths: np.ndarray) -> dict[str, object]:
        theoretical_mean, theoretical_std = self.analytic_short_rate_moments(times)
        terminal = paths[-1]
        return {
            "mean_path": np.mean(paths, axis=1),
            "median_path": np.median(paths, axis=1),
            "p05_path": np.percentile(paths, 5, axis=1),
            "p95_path": np.percentile(paths, 95, axis=1),
            "theoretical_mean_path": theoretical_mean,
            "theoretical_std_path": theoretical_std,
            "terminal_mean": float(np.mean(terminal)),
            "terminal_std": float(np.std(terminal)),
            "terminal_p05": float(np.percentile(terminal, 5)),
            "terminal_p95": float(np.percentile(terminal, 95)),
            "terminal_theoretical_mean": float(theoretical_mean[-1]),
            "terminal_theoretical_std": float(theoretical_std[-1]),
            "terminal_mean_error": float(np.mean(terminal) - theoretical_mean[-1]),
            "terminal_std_error": float(np.std(terminal) - theoretical_std[-1]),
            "negative_terminal_share": float(np.mean(terminal < 0.0)),
        }

    def simulate_constant_maturity_zero_yield_paths(
        self, times: np.ndarray, short_rate_paths: np.ndarray, maturities: list[float]
    ) -> dict[float, np.ndarray]:
        output: dict[float, np.ndarray] = {}
        for maturity in maturities:
            if times[-1] + maturity > self.max_curve_maturity + 1e-12:
                raise ValueError("Tracked maturity plus horizon exceeds the curve.")
            paths = np.empty_like(short_rate_paths)
            for index, time in enumerate(times):
                paths[index] = self.zero_coupon_yield(time, time + maturity, short_rate_paths[index])
            output[maturity] = paths
        return output

    @staticmethod
    def yield_path_summary(yield_paths: Mapping[float, np.ndarray]) -> dict[float, dict[str, object]]:
        return {
            maturity: {
                "mean_path": np.mean(paths, axis=1),
                "median_path": np.median(paths, axis=1),
                "p05_path": np.percentile(paths, 5, axis=1),
                "p95_path": np.percentile(paths, 95, axis=1),
                "terminal_mean": float(np.mean(paths[-1])),
                "terminal_median": float(np.median(paths[-1])),
                "terminal_p05": float(np.percentile(paths[-1], 5)),
                "terminal_p95": float(np.percentile(paths[-1], 95)),
            }
            for maturity, paths in yield_paths.items()
        }

    def print_model_summary(self) -> None:
        print("=" * 78)
        print("HULL-WHITE CURVE ENGINE v4.1")
        print("=" * 78)
        print(f"Curve source          : {self.curve_source}")
        print(f"Curve date            : {self.curve_date}")
        print(f"Input curve type      : {self.curve_kind}")
        print(f"Curve conversion      : {self.curve_conversion}")
        print(f"Mean reversion a      : {self.params.mean_reversion:.6f} (user specified)")
        print(f"Sigma                 : {fmt_pct(self.params.sigma)} (user specified)")
        print("Parameter calibration : Not performed")
        print(f"Initial short rate    : {fmt_pct(self.initial_short_rate())}")
        print(f"Max curve maturity    : {self.max_curve_maturity:.2f} years")
        print(f"Initial repricing err : {self.initial_curve_repricing_error():.3e}")
        print("=" * 78)

    def print_curve_diagnostics(self) -> None:
        d = self.curve_diagnostics()
        print("=" * 78)
        print("CURVE / FITTING DIAGNOSTICS")
        print("=" * 78)
        print(f"Zero-yield range      : {fmt_pct(d['min_zero'])} to {fmt_pct(d['max_zero'])}")
        print(f"Forward range (raw)   : {fmt_pct(d['min_forward_raw'])} to {fmt_pct(d['max_forward_raw'])}")
        print(f"Forward range (model) : {fmt_pct(d['min_forward_smoothed'])} to {fmt_pct(d['max_forward_smoothed'])}")
        print(f"Theta range (raw)     : {fmt_pct(d['min_theta_raw'])} to {fmt_pct(d['max_theta_raw'])}")
        print(f"Theta range (model)   : {fmt_pct(d['min_theta_smoothed'])} to {fmt_pct(d['max_theta_smoothed'])}")
        print(f"Max theta jump (raw)  : {fmt_pct(d['max_abs_theta_jump_raw'])}")
        print(f"Max theta jump (model): {fmt_pct(d['max_abs_theta_jump_smoothed'])}")
        print(f"Drift smoothing factor: {d['drift_smoothing_factor']:.3e}")
        print(f"Max market DF deviation: {d['market_curve_fit_error']:.3e}")
        print(f"Discounts nonincreasing: {d['discounts_nonincreasing']}")
        print(f"Max initial repricing error: {d['max_repricing_error']:.3e}")
        for warning in self.stability_warnings():
            print(f"  - {warning}")
        print("=" * 78)

    def print_simulation_summary(self, config: SimulationConfig, summary: Mapping[str, object]) -> None:
        print("=" * 78)
        print("EXACT-TRANSITION SHORT-RATE MONTE CARLO SUMMARY")
        print("=" * 78)
        print(f"Paths                 : {config.n_paths}")
        print(f"Horizon               : {config.horizon_years:.2f} years")
        print(f"Actual dt             : {config.actual_dt:.8f}")
        print(f"Terminal mean         : {fmt_pct(summary['terminal_mean'])}")
        print(f"Theoretical mean      : {fmt_pct(summary['terminal_theoretical_mean'])}")
        print(f"Terminal std          : {fmt_pct(summary['terminal_std'])}")
        print(f"Theoretical std       : {fmt_pct(summary['terminal_theoretical_std'])}")
        print(f"Terminal 5th pct      : {fmt_pct(summary['terminal_p05'])}")
        print(f"Terminal 95th pct     : {fmt_pct(summary['terminal_p95'])}")
        print("=" * 78)

    @staticmethod
    def print_yield_path_summary(yield_summaries: Mapping[float, Mapping[str, object]]) -> None:
        print("=" * 78)
        print("CONSTANT-MATURITY ZERO-COUPON YIELD PATH SUMMARY")
        print("=" * 78)
        for maturity, summary in yield_summaries.items():
            print(f"{maturity:.1f}Y zero yield: mean={fmt_pct(summary['terminal_mean'])}, "
                  f"5th={fmt_pct(summary['terminal_p05'])}, 95th={fmt_pct(summary['terminal_p95'])}")

    def plot_curve_dashboard(self) -> None:
        grid = self.build_curve_grid()
        fig, axes = plt.subplots(2, 2, figsize=(14, 9))
        axes[0, 0].plot(self.input_maturities, self.input_yields * 100, "o", label="Market inputs")
        axes[0, 0].plot(grid, self.zero_yield(grid) * 100, label="Model zero curve")
        axes[0, 1].plot(grid, self.discount_factor(grid), label="Model discount curve")
        axes[1, 0].plot(grid, self.market_instantaneous_forward_rate(grid) * 100, label="Raw")
        axes[1, 0].plot(grid, self.instantaneous_forward_rate(grid) * 100, label="Regularized")
        axes[1, 1].plot(grid, self.hull_white_theta(grid, False) * 100, label="Raw theta")
        axes[1, 1].plot(grid, self.hull_white_theta(grid, True) * 100, label="Regularized theta")
        titles = ["Market Inputs and Zero Curve", "Discount Curve", "Instantaneous Forward Curve", "Hull-White Drift Fitting Function"]
        for axis, title in zip(axes.flat, titles):
            axis.set_title(title); axis.grid(True); axis.legend()
        plt.tight_layout(); show_plot_nonblocking()

    def plot_simulation_results(self, times: np.ndarray, paths: np.ndarray, summary: Mapping[str, object]) -> None:
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        axes[0].plot(times, paths[:, :50] * 100, alpha=0.7)
        axes[1].plot(times, np.asarray(summary["mean_path"]) * 100, label="Monte Carlo mean")
        axes[1].plot(times, np.asarray(summary["theoretical_mean_path"]) * 100, "--", label="Theoretical mean")
        axes[1].fill_between(times, np.asarray(summary["p05_path"]) * 100, np.asarray(summary["p95_path"]) * 100, alpha=0.3)
        axes[2].hist(paths[-1] * 100, bins=50)
        for axis in axes: axis.grid(True)
        axes[1].legend(); plt.tight_layout(); show_plot_nonblocking()

    @staticmethod
    def plot_yield_paths(times: np.ndarray, summaries: Mapping[float, Mapping[str, object]]) -> None:
        fig, axes = plt.subplots(len(summaries), 1, figsize=(12, 4 * len(summaries)), squeeze=False)
        for axis, (maturity, summary) in zip(axes.flat, summaries.items()):
            axis.plot(times, np.asarray(summary["mean_path"]) * 100, label="Mean")
            axis.fill_between(times, np.asarray(summary["p05_path"]) * 100, np.asarray(summary["p95_path"]) * 100, alpha=0.3, label="5%-95%")
            axis.set_title(f"{maturity:.1f}Y Constant-Maturity Zero Yield"); axis.grid(True); axis.legend()
        plt.tight_layout(); show_plot_nonblocking()

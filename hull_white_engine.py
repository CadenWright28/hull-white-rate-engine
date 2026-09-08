"""Small Hull-White learning project.

The point of this file is to keep the whole model visible in one place:
Treasury curve -> zero curve -> forwards -> Hull-White -> simulation.
"""

from __future__ import annotations

import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator, UnivariateSpline


TREASURY_URL = (
    "https://home.treasury.gov/resource-center/data-chart-center/interest-rates/"
    "pages/xml?data=daily_treasury_yield_curve"
)

TREASURY_FIELDS = {
    "BC1MONTH": 1 / 12,
    "BC15MONTH": 1.5 / 12,
    "BC2MONTH": 2 / 12,
    "BC3MONTH": 3 / 12,
    "BC4MONTH": 4 / 12,
    "BC6MONTH": 0.5,
    "BC1YEAR": 1.0,
    "BC2YEAR": 2.0,
    "BC3YEAR": 3.0,
    "BC5YEAR": 5.0,
    "BC7YEAR": 7.0,
    "BC10YEAR": 10.0,
    "BC20YEAR": 20.0,
    "BC30YEAR": 30.0,
}


@dataclass
class CurveData:
    maturities: np.ndarray
    yields: np.ndarray
    curve_date: str | None = None
    source: str | None = None
    curve_kind: str = "zero"
    quote_convention: str = "continuous"


@dataclass
class SimulationConfig:
    n_paths: int = 5000
    horizon_years: float = 1.0
    dt: float = 1 / 12
    seed: int = 42


# ----------------------------- curve setup -----------------------------


def _tag_name(tag: str) -> str:
    if "}" in tag:
        tag = tag.split("}", 1)[1]
    return "".join(ch for ch in tag.upper() if ch.isalnum())


def load_latest_treasury_curve(timeout: int = 15) -> CurveData:
    """Load the most recent Treasury CMT par curve."""
    years = [pd.Timestamp.today().year, pd.Timestamp.today().year - 1]
    best_date = None
    best_points = None

    for year in years:
        url = f"{TREASURY_URL}&field_tdr_date_value={year}"
        request = urllib.request.Request(url, headers={"User-Agent": "HullWhiteLearning/1.0"})
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                root = ET.fromstring(response.read())
        except Exception:
            continue

        for entry in root.iter():
            if _tag_name(entry.tag) != "ENTRY":
                continue

            row = {}
            for item in entry.iter():
                if item.text and item.text.strip():
                    row[_tag_name(item.tag)] = item.text.strip()

            date = pd.to_datetime(row.get("NEWDATE") or row.get("TIMEPERIOD"), errors="coerce")
            if pd.isna(date):
                continue

            points = {}
            for field, maturity in TREASURY_FIELDS.items():
                try:
                    points[maturity] = float(row[field]) / 100.0
                except (KeyError, ValueError):
                    pass

            if len(points) >= 5 and (best_date is None or date > best_date):
                best_date = date
                best_points = points

    if best_points is None:
        raise RuntimeError("Could not load a usable Treasury curve.")

    maturities = np.array(sorted(best_points), dtype=float)
    yields = np.array([best_points[t] for t in maturities], dtype=float)
    return CurveData(
        maturities,
        yields,
        curve_date=str(best_date.date()),
        source="U.S. Treasury CMT",
        curve_kind="par",
        quote_convention="semiannual",
    )


def zero_quote_discount_factors(
    maturities: np.ndarray,
    zero_yields: np.ndarray,
    convention: str = "continuous",
) -> np.ndarray:
    if convention == "continuous":
        return np.exp(-zero_yields * maturities)
    if convention in {"semiannual", "bond_equivalent_semiannual"}:
        return (1 + zero_yields / 2) ** (-2 * maturities)
    if convention == "simple":
        return 1 / (1 + zero_yields * maturities)
    raise ValueError("Unknown zero-yield convention.")


def bootstrap_zero_curve_from_par(
    maturities: np.ndarray,
    par_yields: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Approximate a zero curve from par yields using semiannual coupons."""
    maturities = np.asarray(maturities, dtype=float)
    par_yields = np.asarray(par_yields, dtype=float)
    order = np.argsort(maturities)
    maturities, par_yields = maturities[order], par_yields[order]

    par = PchipInterpolator(maturities, par_yields, extrapolate=False)

    short_nodes = maturities[maturities < 0.5 - 1e-12]
    max_half_year = np.floor(maturities[-1] * 2) / 2
    coupon_nodes = np.arange(0.5, max_half_year + 0.25, 0.5)
    grid = np.unique(np.r_[short_nodes, coupon_nodes])

    discounts: dict[float, float] = {}

    # Very short CMT points have no earlier coupon cash flows to strip out.
    for t in short_nodes:
        y = float(par(t))
        discounts[float(t)] = (1 + y / 2) ** (-2 * t)

    previous_coupon_dfs: list[float] = []
    for t in coupon_nodes:
        y = float(par(t))
        coupon = y / 2
        pv_earlier_coupons = coupon * sum(previous_coupon_dfs)
        df = (1 - pv_earlier_coupons) / (1 + coupon)
        if df <= 0:
            raise ValueError("Par curve produced a non-positive discount factor.")
        discounts[float(t)] = df
        previous_coupon_dfs.append(df)

    dfs = np.array([discounts[float(t)] for t in grid])
    zeros = -np.log(dfs) / grid
    return grid, zeros, dfs


# ----------------------------- Hull-White -----------------------------


class HullWhiteCurveEngine:
    def __init__(
        self,
        curve_data: CurveData,
        mean_reversion: float = 0.15,
        sigma: float = 0.01,
        smooth_curve: bool = True,
    ) -> None:
        self.curve_data = curve_data
        self.a = float(mean_reversion)
        self.sigma = float(sigma)

        if self.a <= 0 or self.sigma < 0:
            raise ValueError("mean_reversion must be positive and sigma cannot be negative.")

        if curve_data.curve_kind.lower() == "par":
            self.maturities, self.zero_curve_points, self.discount_factors = (
                bootstrap_zero_curve_from_par(curve_data.maturities, curve_data.yields)
            )
        else:
            self.maturities = np.asarray(curve_data.maturities, dtype=float)
            self.zero_curve_points = np.asarray(curve_data.yields, dtype=float)
            self.discount_factors = zero_quote_discount_factors(
                self.maturities,
                self.zero_curve_points,
                curve_data.quote_convention.lower(),
            )

        times = np.r_[0.0, self.maturities]
        log_df = np.r_[0.0, np.log(self.discount_factors)]
        self.market_curve = PchipInterpolator(times, log_df, extrapolate=False)

        # Differentiating a bumpy curve makes theta noisy. A light spline smooth
        # keeps the model usable without moving far from the bootstrapped points.
        if smooth_curve and len(times) >= 4:
            self.model_curve = UnivariateSpline(times, log_df, k=3, s=1e-7, ext=2)
            self._curve_offset = float(self.model_curve(0.0))
        else:
            self.model_curve = self.market_curve
            self._curve_offset = 0.0

        self.max_curve_maturity = float(self.maturities[-1])

    def _check_time(self, t: float | np.ndarray) -> np.ndarray:
        arr = np.asarray(t, dtype=float)
        if np.any(arr < 0) or np.any(arr > self.max_curve_maturity):
            raise ValueError("Time is outside the curve range.")
        return arr

    def market_discount_factor(self, t: float | np.ndarray) -> np.ndarray:
        t = self._check_time(t)
        return np.exp(self.market_curve(t))

    def discount_factor(self, t: float | np.ndarray) -> np.ndarray:
        t = self._check_time(t)
        return np.exp(self.model_curve(t) - self._curve_offset)

    def zero_yield(self, t: float | np.ndarray) -> np.ndarray:
        t = self._check_time(t)
        df = self.discount_factor(t)
        return np.where(t > 1e-12, -np.log(df) / np.maximum(t, 1e-12), self.initial_short_rate())

    def instantaneous_forward_rate(self, t: float | np.ndarray) -> np.ndarray:
        t = self._check_time(t)
        return -self.model_curve.derivative(1)(t)

    def forward_rate_derivative(self, t: float | np.ndarray) -> np.ndarray:
        t = self._check_time(t)
        return -self.model_curve.derivative(2)(t)

    def market_instantaneous_forward_rate(self, t: float | np.ndarray) -> np.ndarray:
        t = self._check_time(t)
        return -self.market_curve.derivative(1)(t)

    def market_forward_rate_derivative(self, t: float | np.ndarray) -> np.ndarray:
        t = self._check_time(t)
        return -self.market_curve.derivative(2)(t)

    def hull_white_theta(self, t: float | np.ndarray, smoothed: bool = True) -> np.ndarray:
        t = self._check_time(t)
        if smoothed:
            f = self.instantaneous_forward_rate(t)
            dfdt = self.forward_rate_derivative(t)
        else:
            f = self.market_instantaneous_forward_rate(t)
            dfdt = self.market_forward_rate_derivative(t)
        convexity = self.sigma**2 / (2 * self.a) * (1 - np.exp(-2 * self.a * t))
        return dfdt + self.a * f + convexity

    def initial_short_rate(self) -> float:
        return float(self.instantaneous_forward_rate(0.0))

    def initial_curve_repricing_error(self) -> float:
        model_df = self.discount_factor(self.maturities)
        return float(np.max(np.abs(model_df - self.discount_factors)))

    def B(self, t: float | np.ndarray, T: float | np.ndarray) -> np.ndarray:
        return (1 - np.exp(-self.a * (np.asarray(T) - np.asarray(t)))) / self.a

    def zero_coupon_bond_price(
        self,
        t: float | np.ndarray,
        T: float | np.ndarray,
        short_rate: float | np.ndarray,
    ) -> np.ndarray:
        t = np.asarray(t, dtype=float)
        T = np.asarray(T, dtype=float)
        if np.any(T < t):
            raise ValueError("Bond maturity T must be after t.")
        if np.any(T > self.max_curve_maturity):
            raise ValueError("Bond maturity is outside the curve range.")

        B = self.B(t, T)
        p0T = self.discount_factor(T)
        p0t = self.discount_factor(t)
        f0t = self.instantaneous_forward_rate(t)
        variance_term = (
            self.sigma**2
            / (4 * self.a)
            * (1 - np.exp(-2 * self.a * t))
            * B**2
        )
        A = p0T / p0t * np.exp(B * f0t - variance_term)
        return A * np.exp(-B * np.asarray(short_rate))

    def simulate_short_rate_paths(
        self,
        config: SimulationConfig,
    ) -> tuple[np.ndarray, np.ndarray]:
        steps = max(1, int(round(config.horizon_years / config.dt)))
        dt = config.horizon_years / steps
        times = np.linspace(0, config.horizon_years, steps + 1)
        paths = np.empty((config.n_paths, steps + 1))
        paths[:, 0] = self.initial_short_rate()

        rng = np.random.default_rng(config.seed)
        decay = np.exp(-self.a * dt)
        shock_std = self.sigma * np.sqrt((1 - np.exp(-2 * self.a * dt)) / (2 * self.a))

        for i in range(steps):
            theta = float(self.hull_white_theta(times[i]))
            mean = paths[:, i] * decay + theta / self.a * (1 - decay)
            paths[:, i + 1] = mean + shock_std * rng.standard_normal(config.n_paths)

        return times, paths

    def analytic_short_rate_moments(
        self,
        config: SimulationConfig,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        steps = max(1, int(round(config.horizon_years / config.dt)))
        dt = config.horizon_years / steps
        times = np.linspace(0, config.horizon_years, steps + 1)
        mean = np.empty(steps + 1)
        variance = np.empty(steps + 1)
        mean[0] = self.initial_short_rate()
        variance[0] = 0.0

        decay = np.exp(-self.a * dt)
        step_var = self.sigma**2 * (1 - np.exp(-2 * self.a * dt)) / (2 * self.a)
        for i in range(steps):
            theta = float(self.hull_white_theta(times[i]))
            mean[i + 1] = mean[i] * decay + theta / self.a * (1 - decay)
            variance[i + 1] = variance[i] * decay**2 + step_var
        return times, mean, np.sqrt(variance)

    def constant_maturity_yields(
        self,
        times: np.ndarray,
        short_rate_paths: np.ndarray,
        maturity_years: float,
    ) -> np.ndarray:
        output = np.full_like(short_rate_paths, np.nan, dtype=float)
        for i, t in enumerate(times):
            T = t + maturity_years
            if T > self.max_curve_maturity:
                break
            prices = self.zero_coupon_bond_price(t, T, short_rate_paths[:, i])
            output[:, i] = -np.log(prices) / maturity_years
        return output

    def plot_curve(self) -> None:
        grid = np.linspace(0.01, self.max_curve_maturity, 500)
        plt.figure(figsize=(9, 5))
        plt.plot(grid, 100 * self.zero_yield(grid), label="zero yield")
        plt.plot(grid, 100 * self.instantaneous_forward_rate(grid), label="forward rate")
        plt.xlabel("Years")
        plt.ylabel("Rate (%)")
        plt.title("Bootstrapped curve")
        plt.legend()
        plt.tight_layout()

    def plot_simulation(self, times: np.ndarray, paths: np.ndarray) -> None:
        plt.figure(figsize=(9, 5))
        for row in paths[:40]:
            plt.plot(times, 100 * row, alpha=0.25)
        plt.plot(times, 100 * paths.mean(axis=0), linewidth=2, label="mean")
        plt.xlabel("Years")
        plt.ylabel("Short rate (%)")
        plt.title("Hull-White short-rate paths")
        plt.legend()
        plt.tight_layout()


def apply_scenario_shock(curve: CurveData, shift_bps: float) -> CurveData:
    return CurveData(
        np.array(curve.maturities, copy=True),
        np.array(curve.yields, copy=True) + shift_bps / 10_000,
        curve.curve_date,
        curve.source,
        curve.curve_kind,
        curve.quote_convention,
    )


def run_demo() -> None:
    curve = load_latest_treasury_curve()
    config = SimulationConfig()

    for name, shift in [("Base", 0), ("+50 bp", 50), ("-50 bp", -50)]:
        scenario_curve = apply_scenario_shock(curve, shift)
        engine = HullWhiteCurveEngine(scenario_curve)
        times, paths = engine.simulate_short_rate_paths(config)
        print(
            f"{name:7s} | short rate now {engine.initial_short_rate():.3%} "
            f"| 1Y mean {paths[:, -1].mean():.3%} "
            f"| 1Y std {paths[:, -1].std(ddof=1):.3%}"
        )

        if name == "Base":
            engine.plot_curve()
            engine.plot_simulation(times, paths)

    plt.show()


if __name__ == "__main__":
    run_demo()

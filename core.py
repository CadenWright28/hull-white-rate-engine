"""Curve data and shared helpers for the Hull-White model."""

from __future__ import annotations

import json
import os
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator

ENGINE_VERSION = "4.1"
TREASURY_XML_BASE_URL = (
    "https://home.treasury.gov/resource-center/data-chart-center/interest-rates/"
    "pages/xml?data=daily_treasury_yield_curve"
)
TREASURY_FIELD_MAP: Mapping[str, float] = {
    "BC1MONTH": 1.0 / 12.0,
    "BC15MONTH": 1.5 / 12.0,
    "BC2MONTH": 2.0 / 12.0,
    "BC3MONTH": 3.0 / 12.0,
    "BC4MONTH": 4.0 / 12.0,
    "BC6MONTH": 6.0 / 12.0,
    "BC1YEAR": 1.0,
    "BC2YEAR": 2.0,
    "BC3YEAR": 3.0,
    "BC5YEAR": 5.0,
    "BC7YEAR": 7.0,
    "BC10YEAR": 10.0,
    "BC20YEAR": 20.0,
    "BC30YEAR": 30.0,
}
MIN_CURVE_MATURITY = 1.0 / 365.0

SCENARIO_DEFINITIONS = {
    "Base": {"type": "base"},
    "Parallel +50bp": {"type": "parallel", "shift_bps": 50},
    "Parallel -50bp": {"type": "parallel", "shift_bps": -50},
}
DISPLAY_SCENARIO = "Parallel +50bp"
RUN_ALL_SCENARIOS = True
SHOW_BASE_ENGINE_PLOTS = True
SHOW_SCENARIO_COMPARISON_DASHBOARD = True
SHOW_ALL_SCENARIO_SUMMARY_CHART = False


@dataclass(frozen=True)
class CurveData:
    maturities: np.ndarray
    yields: np.ndarray
    curve_date: str | None = None
    source: str | None = None
    curve_kind: str = "zero"
    quote_convention: str = "continuous"


@dataclass(frozen=True)
class HullWhiteParams:
    mean_reversion: float
    sigma: float


@dataclass(frozen=True)
class SimulationConfig:
    n_paths: int
    horizon_years: float
    dt: float
    seed: int = 42

    def __post_init__(self) -> None:
        if int(self.n_paths) < 1:
            raise ValueError("n_paths must be positive.")
        if not np.isfinite(self.horizon_years) or self.horizon_years <= 0:
            raise ValueError("horizon_years must be positive and finite.")
        if not np.isfinite(self.dt) or self.dt <= 0:
            raise ValueError("dt must be positive and finite.")
        if self.dt > self.horizon_years:
            raise ValueError("dt cannot exceed horizon_years.")

    @property
    def n_steps(self) -> int:
        return max(1, int(round(self.horizon_years / self.dt)))

    @property
    def actual_dt(self) -> float:
        return self.horizon_years / self.n_steps


@dataclass(frozen=True)
class StabilityConfig:
    theta_warning_abs_level: float = 0.10
    theta_jump_warning_level: float = 0.03
    yield_warning_abs_level: float = 0.15
    forward_warning_abs_level: float = 0.25
    min_curve_points: int = 5
    diagnostics_grid_size: int = 750
    negative_terminal_rate_warning_share: float = 0.05
    theta_grid_size: int = 3000
    use_drift_curve_regularization: bool = True
    drift_curve_max_discount_deviation: float = 1.5e-4
    drift_smoothing_min_factor: float = 1e-12
    drift_smoothing_max_factor: float = 1e-3
    drift_smoothing_steps: int = 24
    repricing_tolerance: float = 1e-10


def fmt_pct(x: float, decimals: int = 3) -> str:
    return f"{100.0 * float(x):.{decimals}f}%"


def show_plot_nonblocking() -> None:
    plt.show(block=False)
    plt.pause(0.001)


def keep_plots_open_until_enter() -> None:
    try:
        input("\nPlots are open. Press Enter to close all plot windows...")
    finally:
        plt.close("all")


def _normalize_xml_field(tag: str) -> str:
    if "}" in tag:
        tag = tag.split("}", 1)[1]
    return "".join(character for character in tag.upper() if character.isalnum())


def _validate_raw_curve_arrays(maturities: np.ndarray, yields: np.ndarray) -> None:
    if maturities.ndim != 1 or yields.ndim != 1:
        raise ValueError("Maturities and yields must be one-dimensional arrays.")
    if len(maturities) != len(yields):
        raise ValueError("Maturities and yields must have equal lengths.")
    if len(maturities) < 2:
        raise ValueError("At least two curve observations are required.")
    if not np.isfinite(maturities).all() or not np.isfinite(yields).all():
        raise ValueError("Curve inputs contain NaN or infinite values.")
    if np.any(maturities <= 0):
        raise ValueError("All market-curve maturities must be positive.")
    if np.any(np.diff(maturities) <= 0):
        raise ValueError("Market-curve maturities must be strictly increasing.")


def load_curve_data_from_csv(
    csv_path: str | os.PathLike[str],
    curve_kind: str = "zero",
    quote_convention: str = "continuous",
) -> CurveData:
    path = Path(csv_path)
    if not path.is_absolute():
        candidate = Path(__file__).resolve().parent / path
        if candidate.exists():
            path = candidate
    frame = pd.read_csv(path)
    required = {"maturity_years", "yield"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Curve CSV is missing columns: {sorted(missing)}")
    frame = frame.dropna(subset=["maturity_years", "yield"]).copy()
    frame = frame.sort_values("maturity_years")
    if frame["maturity_years"].duplicated().any():
        raise ValueError("Curve CSV contains duplicate maturities.")
    maturities = frame["maturity_years"].to_numpy(dtype=float)
    yields = frame["yield"].to_numpy(dtype=float)
    mask = maturities >= MIN_CURVE_MATURITY
    maturities = maturities[mask]
    yields = yields[mask]
    _validate_raw_curve_arrays(maturities, yields)
    curve_date = None
    if "curve_date" in frame.columns:
        dates = frame["curve_date"].dropna().astype(str)
        if not dates.empty:
            curve_date = dates.iloc[0]
    if "curve_kind" in frame.columns:
        kinds = frame["curve_kind"].dropna().astype(str)
        if not kinds.empty:
            curve_kind = kinds.iloc[0]
    if "quote_convention" in frame.columns:
        conventions = frame["quote_convention"].dropna().astype(str)
        if not conventions.empty:
            quote_convention = conventions.iloc[0]
    return CurveData(
        maturities=maturities,
        yields=yields,
        curve_date=curve_date,
        source=f"csv:{path.name}",
        curve_kind=str(curve_kind).lower(),
        quote_convention=str(quote_convention).lower(),
    )


def _fetch_treasury_year(year: int, timeout: int) -> CurveData | None:
    url = f"{TREASURY_XML_BASE_URL}&field_tdr_date_value={year}"
    request = urllib.request.Request(
        url,
        headers={"User-Agent": f"HullWhiteCurveEngine/{ENGINE_VERSION}"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        xml_bytes = response.read()
    root = ET.fromstring(xml_bytes)
    best_timestamp: pd.Timestamp | None = None
    best_points: dict[float, float] | None = None
    for entry in root.iter():
        if _normalize_xml_field(entry.tag) != "ENTRY":
            continue
        record: dict[str, str] = {}
        for element in entry.iter():
            text = element.text.strip() if element.text else ""
            if text:
                record[_normalize_xml_field(element.tag)] = text
        raw_date = record.get("NEWDATE") or record.get("TIMEPERIOD")
        parsed_date = pd.to_datetime(raw_date, errors="coerce")
        if pd.isna(parsed_date):
            continue
        points: dict[float, float] = {}
        for field_name, maturity in TREASURY_FIELD_MAP.items():
            raw_value = record.get(field_name)
            if raw_value is None:
                continue
            try:
                value = float(raw_value)
            except ValueError:
                continue
            if np.isfinite(value):
                points[maturity] = value / 100.0
        if len(points) < 5:
            continue
        if best_timestamp is None or parsed_date > best_timestamp:
            best_timestamp = parsed_date
            best_points = points
    if best_timestamp is None or best_points is None:
        return None
    maturities = np.array(sorted(best_points), dtype=float)
    yields = np.array([best_points[maturity] for maturity in maturities], dtype=float)
    _validate_raw_curve_arrays(maturities, yields)
    return CurveData(
        maturities=maturities,
        yields=yields,
        curve_date=str(best_timestamp.date()),
        source="U.S. Treasury CMT par curve",
        curve_kind="par",
        quote_convention="bond_equivalent_semiannual",
    )


def load_latest_treasury_curve(timeout: int = 20) -> CurveData:
    current_year = pd.Timestamp.today().year
    errors: list[str] = []
    candidates: list[CurveData] = []
    for year in (current_year, current_year - 1):
        try:
            candidate = _fetch_treasury_year(year, timeout)
            if candidate is not None:
                candidates.append(candidate)
        except Exception as exc:
            errors.append(f"{year}: {type(exc).__name__}: {exc}")
    if not candidates:
        details = "; ".join(errors) if errors else "No usable Treasury entries were returned."
        raise RuntimeError(f"Unable to load a usable Treasury curve. {details}")
    return max(candidates, key=lambda item: pd.Timestamp(item.curve_date))


def load_settings_from_json(json_path: str | None = None) -> dict[str, object]:
    defaults: dict[str, object] = {
        "source": "treasury",
        "csv_path": None,
        "csv_curve_kind": "zero",
        "csv_quote_convention": "continuous",
        "mean_reversion": 0.15,
        "sigma": 0.01,
        "n_paths": 5000,
        "horizon_years": 1.0,
        "dt": 1.0 / 12.0,
        "seed": 42,
        "yield_maturities_to_track": [2.0, 5.0, 10.0],
        "theta_warning_abs_level": 0.10,
        "theta_jump_warning_level": 0.03,
        "yield_warning_abs_level": 0.15,
        "forward_warning_abs_level": 0.25,
        "negative_terminal_rate_warning_share": 0.05,
        "theta_grid_size": 3000,
        "use_drift_curve_regularization": True,
        "drift_curve_max_discount_deviation": 1.5e-4,
        "drift_smoothing_min_factor": 1e-12,
        "drift_smoothing_max_factor": 1e-3,
        "drift_smoothing_steps": 24,
        "repricing_tolerance": 1e-10,
    }
    if json_path is None:
        return defaults
    with open(json_path, "r", encoding="utf-8") as file:
        user_settings = json.load(file)
    defaults.update(user_settings)
    return defaults


def get_curve_data(settings: Mapping[str, object]) -> CurveData:
    source = str(settings.get("source", "treasury")).lower()
    if source == "treasury":
        return load_latest_treasury_curve()
    if source == "csv":
        csv_path = settings.get("csv_path")
        if not csv_path:
            raise ValueError("settings['csv_path'] is required when source='csv'.")
        return load_curve_data_from_csv(
            str(csv_path),
            curve_kind=str(settings.get("csv_curve_kind", "zero")),
            quote_convention=str(settings.get("csv_quote_convention", "continuous")),
        )
    raise ValueError("settings['source'] must be 'treasury' or 'csv'.")


def _discount_factors_from_zero_quotes(
    maturities: np.ndarray,
    zero_yields: np.ndarray,
    quote_convention: str,
) -> np.ndarray:
    convention = quote_convention.lower()
    if convention == "continuous":
        discount_factors = np.exp(-zero_yields * maturities)
    elif convention in {"semiannual", "bond_equivalent_semiannual"}:
        base = 1.0 + zero_yields / 2.0
        if np.any(base <= 0):
            raise ValueError("Semiannual zero-yield quotes imply a nonpositive compounding base.")
        discount_factors = base ** (-2.0 * maturities)
    elif convention == "simple":
        denominator = 1.0 + zero_yields * maturities
        if np.any(denominator <= 0):
            raise ValueError("Simple zero-yield quotes imply nonpositive discount factors.")
        discount_factors = 1.0 / denominator
    else:
        raise ValueError(
            "Unsupported zero-yield quote convention. Use continuous, semiannual, or simple."
        )
    return discount_factors


def bootstrap_zero_curve_from_par(
    maturities: np.ndarray,
    par_yields: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    _validate_raw_curve_arrays(maturities, par_yields)
    if maturities[-1] < 1.0:
        raise ValueError("Par bootstrapping requires a curve extending to at least one year.")
    par_interp = PchipInterpolator(maturities, par_yields, extrapolate=False)
    max_half_year = np.floor(maturities[-1] * 2.0 + 1e-12) / 2.0
    coupon_grid = np.arange(0.5, max_half_year + 0.25, 0.5)
    discount_by_time: dict[float, float] = {}
    for maturity in maturities[maturities < 0.5 - 1e-12]:
        par_rate = float(par_interp(maturity))
        base = 1.0 + par_rate / 2.0
        if base <= 0:
            raise ValueError("A short par yield implies a nonpositive compounding base.")
        discount_by_time[float(maturity)] = base ** (-2.0 * float(maturity))
    coupon_discounts: list[float] = []
    for maturity in coupon_grid:
        par_rate = float(par_interp(maturity))
        semiannual_coupon = par_rate / 2.0
        numerator = 1.0 - semiannual_coupon * float(np.sum(coupon_discounts))
        denominator = 1.0 + semiannual_coupon
        discount_factor = numerator / denominator
        if not np.isfinite(discount_factor) or discount_factor <= 0:
            raise ValueError(
                f"Par bootstrap produced an invalid discount factor at {maturity:.2f} years."
            )
        coupon_discounts.append(discount_factor)
        discount_by_time[float(maturity)] = discount_factor
    curve_maturities = np.array(sorted(discount_by_time), dtype=float)
    discount_factors = np.array(
        [discount_by_time[float(maturity)] for maturity in curve_maturities],
        dtype=float,
    )
    zero_yields = -np.log(discount_factors) / curve_maturities
    return curve_maturities, zero_yields, discount_factors


def _bps_to_decimal(bps: float) -> float:
    return float(bps) / 10000.0


def apply_scenario_shock(
    base_curve_data: CurveData,
    scenario_name: str,
    scenario_definitions: Mapping[str, Mapping[str, object]] | None = None,
) -> CurveData:
    definitions = scenario_definitions or SCENARIO_DEFINITIONS
    if scenario_name not in definitions:
        raise ValueError(f"Unknown scenario {scenario_name!r}. Available: {list(definitions)}")
    scenario = definitions[scenario_name]
    shock_type = str(scenario.get("type", "base")).lower()
    maturities = np.array(base_curve_data.maturities, dtype=float)
    base_yields = np.array(base_curve_data.yields, dtype=float)
    shocked_yields = base_yields.copy()
    if shock_type == "base":
        pass
    elif shock_type == "parallel":
        shocked_yields += _bps_to_decimal(float(scenario.get("shift_bps", 0.0)))
    elif shock_type == "tilt":
        short_shift = _bps_to_decimal(float(scenario.get("short_end_bps", 0.0)))
        long_shift = _bps_to_decimal(float(scenario.get("long_end_bps", 0.0)))
        decay_years = float(scenario.get("decay_years", max(maturities[-1], 1.0)))
        weight = np.clip(maturities / decay_years, 0.0, 1.0)
        shocked_yields += short_shift + (long_shift - short_shift) * weight
    else:
        raise ValueError(f"Unsupported scenario type: {shock_type}")
    floor = scenario.get("yield_floor")
    cap = scenario.get("yield_cap")
    if floor is not None:
        shocked_yields = np.maximum(shocked_yields, float(floor))
    if cap is not None:
        shocked_yields = np.minimum(shocked_yields, float(cap))
    return CurveData(
        maturities=maturities,
        yields=shocked_yields,
        curve_date=base_curve_data.curve_date,
        source=f"{base_curve_data.source}:{scenario_name}",
        curve_kind=base_curve_data.curve_kind,
        quote_convention=base_curve_data.quote_convention,
    )

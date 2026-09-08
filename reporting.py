from __future__ import annotations

from typing import Mapping

import matplotlib.pyplot as plt
import numpy as np

from core import (
    CurveData,
    DISPLAY_SCENARIO,
    SCENARIO_DEFINITIONS,
    SimulationConfig,
    StabilityConfig,
    apply_scenario_shock,
    fmt_pct,
    show_plot_nonblocking,
)
from engine import HullWhiteCurveEngine


def run_single_engine(
    curve_data: CurveData,
    settings: Mapping[str, object],
    stability_config: StabilityConfig,
) -> dict[str, object]:
    engine = HullWhiteCurveEngine(
        curve_data,
        mean_reversion=float(settings["mean_reversion"]),
        sigma=float(settings["sigma"]),
        stability_config=stability_config,
    )
    config = SimulationConfig(
        n_paths=int(settings["n_paths"]),
        horizon_years=float(settings["horizon_years"]),
        dt=float(settings["dt"]),
        seed=int(settings["seed"]),
    )
    times, short_rate_paths = engine.simulate_short_rate_paths(config)
    sim_summary = engine.simulation_summary(times, short_rate_paths)
    maturities = [float(value) for value in settings["yield_maturities_to_track"]]
    yield_paths = engine.simulate_constant_maturity_zero_yield_paths(
        times, short_rate_paths, maturities
    )
    return {
        "engine": engine,
        "sim_config": config,
        "times": times,
        "short_rate_paths": short_rate_paths,
        "sim_summary": sim_summary,
        "yield_paths": yield_paths,
        "yield_summaries": engine.yield_path_summary(yield_paths),
    }


def run_all_scenarios(
    base_curve_data: CurveData,
    settings: Mapping[str, object],
    stability_config: StabilityConfig,
    scenario_definitions: Mapping[str, Mapping[str, object]] | None = None,
) -> dict[str, dict[str, object]]:
    definitions = scenario_definitions or SCENARIO_DEFINITIONS
    return {
        name: run_single_engine(
            apply_scenario_shock(base_curve_data, name, definitions),
            settings,
            stability_config,
        )
        for name in definitions
    }


def print_scenario_summary_table(
    results: Mapping[str, Mapping[str, object]], tracked_maturity: float = 10.0
) -> None:
    print("=" * 110)
    print(f"SCENARIO COMPARISON ({tracked_maturity:.1f}Y terminal zero yield and short rate)")
    print("=" * 110)
    header = (
        f"{'Scenario':<22}{'r(T) Mean':>13}{'r(T) 5%':>12}{'r(T) 95%':>12}"
        f"{f'{tracked_maturity:.1f}Y Mean':>15}{f'{tracked_maturity:.1f}Y 5%':>13}"
        f"{f'{tracked_maturity:.1f}Y 95%':>13}"
    )
    print(header)
    print("-" * len(header))
    for name, result in results.items():
        sim = result["sim_summary"]
        yields = result["yield_summaries"][tracked_maturity]
        print(
            f"{name:<22}{fmt_pct(sim['terminal_mean']):>13}"
            f"{fmt_pct(sim['terminal_p05']):>12}{fmt_pct(sim['terminal_p95']):>12}"
            f"{fmt_pct(yields['terminal_mean']):>15}{fmt_pct(yields['terminal_p05']):>13}"
            f"{fmt_pct(yields['terminal_p95']):>13}"
        )
    print("=" * 110)


def plot_scenario_comparison_dashboard(
    results: Mapping[str, Mapping[str, object]],
    display_scenario: str = DISPLAY_SCENARIO,
    tracked_maturity: float = 10.0,
) -> None:
    base, active = results["Base"], results[display_scenario]
    base_engine, active_engine = base["engine"], active["engine"]
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    grid = base_engine.build_curve_grid()
    axes[0, 0].plot(grid, base_engine.zero_yield(grid) * 100, label="Base")
    axes[0, 0].plot(grid, active_engine.zero_yield(grid) * 100, label=display_scenario)
    axes[0, 0].set_title("Initial Model Zero Curve")
    axes[0, 1].plot(base["times"], np.asarray(base["sim_summary"]["mean_path"]) * 100, label="Base")
    axes[0, 1].plot(active["times"], np.asarray(active["sim_summary"]["mean_path"]) * 100, label=display_scenario)
    axes[0, 1].set_title("Mean Short-Rate Path")
    base_y = base["yield_summaries"][tracked_maturity]
    active_y = active["yield_summaries"][tracked_maturity]
    axes[1, 0].plot(base["times"], np.asarray(base_y["mean_path"]) * 100, label="Base")
    axes[1, 0].plot(active["times"], np.asarray(active_y["mean_path"]) * 100, label=display_scenario)
    axes[1, 0].set_title(f"Mean {tracked_maturity:.1f}Y Zero Yield")
    axes[1, 1].hist(base["yield_paths"][tracked_maturity][-1] * 100, bins=40, alpha=0.55, label="Base")
    axes[1, 1].hist(active["yield_paths"][tracked_maturity][-1] * 100, bins=40, alpha=0.55, label=display_scenario)
    axes[1, 1].set_title(f"Terminal {tracked_maturity:.1f}Y Zero-Yield Distribution")
    for axis in axes.flat:
        axis.grid(True)
        axis.legend()
    plt.tight_layout()
    show_plot_nonblocking()


def _build_stability_config(settings: Mapping[str, object]) -> StabilityConfig:
    return StabilityConfig(
        theta_warning_abs_level=float(settings["theta_warning_abs_level"]),
        theta_jump_warning_level=float(settings["theta_jump_warning_level"]),
        yield_warning_abs_level=float(settings["yield_warning_abs_level"]),
        forward_warning_abs_level=float(settings["forward_warning_abs_level"]),
        negative_terminal_rate_warning_share=float(settings["negative_terminal_rate_warning_share"]),
        theta_grid_size=int(settings["theta_grid_size"]),
        use_drift_curve_regularization=bool(settings["use_drift_curve_regularization"]),
        drift_curve_max_discount_deviation=float(settings["drift_curve_max_discount_deviation"]),
        drift_smoothing_min_factor=float(settings["drift_smoothing_min_factor"]),
        drift_smoothing_max_factor=float(settings["drift_smoothing_max_factor"]),
        drift_smoothing_steps=int(settings["drift_smoothing_steps"]),
        repricing_tolerance=float(settings["repricing_tolerance"]),
    )

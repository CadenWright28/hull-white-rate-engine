"""Public entry point for the Hull-White yield-curve and rate engine."""
from __future__ import annotations

import traceback

from core import *
from engine import HullWhiteCurveEngine
from reporting import *


if __name__ == "__main__":
    try:
        SETTINGS_JSON_PATH = None
        settings = load_settings_from_json(SETTINGS_JSON_PATH)
        curve_data = get_curve_data(settings)
        stability = _build_stability_config(settings)

        base_result = run_single_engine(curve_data, settings, stability)
        engine = base_result["engine"]
        config = base_result["sim_config"]

        engine.print_model_summary()
        engine.print_curve_diagnostics()
        engine.print_simulation_summary(config, base_result["sim_summary"])
        engine.print_yield_path_summary(base_result["yield_summaries"])

        plots_shown = False
        if RUN_ALL_SCENARIOS:
            results = run_all_scenarios(curve_data, settings, stability)
            print_scenario_summary_table(results, tracked_maturity=10.0)
            if SHOW_SCENARIO_COMPARISON_DASHBOARD and DISPLAY_SCENARIO != "Base":
                plot_scenario_comparison_dashboard(
                    results, DISPLAY_SCENARIO, tracked_maturity=10.0
                )
                plots_shown = True

        if SHOW_BASE_ENGINE_PLOTS:
            engine.plot_curve_dashboard()
            engine.plot_simulation_results(
                base_result["times"],
                base_result["short_rate_paths"],
                base_result["sim_summary"],
            )
            engine.plot_yield_paths(
                base_result["times"], base_result["yield_summaries"]
            )
            plots_shown = True

        if plots_shown:
            keep_plots_open_until_enter()

    except Exception as exc:
        print("\n" + "=" * 78)
        print("SCRIPT FAILED TO RUN")
        print("=" * 78)
        print(f"Error type: {type(exc).__name__}")
        print(f"Error message: {exc}")
        traceback.print_exc()
        try:
            input("\nPress Enter to close...")
        except EOFError:
            pass

"""
Sensitivity Analysis Module.

Tornado diagrams (Spearman rank correlation), criticality index,
and scenario analysis (what-if) for the Monte Carlo simulation.
"""

from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

from .monte_carlo import SimulationResults, SimulationConfig, run_simulation
from .risks import RiskRegistry


def tornado_analysis(results: SimulationResults, activities_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute Spearman rank correlation between each activity's sampled
    duration and total project duration.

    This identifies which activities have the strongest influence on
    overall project duration.

    Parameters
    ----------
    results : SimulationResults
        Results from Monte Carlo simulation.
    activities_df : pd.DataFrame
        Activity data.

    Returns
    -------
    pd.DataFrame
        Sorted by absolute correlation, columns:
        activity_id, name, correlation, abs_correlation, rank
    """
    total = results.total_durations
    rows = []

    for _, row in activities_df.iterrows():
        aid = row["id"]
        if aid in results.activity_durations:
            durations = results.activity_durations[aid]
            corr, p_value = sp_stats.spearmanr(durations, total)
            rows.append({
                "activity_id": aid,
                "name": row.get("name", f"Activity {aid}"),
                "category": row.get("category", ""),
                "correlation": corr,
                "abs_correlation": abs(corr),
                "p_value": p_value,
            })

    df = pd.DataFrame(rows)
    df = df.sort_values("abs_correlation", ascending=False).reset_index(drop=True)
    df["rank"] = range(1, len(df) + 1)
    return df


def criticality_analysis(
    results: SimulationResults, activities_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Compute criticality index: percentage of iterations where each
    activity was on the critical path.

    Parameters
    ----------
    results : SimulationResults
    activities_df : pd.DataFrame

    Returns
    -------
    pd.DataFrame
        Sorted by criticality_index descending.
    """
    ci = results.criticality_index
    rows = []
    for _, row in activities_df.iterrows():
        aid = row["id"]
        rows.append({
            "activity_id": aid,
            "name": row.get("name", f"Activity {aid}"),
            "category": row.get("category", ""),
            "criticality_index": ci.get(aid, 0.0),
            "mean_duration": float(np.mean(results.activity_durations.get(aid, [0]))),
            "std_duration": float(np.std(results.activity_durations.get(aid, [0]))),
        })

    df = pd.DataFrame(rows)
    df = df.sort_values("criticality_index", ascending=False).reset_index(drop=True)
    return df


def significance_index(results: SimulationResults, activities_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute significance index (SI) = criticality_index * correlation.

    SI combines both the frequency on the critical path and the
    strength of correlation with total duration.
    """
    tornado = tornado_analysis(results, activities_df)
    crit = criticality_analysis(results, activities_df)

    merged = tornado.merge(
        crit[["activity_id", "criticality_index"]],
        on="activity_id",
        how="left",
    )
    merged["significance_index"] = (
        merged["criticality_index"] * merged["abs_correlation"]
    )
    merged = merged.sort_values("significance_index", ascending=False).reset_index(drop=True)
    return merged


def scenario_analysis(
    activities_df: pd.DataFrame,
    scenarios: dict[str, dict],
    risk_registry: Optional[RiskRegistry] = None,
    config: Optional[SimulationConfig] = None,
) -> pd.DataFrame:
    """
    Run Monte Carlo simulation under different scenarios and compare.

    Parameters
    ----------
    activities_df : pd.DataFrame
        Base activity data.
    scenarios : dict
        Mapping of scenario_name -> modification dict.
        Modification dict can contain:
        - 'duration_factor': float, multiply all durations by this factor
        - 'risk_probability_factor': float, multiply all risk probabilities
        - 'affected_activities': dict of activity_id -> {'a': ..., 'm': ..., 'b': ...}
        - 'additional_risks': list of RiskEvent objects
    risk_registry : RiskRegistry, optional
    config : SimulationConfig, optional

    Returns
    -------
    pd.DataFrame
        Comparison of scenarios with P50, P80, P90, mean, std for each.

    Example
    -------
    >>> scenarios = {
    ...     "Base Case": {},
    ...     "Regulatory Change": {"duration_factor": 1.15},
    ...     "Extreme Weather": {"risk_probability_factor": 2.0},
    ...     "Combined Adverse": {"duration_factor": 1.15, "risk_probability_factor": 2.0},
    ... }
    """
    if config is None:
        config = SimulationConfig()

    results_list = []

    for scenario_name, modifications in scenarios.items():
        # Copy activities
        df_mod = activities_df.copy()

        # Apply duration factor
        factor = modifications.get("duration_factor", 1.0)
        if factor != 1.0:
            for col in ["a", "m", "b"]:
                if col in df_mod.columns:
                    df_mod[col] = df_mod[col] * factor

        # Apply specific activity modifications
        affected = modifications.get("affected_activities", {})
        for aid, params in affected.items():
            mask = df_mod["id"] == aid
            for key, val in params.items():
                if key in df_mod.columns:
                    df_mod.loc[mask, key] = val

        # Modify risk registry
        reg = RiskRegistry()
        if risk_registry:
            for risk in risk_registry.risks.values():
                from copy import deepcopy
                r = deepcopy(risk)
                prob_factor = modifications.get("risk_probability_factor", 1.0)
                r.probability = min(r.probability * prob_factor, 1.0)
                reg.add_risk(r)

        # Add additional risks
        for extra_risk in modifications.get("additional_risks", []):
            reg.add_risk(extra_risk)

        # Run simulation
        sim_results = run_simulation(df_mod, reg, config)

        results_list.append({
            "scenario": scenario_name,
            "mean": sim_results.statistics["mean"],
            "std": sim_results.statistics["std"],
            "P50": sim_results.statistics["P50"],
            "P80": sim_results.statistics["P80"],
            "P90": sim_results.statistics["P90"],
            "min": sim_results.statistics["min"],
            "max": sim_results.statistics["max"],
        })

    return pd.DataFrame(results_list)


def sensitivity_to_risks(
    activities_df: pd.DataFrame,
    risk_registry: RiskRegistry,
    config: Optional[SimulationConfig] = None,
) -> pd.DataFrame:
    """
    Measure sensitivity of project duration to each individual risk.

    Runs simulation with each risk individually and compares to baseline.

    Returns
    -------
    pd.DataFrame
        Risk sensitivity ranking.
    """
    if config is None:
        config = SimulationConfig(n_iterations=5000)

    # Baseline: no risks
    baseline = run_simulation(activities_df, None, config)
    baseline_p50 = baseline.statistics["P50"]

    rows = []
    for risk_id, risk in risk_registry.risks.items():
        if risk.status != "active":
            continue
        single_reg = RiskRegistry()
        single_reg.add_risk(risk)
        result = run_simulation(activities_df, single_reg, config)

        rows.append({
            "risk_id": risk_id,
            "risk_name": risk.name,
            "probability": risk.probability,
            "impact_mode": risk.impact_mode,
            "P50_with_risk": result.statistics["P50"],
            "P50_baseline": baseline_p50,
            "delta_P50": result.statistics["P50"] - baseline_p50,
            "delta_pct": (result.statistics["P50"] - baseline_p50) / baseline_p50 * 100,
        })

    df = pd.DataFrame(rows)
    df = df.sort_values("delta_P50", ascending=False).reset_index(drop=True)
    return df


def what_if_duration_change(
    activities_df: pd.DataFrame,
    activity_id: int,
    new_a: float,
    new_m: float,
    new_b: float,
    risk_registry: Optional[RiskRegistry] = None,
    config: Optional[SimulationConfig] = None,
) -> dict:
    """
    What-if analysis: change a single activity's estimates and compare.

    Returns
    -------
    dict with 'original' and 'modified' simulation statistics.
    """
    if config is None:
        config = SimulationConfig(n_iterations=5000)

    original = run_simulation(activities_df, risk_registry, config)

    df_mod = activities_df.copy()
    mask = df_mod["id"] == activity_id
    df_mod.loc[mask, "a"] = new_a
    df_mod.loc[mask, "m"] = new_m
    df_mod.loc[mask, "b"] = new_b

    modified = run_simulation(df_mod, risk_registry, config)

    return {
        "original": original.statistics,
        "modified": modified.statistics,
        "delta_P50": modified.statistics["P50"] - original.statistics["P50"],
        "delta_P90": modified.statistics["P90"] - original.statistics["P90"],
    }

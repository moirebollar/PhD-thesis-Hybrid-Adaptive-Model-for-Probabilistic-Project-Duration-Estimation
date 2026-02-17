"""
Earned Schedule Module.

Calculates Earned Schedule (ES), Schedule Variance SV(t),
Schedule Performance Index SPI(t), and Independent Estimate
at Completion IEAC(t) metrics. Compares ES predictions with
Monte Carlo predictions and feeds into the Bayesian updater.
"""

from typing import Optional

import numpy as np
import pandas as pd


class EarnedScheduleCalculator:
    """
    Earned Schedule calculator for project progress tracking.

    Earned Schedule extends Earned Value Management by measuring
    schedule performance in time units rather than cost units.
    """

    def __init__(
        self,
        planned_schedule: pd.DataFrame,
        planned_duration: float,
    ):
        """
        Parameters
        ----------
        planned_schedule : pd.DataFrame
            Planned activity schedule with columns:
            id, name, planned_start, planned_finish, planned_duration,
            weight (proportion of total work).
        planned_duration : float
            Total planned project duration (PD).
        """
        self.planned = planned_schedule.copy()
        self.pd_total = planned_duration

        # Ensure weight column exists (default: equal weight)
        if "weight" not in self.planned.columns:
            n = len(self.planned)
            self.planned["weight"] = 1.0 / n

        # Normalize weights to sum to 1
        total_weight = self.planned["weight"].sum()
        if total_weight > 0:
            self.planned["weight"] = self.planned["weight"] / total_weight

        # Build planned value curve (PV over time)
        self._build_pv_curve()

    def _build_pv_curve(self):
        """Build the cumulative planned value curve over time."""
        max_time = int(np.ceil(self.pd_total * 1.5))
        self.pv_curve = np.zeros(max_time + 1)

        for _, row in self.planned.iterrows():
            start = row.get("planned_start", 0)
            finish = row.get("planned_finish", start + row.get("planned_duration", 0))
            duration = finish - start
            weight = row["weight"]

            if duration > 0:
                daily_pv = weight / duration
                for t in range(int(np.floor(start)), min(int(np.ceil(finish)), max_time + 1)):
                    self.pv_curve[t] += daily_pv

        # Cumulative PV
        self.cpv_curve = np.cumsum(self.pv_curve)
        self.cpv_curve = np.clip(self.cpv_curve, 0, 1.0)

    def calculate_es(
        self,
        actual_time: float,
        completed_activities: dict[int, float],
        in_progress_activities: Optional[dict[int, float]] = None,
    ) -> dict:
        """
        Calculate Earned Schedule metrics at a given point in time.

        Parameters
        ----------
        actual_time : float
            Current actual time (AT) in days from project start.
        completed_activities : dict[int, float]
            {activity_id: actual_finish_time} for completed activities.
        in_progress_activities : dict[int, float], optional
            {activity_id: percent_complete (0-1)} for in-progress activities.

        Returns
        -------
        dict with ES, SV(t), SPI(t), IEAC(t), and other metrics.
        """
        if in_progress_activities is None:
            in_progress_activities = {}

        # Calculate Earned Value (EV) - cumulative work completed
        ev = 0.0
        for _, row in self.planned.iterrows():
            aid = row["id"]
            weight = row["weight"]

            if aid in completed_activities:
                ev += weight  # 100% complete
            elif aid in in_progress_activities:
                ev += weight * in_progress_activities[aid]

        # Calculate Earned Schedule (ES)
        # ES = time at which PV equals current EV
        es = self._find_es(ev)

        # Schedule metrics
        at = actual_time
        sv_t = es - at  # Schedule Variance (time)
        spi_t = es / at if at > 0 else 1.0  # Schedule Performance Index (time)

        # Independent Estimate at Completion
        # IEAC(t) = AT + (PD - ES) / SPI(t)
        if spi_t > 0:
            ieac_t = at + (self.pd_total - es) / spi_t
        else:
            ieac_t = float("inf")

        # To Complete Schedule Performance Index
        # TSPI = (PD - ES) / (PD - AT)
        remaining_planned = self.pd_total - at
        if remaining_planned > 0:
            tspi = (self.pd_total - es) / remaining_planned
        else:
            tspi = float("inf")

        # Percent schedule complete
        pct_complete = ev
        pct_time_elapsed = at / self.pd_total if self.pd_total > 0 else 0

        return {
            "actual_time": at,
            "earned_value": ev,
            "earned_schedule": es,
            "planned_duration": self.pd_total,
            "SV_t": sv_t,
            "SPI_t": spi_t,
            "IEAC_t": ieac_t,
            "TSPI": tspi,
            "percent_complete": pct_complete,
            "percent_time_elapsed": pct_time_elapsed,
            "status": self._interpret_status(spi_t),
        }

    def _find_es(self, ev: float) -> float:
        """
        Find Earned Schedule: the time at which planned value equals EV.
        Uses linear interpolation between PV curve points.
        """
        ev = min(ev, 1.0)

        # Find where cumulative PV curve crosses the EV level
        for t in range(len(self.cpv_curve) - 1):
            if self.cpv_curve[t + 1] >= ev:
                # Linear interpolation
                if self.cpv_curve[t + 1] == self.cpv_curve[t]:
                    return float(t)
                fraction = (ev - self.cpv_curve[t]) / (
                    self.cpv_curve[t + 1] - self.cpv_curve[t]
                )
                return t + fraction

        return float(len(self.cpv_curve) - 1)

    @staticmethod
    def _interpret_status(spi_t: float) -> str:
        """Interpret SPI(t) value."""
        if spi_t > 1.05:
            return "Ahead of schedule"
        elif spi_t > 0.95:
            return "On schedule"
        elif spi_t > 0.80:
            return "Slightly behind schedule"
        else:
            return "Significantly behind schedule"

    def tracking_table(
        self,
        tracking_data: list[dict],
    ) -> pd.DataFrame:
        """
        Generate a tracking table from a time series of progress reports.

        Parameters
        ----------
        tracking_data : list[dict]
            Each dict has:
            - 'time': actual time
            - 'completed': {activity_id: finish_time}
            - 'in_progress': {activity_id: percent_complete}

        Returns
        -------
        pd.DataFrame
            Time series of ES metrics.
        """
        rows = []
        for report in tracking_data:
            metrics = self.calculate_es(
                actual_time=report["time"],
                completed_activities=report.get("completed", {}),
                in_progress_activities=report.get("in_progress", {}),
            )
            rows.append(metrics)
        return pd.DataFrame(rows)


def compare_es_with_montecarlo(
    es_metrics: dict,
    mc_statistics: dict,
) -> dict:
    """
    Compare Earned Schedule predictions with Monte Carlo predictions.

    Parameters
    ----------
    es_metrics : dict
        From EarnedScheduleCalculator.calculate_es().
    mc_statistics : dict
        From SimulationResults.statistics.

    Returns
    -------
    dict with comparison metrics.
    """
    ieac = es_metrics["IEAC_t"]
    mc_p50 = mc_statistics["P50"]
    mc_p80 = mc_statistics["P80"]
    mc_p90 = mc_statistics["P90"]

    return {
        "ES_prediction_IEAC": ieac,
        "MC_P50": mc_p50,
        "MC_P80": mc_p80,
        "MC_P90": mc_p90,
        "ES_vs_MC_P50_delta": ieac - mc_p50,
        "ES_vs_MC_P80_delta": ieac - mc_p80,
        "ES_closest_percentile": _find_closest_percentile(
            ieac, mc_statistics
        ),
        "SPI_t": es_metrics["SPI_t"],
        "recommendation": _recommend_estimate(es_metrics, mc_statistics),
    }


def _find_closest_percentile(value: float, mc_stats: dict) -> str:
    """Find which MC percentile the ES prediction is closest to."""
    percentiles = {
        "P10": mc_stats.get("P10", 0),
        "P25": mc_stats.get("P25", 0),
        "P50": mc_stats.get("P50", 0),
        "P80": mc_stats.get("P80", 0),
        "P90": mc_stats.get("P90", 0),
        "P95": mc_stats.get("P95", 0),
    }
    closest = min(percentiles, key=lambda k: abs(percentiles[k] - value))
    return closest


def _recommend_estimate(es_metrics: dict, mc_stats: dict) -> str:
    """Generate a recommendation based on ES and MC comparison."""
    spi = es_metrics["SPI_t"]
    ieac = es_metrics["IEAC_t"]
    p80 = mc_stats["P80"]

    if spi >= 0.95 and abs(ieac - p80) / p80 < 0.05:
        return "ES and MC predictions are consistent. Use MC P80 for planning."
    elif spi < 0.90:
        return (
            "Project is significantly behind schedule. "
            "Consider re-estimating remaining activities and running Bayesian update."
        )
    elif ieac > p80:
        return (
            "ES predicts longer duration than MC P80. "
            "The project may have systematic delays not captured in the model. "
            "Feed ES data into Bayesian updater."
        )
    else:
        return "Use the higher of ES IEAC and MC P80 for conservative planning."


def create_planned_schedule_from_cpm(
    activities_df: pd.DataFrame,
    cpm_schedule: pd.DataFrame,
) -> pd.DataFrame:
    """
    Create a planned schedule DataFrame from CPM deterministic analysis.

    Parameters
    ----------
    activities_df : pd.DataFrame
        Activity definitions.
    cpm_schedule : pd.DataFrame
        From monte_carlo.deterministic_cpm() with ES, EF columns.

    Returns
    -------
    pd.DataFrame
        Formatted for EarnedScheduleCalculator.
    """
    merged = activities_df.merge(
        cpm_schedule[["id", "ES", "EF", "duration"]],
        on="id",
        how="left",
    )

    schedule = pd.DataFrame({
        "id": merged["id"],
        "name": merged.get("name", merged["id"].astype(str)),
        "planned_start": merged["ES"],
        "planned_finish": merged["EF"],
        "planned_duration": merged["duration"],
        "weight": merged["duration"] / merged["duration"].sum(),
    })

    return schedule

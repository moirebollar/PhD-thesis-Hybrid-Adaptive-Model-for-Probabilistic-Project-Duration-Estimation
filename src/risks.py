"""
Risk Registry & Discrete Events Module.

Models discrete risk events as Bernoulli(p) with impact distributions.
Supports conditional risks, pre-built risk types from the thesis,
and unidentified/residual risk reserves.
"""

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from .distributions import create_distribution


@dataclass
class RiskEvent:
    """
    A discrete risk event that may occur during a project.

    Attributes
    ----------
    risk_id : str
        Unique identifier.
    name : str
        Descriptive name.
    probability : float
        Probability of occurrence (0-1).
    impact_dist_type : str
        Distribution type for impact ('triangular', 'betapert', 'uniform').
    impact_min : float
        Minimum impact in days.
    impact_mode : float
        Most likely impact in days.
    impact_max : float
        Maximum impact in days.
    applies_to : list[int] or str
        Activity IDs affected, or 'all' for project-level risk.
    category : str
        Risk category (technical, environmental, regulatory, supply_chain, residual).
    is_conditional : bool
        Whether this risk depends on another risk occurring first.
    condition_risk_id : str or None
        ID of the risk that must trigger for this one to be evaluated.
    status : str
        'active', 'mitigated', or 'closed'.
    """
    risk_id: str
    name: str
    probability: float
    impact_dist_type: str = "triangular"
    impact_min: float = 0.0
    impact_mode: float = 0.0
    impact_max: float = 0.0
    applies_to: list = field(default_factory=lambda: ["all"])
    category: str = "technical"
    is_conditional: bool = False
    condition_risk_id: Optional[str] = None
    status: str = "active"

    def __post_init__(self):
        if not (0 <= self.probability <= 1):
            raise ValueError(f"Probability must be 0-1, got {self.probability}")
        if isinstance(self.applies_to, str):
            if self.applies_to.lower() == "all":
                self.applies_to = ["all"]
            else:
                self.applies_to = [
                    int(x.strip()) for x in self.applies_to.split(",") if x.strip()
                ]

    def sample_occurs(self, rng: np.random.Generator) -> bool:
        """Sample whether this risk event occurs."""
        if self.status != "active":
            return False
        return rng.random() < self.probability

    def sample_impact(self, rng: np.random.Generator) -> float:
        """Sample the impact duration if the risk occurs."""
        if self.impact_dist_type.lower() == "uniform":
            return rng.uniform(self.impact_min, self.impact_max)
        else:
            dist = create_distribution(
                self.impact_dist_type,
                self.impact_min,
                self.impact_mode,
                self.impact_max,
            )
            return float(dist.sample(1, rng=rng)[0])

    def affects_activity(self, activity_id: int) -> bool:
        """Check if this risk applies to a given activity."""
        if "all" in self.applies_to:
            return True
        return activity_id in self.applies_to


class RiskRegistry:
    """
    Registry of all project risks.

    Manages risk events, simulates their occurrence, and calculates
    aggregate impact on project activities.
    """

    def __init__(self):
        self.risks: dict[str, RiskEvent] = {}

    def add_risk(self, risk: RiskEvent) -> None:
        """Add a risk event to the registry."""
        self.risks[risk.risk_id] = risk

    def remove_risk(self, risk_id: str) -> None:
        """Remove a risk from the registry."""
        self.risks.pop(risk_id, None)

    def get_risk(self, risk_id: str) -> Optional[RiskEvent]:
        return self.risks.get(risk_id)

    def active_risks(self) -> list[RiskEvent]:
        """Return all active risks."""
        return [r for r in self.risks.values() if r.status == "active"]

    def simulate_risks(
        self, activity_ids: list[int], rng: np.random.Generator
    ) -> dict[int, float]:
        """
        Simulate all risk events for one Monte Carlo iteration.

        Returns a dictionary mapping activity_id -> total additional delay (days).

        Parameters
        ----------
        activity_ids : list[int]
            All activity IDs in the project.
        rng : np.random.Generator
            Random number generator for reproducibility.

        Returns
        -------
        dict[int, float]
            Mapping of activity_id to total risk-induced delay.
        """
        delays = {aid: 0.0 for aid in activity_ids}
        triggered = set()

        # Process non-conditional risks first
        for risk in self.active_risks():
            if risk.is_conditional:
                continue
            if risk.sample_occurs(rng):
                triggered.add(risk.risk_id)
                impact = risk.sample_impact(rng)
                self._apply_impact(risk, impact, activity_ids, delays)

        # Process conditional risks
        for risk in self.active_risks():
            if not risk.is_conditional:
                continue
            if risk.condition_risk_id in triggered:
                if risk.sample_occurs(rng):
                    triggered.add(risk.risk_id)
                    impact = risk.sample_impact(rng)
                    self._apply_impact(risk, impact, activity_ids, delays)

        return delays

    def _apply_impact(
        self,
        risk: RiskEvent,
        impact: float,
        activity_ids: list[int],
        delays: dict[int, float],
    ) -> None:
        """Apply risk impact to affected activities."""
        if "all" in risk.applies_to:
            # Distribute impact across all activities proportionally
            n = len(activity_ids)
            per_activity = impact / n if n > 0 else 0
            for aid in activity_ids:
                delays[aid] += per_activity
        else:
            # Apply impact equally among specified activities
            affected = [aid for aid in risk.applies_to if aid in activity_ids]
            n = len(affected)
            per_activity = impact / n if n > 0 else impact
            for aid in affected:
                delays[aid] += per_activity

    def load_from_dataframe(self, df: pd.DataFrame) -> None:
        """
        Load risks from a DataFrame (e.g., from Excel template).

        Expected columns: risk_id, name, probability, impact_dist_type,
        impact_min, impact_mode, impact_max, applies_to, category, status
        """
        for _, row in df.iterrows():
            applies_to = row.get("applies_to", "all")
            if isinstance(applies_to, str):
                if applies_to.lower().strip() == "all":
                    applies_to = ["all"]
                else:
                    applies_to = [
                        int(x.strip()) for x in applies_to.split(",") if x.strip()
                    ]
            elif pd.isna(applies_to):
                applies_to = ["all"]

            risk = RiskEvent(
                risk_id=str(row.get("risk_id", row.get("Risk ID", ""))),
                name=str(row.get("name", row.get("Risk Name", ""))),
                probability=float(row.get("probability", row.get("Probability (0-1)", 0))),
                impact_dist_type=str(row.get("impact_dist_type",
                                             row.get("Impact Distribution", "triangular"))),
                impact_min=float(row.get("impact_min",
                                        row.get("Impact Min (days)", 0))),
                impact_mode=float(row.get("impact_mode",
                                         row.get("Impact Most Likely (days)", 0))),
                impact_max=float(row.get("impact_max",
                                        row.get("Impact Max (days)", 0))),
                applies_to=applies_to,
                category=str(row.get("category", row.get("Risk Category", "technical"))),
                status=str(row.get("status", row.get("Status", "active"))).lower(),
            )
            self.add_risk(risk)

    def load_from_excel(self, filepath: str, sheet_name: str = "Risk Registry") -> None:
        """Load risks from an Excel risk registry file."""
        df = pd.read_excel(filepath, sheet_name=sheet_name, engine="openpyxl")
        self.load_from_dataframe(df)

    def to_dataframe(self) -> pd.DataFrame:
        """Export all risks to a DataFrame."""
        rows = []
        for risk in self.risks.values():
            rows.append({
                "risk_id": risk.risk_id,
                "name": risk.name,
                "probability": risk.probability,
                "impact_dist_type": risk.impact_dist_type,
                "impact_min": risk.impact_min,
                "impact_mode": risk.impact_mode,
                "impact_max": risk.impact_max,
                "applies_to": ",".join(str(x) for x in risk.applies_to),
                "category": risk.category,
                "status": risk.status,
            })
        return pd.DataFrame(rows)

    def summary(self) -> dict:
        """Return a summary of the risk registry."""
        active = self.active_risks()
        return {
            "total_risks": len(self.risks),
            "active_risks": len(active),
            "expected_total_impact": sum(
                r.probability * r.impact_mode for r in active
            ),
            "max_total_impact": sum(r.impact_max for r in active),
            "categories": list(set(r.category for r in active)),
        }


def create_default_risks() -> RiskRegistry:
    """
    Create a risk registry with default risks from the thesis example.

    Returns
    -------
    RiskRegistry
        Pre-populated risk registry.
    """
    registry = RiskRegistry()

    registry.add_risk(RiskEvent(
        risk_id="R1",
        name="Unexpected contaminant",
        probability=0.25,
        impact_dist_type="triangular",
        impact_min=8,
        impact_mode=15,
        impact_max=28,
        applies_to=[1, 2, 5],
        category="technical",
    ))

    registry.add_risk(RiskEvent(
        risk_id="R2",
        name="Extreme weather event",
        probability=0.20,
        impact_dist_type="triangular",
        impact_min=5,
        impact_mode=10,
        impact_max=20,
        applies_to=[3, 4, 5, 6],
        category="environmental",
    ))

    registry.add_risk(RiskEvent(
        risk_id="R3",
        name="Equipment delivery delay",
        probability=0.30,
        impact_dist_type="betapert",
        impact_min=10,
        impact_mode=20,
        impact_max=45,
        applies_to=[5, 6],
        category="supply_chain",
    ))

    registry.add_risk(RiskEvent(
        risk_id="R4",
        name="Regulatory change",
        probability=0.15,
        impact_dist_type="triangular",
        impact_min=15,
        impact_mode=30,
        impact_max=60,
        applies_to=[1, 2],
        category="regulatory",
    ))

    registry.add_risk(RiskEvent(
        risk_id="R5",
        name="Unidentified risk reserve",
        probability=0.07,
        impact_dist_type="uniform",
        impact_min=15,
        impact_mode=45,
        impact_max=90,
        applies_to=["all"],
        category="residual",
    ))

    return registry

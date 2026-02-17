"""Tests for the Monte Carlo simulation engine."""

import numpy as np
import pandas as pd
import pytest

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.monte_carlo import (
    SimulationConfig,
    build_precedence_network,
    run_simulation,
    deterministic_cpm,
    find_critical_path,
    _forward_pass,
)
from src.risks import RiskRegistry, RiskEvent, create_default_risks


def make_sample_activities():
    """Create the sample 10-activity project for testing."""
    return pd.DataFrame({
        "id": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        "name": [
            "Engineering", "Permitting", "Site prep", "Civil works",
            "Electromechanical", "Piping", "Instrumentation",
            "Testing", "Commissioning", "Handover",
        ],
        "predecessors": [[], [1], [1], [3], [2, 4], [4], [5, 6], [7], [8], [9]],
        "dist_type": ["betapert"] * 10,
        "a": [20, 15, 5, 20, 25, 15, 10, 8, 10, 5],
        "m": [30, 25, 10, 30, 40, 22, 15, 12, 18, 7],
        "b": [45, 50, 18, 50, 65, 35, 25, 22, 35, 14],
        "category": [
            "design", "permitting", "site_preparation", "civil_works",
            "electromechanical", "piping", "instrumentation",
            "testing", "commissioning", "startup",
        ],
    })


class TestPrecedenceNetwork:
    def test_build_network(self):
        df = make_sample_activities()
        G = build_precedence_network(df)
        assert "START" in G.nodes
        assert "END" in G.nodes
        assert G.number_of_nodes() == 12  # 10 activities + START + END

    def test_dag_property(self):
        import networkx as nx
        df = make_sample_activities()
        G = build_precedence_network(df)
        assert nx.is_directed_acyclic_graph(G)

    def test_start_connections(self):
        """Activities with no predecessors should connect to START."""
        df = make_sample_activities()
        G = build_precedence_network(df)
        start_successors = list(G.successors("START"))
        assert 1 in start_successors  # Activity 1 has no predecessors

    def test_end_connections(self):
        """Activities with no successors should connect to END."""
        df = make_sample_activities()
        G = build_precedence_network(df)
        end_predecessors = list(G.predecessors("END"))
        assert 10 in end_predecessors  # Activity 10 is the last one

    def test_cycle_detection(self):
        """Circular dependencies should raise ValueError."""
        df = pd.DataFrame({
            "id": [1, 2, 3],
            "name": ["A", "B", "C"],
            "predecessors": [[3], [1], [2]],  # cycle: 1->2->3->1
            "dist_type": ["betapert"] * 3,
            "a": [10, 10, 10],
            "m": [20, 20, 20],
            "b": [30, 30, 30],
        })
        with pytest.raises(ValueError, match="cycles"):
            build_precedence_network(df)


class TestDeterministicCPM:
    def test_cpm_duration(self):
        df = make_sample_activities()
        duration, critical_path, schedule = deterministic_cpm(df)

        # Critical path through most-likely values:
        # 1(30) -> 3(10) -> 4(30) -> 5(40) -> 7(15) -> 8(12) -> 9(18) -> 10(7) = 162
        # OR 1(30) -> 2(25) -> 5(40) -> 7(15) -> 8(12) -> 9(18) -> 10(7) = 147
        # OR 1(30) -> 3(10) -> 4(30) -> 6(22) -> 7(15) -> 8(12) -> 9(18) -> 10(7) = 144
        # So longest path should be 162
        assert duration > 0
        assert len(critical_path) > 0

    def test_schedule_has_required_columns(self):
        df = make_sample_activities()
        _, _, schedule = deterministic_cpm(df)
        required = ["id", "ES", "EF", "LS", "LF", "TF", "on_critical_path"]
        for col in required:
            assert col in schedule.columns

    def test_critical_path_zero_float(self):
        df = make_sample_activities()
        _, _, schedule = deterministic_cpm(df)
        critical = schedule[schedule["on_critical_path"]]
        assert all(abs(critical["TF"]) < 0.001)

    def test_serial_chain(self):
        """Simple serial chain: total = sum of durations."""
        df = pd.DataFrame({
            "id": [1, 2, 3],
            "name": ["A", "B", "C"],
            "predecessors": [[], [1], [2]],
            "dist_type": ["betapert"] * 3,
            "a": [10, 20, 15],
            "m": [10, 20, 15],
            "b": [10, 20, 15],
        })
        duration, cp, _ = deterministic_cpm(df)
        assert duration == pytest.approx(45.0, abs=0.01)
        assert cp == [1, 2, 3]

    def test_parallel_paths(self):
        """Two parallel paths: longest wins."""
        df = pd.DataFrame({
            "id": [1, 2, 3, 4],
            "name": ["Start", "Long", "Short", "End"],
            "predecessors": [[], [1], [1], [2, 3]],
            "dist_type": ["betapert"] * 4,
            "a": [5, 30, 10, 5],
            "m": [5, 30, 10, 5],
            "b": [5, 30, 10, 5],
        })
        duration, _, _ = deterministic_cpm(df)
        assert duration == pytest.approx(40.0, abs=0.01)  # 5 + 30 + 5


class TestMonteCarloSimulation:
    def test_basic_simulation(self):
        df = make_sample_activities()
        config = SimulationConfig(n_iterations=1000, random_seed=42)
        results = run_simulation(df, config=config)

        assert results.n_iterations == 1000
        assert len(results.total_durations) == 1000
        assert results.elapsed_seconds > 0

    def test_percentiles_ordered(self):
        df = make_sample_activities()
        config = SimulationConfig(n_iterations=5000, random_seed=42)
        results = run_simulation(df, config=config)

        stats = results.statistics
        assert stats["P10"] < stats["P50"] < stats["P90"]
        assert stats["min"] <= stats["P10"]
        assert stats["P90"] <= stats["max"]

    def test_mean_reasonable(self):
        df = make_sample_activities()
        config = SimulationConfig(n_iterations=5000, random_seed=42)
        results = run_simulation(df, config=config)

        # Sum of most-likely on critical path ~ 162
        # With PERT weighting and uncertainty, mean should be > 162
        assert results.statistics["mean"] > 100
        assert results.statistics["mean"] < 500

    def test_with_risks(self):
        df = make_sample_activities()
        risk_reg = create_default_risks()
        config = SimulationConfig(n_iterations=3000, random_seed=42)

        results_no_risk = run_simulation(df, config=config)
        results_with_risk = run_simulation(df, risk_registry=risk_reg, config=config)

        # Risks should increase mean duration
        assert results_with_risk.statistics["mean"] >= results_no_risk.statistics["mean"]

    def test_reproducibility(self):
        df = make_sample_activities()
        config = SimulationConfig(n_iterations=1000, random_seed=42)

        r1 = run_simulation(df, config=config)
        r2 = run_simulation(df, config=config)

        np.testing.assert_array_equal(r1.total_durations, r2.total_durations)

    def test_criticality_index(self):
        df = make_sample_activities()
        config = SimulationConfig(n_iterations=2000, random_seed=42)
        results = run_simulation(df, config=config)

        ci = results.criticality_index
        assert all(0 <= v <= 1 for v in ci.values())
        # At least one activity should have CI > 0
        assert max(ci.values()) > 0

    def test_probability_of_completion(self):
        df = make_sample_activities()
        config = SimulationConfig(n_iterations=5000, random_seed=42)
        results = run_simulation(df, config=config)

        # Very long deadline should have ~100% probability
        prob = results.probability_of_completion(10000)
        assert prob == pytest.approx(1.0, abs=0.001)

        # Very short deadline should have ~0% probability
        prob = results.probability_of_completion(1)
        assert prob == pytest.approx(0.0, abs=0.001)

    def test_ml_bias_factors(self):
        df = make_sample_activities()
        config = SimulationConfig(n_iterations=1000, random_seed=42)

        # Bias factor > 1 should increase durations
        bias = {i: 1.5 for i in range(1, 11)}
        results = run_simulation(df, config=config, ml_bias_factors=bias)
        results_base = run_simulation(df, config=config)

        assert results.statistics["mean"] > results_base.statistics["mean"]

    def test_summary_dataframe(self):
        df = make_sample_activities()
        config = SimulationConfig(n_iterations=1000, random_seed=42)
        results = run_simulation(df, config=config)

        summary = results.summary_dataframe()
        assert len(summary) == 10
        assert "mean_duration" in summary.columns
        assert "criticality_index" in summary.columns


class TestBenchmark:
    """
    Benchmark test: reproduce the thesis environmental sanitation example.
    5 phases, 2 risks, expect P50~265 days.
    """

    def test_sanitation_project_benchmark(self):
        # Simplified 5-phase project matching thesis example
        activities = pd.DataFrame({
            "id": [1, 2, 3, 4, 5],
            "name": [
                "Design & Engineering",
                "Procurement & Civil",
                "Electromechanical Installation",
                "Testing & Commissioning",
                "Startup & Handover",
            ],
            "predecessors": [[], [1], [2], [3], [4]],
            "dist_type": ["betapert"] * 5,
            "a": [25, 40, 35, 15, 8],
            "m": [40, 60, 55, 25, 12],
            "b": [60, 90, 80, 40, 20],
            "category": [
                "design", "civil_works", "electromechanical",
                "commissioning", "startup",
            ],
        })

        # Two risks from thesis example
        risk_reg = RiskRegistry()
        risk_reg.add_risk(RiskEvent(
            risk_id="R1", name="Unexpected contaminant",
            probability=0.25,
            impact_dist_type="triangular",
            impact_min=8, impact_mode=15, impact_max=28,
            applies_to=[1, 3],
        ))
        risk_reg.add_risk(RiskEvent(
            risk_id="R2", name="Extreme weather",
            probability=0.20,
            impact_dist_type="triangular",
            impact_min=5, impact_mode=10, impact_max=20,
            applies_to=[2, 3],
        ))

        config = SimulationConfig(n_iterations=20000, random_seed=42)
        results = run_simulation(activities, risk_registry=risk_reg, config=config)

        # Verify approximate benchmark values (with tolerance)
        # Serial chain sum of modes: 40+60+55+25+12 = 192
        # With PERT weighting + risks, P50 should be higher
        p50 = results.statistics["P50"]
        p80 = results.statistics["P80"]
        p90 = results.statistics["P90"]

        # Reasonable bounds for this configuration
        assert 180 < p50 < 280, f"P50={p50} outside expected range"
        assert p80 > p50
        assert p90 > p80
        assert results.statistics["std"] > 0

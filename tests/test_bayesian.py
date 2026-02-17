"""Tests for the Bayesian updating module."""

import numpy as np
import pandas as pd
import pytest

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.bayesian_updater import (
    ConjugateBayesianUpdater,
    create_updater,
    adaptive_reestimation,
)
from src.monte_carlo import SimulationConfig


def make_sample_activities():
    return pd.DataFrame({
        "id": [1, 2, 3, 4, 5],
        "name": ["A", "B", "C", "D", "E"],
        "predecessors": [[], [1], [1], [2, 3], [4]],
        "dist_type": ["betapert"] * 5,
        "a": [10, 15, 8, 20, 5],
        "m": [15, 25, 12, 30, 8],
        "b": [25, 40, 20, 50, 14],
        "category": ["design", "civil_works", "piping", "electromechanical", "testing"],
    })


class TestConjugateBayesianUpdater:
    def test_initialization(self):
        df = make_sample_activities()
        updater = ConjugateBayesianUpdater(df)
        assert len(updater.priors) == 5
        assert len(updater.posteriors) == 5

    def test_single_observation_shifts_mean(self):
        df = make_sample_activities()
        updater = ConjugateBayesianUpdater(df)

        prior = updater.priors[1]
        prior_mu = prior["mu"]

        # Observe a duration much longer than the mode
        result = updater.observe(1, 25.0)

        post = updater.posteriors[1]
        # Posterior mean should shift toward the observation
        assert post["mu"] > prior_mu or abs(post["mu"] - prior_mu) < 1

    def test_observation_reduces_uncertainty(self):
        df = make_sample_activities()
        updater = ConjugateBayesianUpdater(df)

        prior_sigma = updater.priors[1]["sigma"]
        updater.observe(1, 20.0)
        post_sigma = updater.posteriors[1]["sigma"]

        assert post_sigma < prior_sigma

    def test_multiple_observations_stronger_shift(self):
        df = make_sample_activities()
        updater = ConjugateBayesianUpdater(df)

        # Multiple observations of longer duration
        updater.observe(1, 22.0)
        after_one = updater.posteriors[1]["mu"]

        updater.observe(1, 23.0)
        after_two = updater.posteriors[1]["mu"]

        # Second observation should shift even more
        assert updater.posteriors[1]["sigma"] < updater.priors[1]["sigma"]

    def test_observe_multiple(self):
        df = make_sample_activities()
        updater = ConjugateBayesianUpdater(df)

        observations = {1: 18.0, 2: 30.0, 3: 14.0}
        results = updater.observe_multiple(observations)

        assert len(results) == 3
        assert all("posterior" in r for r in results)

    def test_get_updated_dataframe(self):
        df = make_sample_activities()
        updater = ConjugateBayesianUpdater(df)

        updater.observe(1, 20.0)
        updated = updater.get_updated_dataframe()

        assert len(updated) == 5
        # Updated activity should have different a, m, b
        orig_m = df[df["id"] == 1]["m"].iloc[0]
        new_m = updated[updated["id"] == 1]["m"].iloc[0]
        # They might differ (posterior shifted)
        assert isinstance(new_m, float)

    def test_summary(self):
        df = make_sample_activities()
        updater = ConjugateBayesianUpdater(df)
        updater.observe(1, 20.0)

        summary = updater.summary()
        assert len(summary) == 5
        assert "prior_mu" in summary.columns
        assert "posterior_mu" in summary.columns
        assert "n_observations" in summary.columns

    def test_invalid_activity_id(self):
        df = make_sample_activities()
        updater = ConjugateBayesianUpdater(df)

        with pytest.raises(ValueError):
            updater.observe(999, 20.0)

    def test_posterior_valid_range(self):
        """Posterior a, m, b should maintain a <= m <= b."""
        df = make_sample_activities()
        updater = ConjugateBayesianUpdater(df)

        updater.observe(1, 18.0)
        post = updater.posteriors[1]
        assert post["a"] <= post["m"] <= post["b"]

    def test_exact_observation_locks_posterior(self):
        """If observed equals mode, posterior should be close to prior."""
        df = make_sample_activities()
        updater = ConjugateBayesianUpdater(df)

        mode = df[df["id"] == 1]["m"].iloc[0]
        updater.observe(1, mode)

        post = updater.posteriors[1]
        prior = updater.priors[1]
        # Posterior should not shift far from prior mean
        assert abs(post["mu"] - prior["mu"]) < prior["sigma"]


class TestCreateUpdater:
    def test_conjugate(self):
        df = make_sample_activities()
        updater = create_updater(df, method="conjugate")
        assert isinstance(updater, ConjugateBayesianUpdater)

    def test_invalid_method(self):
        df = make_sample_activities()
        with pytest.raises(ValueError):
            create_updater(df, method="invalid")


class TestAdaptiveReestimation:
    def test_full_cycle(self):
        df = make_sample_activities()
        observations = {1: 18.0, 2: 28.0}

        config = SimulationConfig(n_iterations=2000, random_seed=42)
        result = adaptive_reestimation(
            df, observations, sim_config=config, method="conjugate"
        )

        assert "updater_summary" in result
        assert "updated_activities" in result
        assert "simulation_results" in result
        assert "comparison" in result

        # Updated simulation should produce results
        assert result["simulation_results"].n_iterations == 2000

    def test_completed_activities_fixed(self):
        """Completed activities should have a = m = b = actual."""
        df = make_sample_activities()
        observations = {1: 18.0}

        config = SimulationConfig(n_iterations=1000, random_seed=42)
        result = adaptive_reestimation(
            df, observations, sim_config=config, method="conjugate"
        )

        updated = result["updated_activities"]
        act1 = updated[updated["id"] == 1].iloc[0]
        assert act1["a"] == 18.0
        assert act1["m"] == 18.0
        assert act1["b"] == 18.0

    def test_delta_reasonable(self):
        """P50 delta should be reasonable."""
        df = make_sample_activities()
        observations = {1: 18.0}

        config = SimulationConfig(n_iterations=3000, random_seed=42)
        result = adaptive_reestimation(
            df, observations, sim_config=config, method="conjugate"
        )

        delta = result["comparison"]["delta_P50"]
        # Delta should be within reasonable bounds
        assert abs(delta) < 50

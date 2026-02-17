"""Tests for the Machine Learning module."""

import numpy as np
import pandas as pd
import pytest

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ml_module import (
    DurationPredictor,
    build_training_data,
    create_synthetic_training_data,
)


def make_sample_activities():
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


class TestSyntheticData:
    def test_create_synthetic(self):
        df = make_sample_activities()
        synthetic = create_synthetic_training_data(df, n_synthetic_projects=10)

        assert len(synthetic) == 10 * 10  # 10 projects * 10 activities
        assert "actual_duration" in synthetic.columns
        assert "project_id" in synthetic.columns

    def test_synthetic_reasonable_values(self):
        df = make_sample_activities()
        synthetic = create_synthetic_training_data(df, n_synthetic_projects=5)

        # All actual durations should be positive
        assert (synthetic["actual_duration"] > 0).all()


class TestBuildTrainingData:
    def test_build_with_actual_durations(self):
        df = make_sample_activities()
        actual = {i: float(df[df["id"] == i]["m"].iloc[0]) * 1.1 for i in range(1, 11)}

        training = build_training_data(df, actual_durations=actual)
        assert "actual_duration" in training.columns
        assert "complexity_score" in training.columns
        assert "vendor_lead_time" in training.columns

    def test_derived_features_added(self):
        df = make_sample_activities()
        training = build_training_data(df)

        assert "weather_index" in training.columns
        assert "regulatory_risk" in training.columns
        assert "team_experience" in training.columns


class TestDurationPredictor:
    def test_create_random_forest(self):
        predictor = DurationPredictor(model_type="random_forest")
        assert predictor.model is not None
        assert not predictor.is_fitted

    def test_create_neural_network(self):
        predictor = DurationPredictor(model_type="neural_network")
        assert predictor.model is not None

    def test_invalid_model_type(self):
        with pytest.raises(ValueError):
            DurationPredictor(model_type="svm")

    def test_train_random_forest(self):
        df = make_sample_activities()
        synthetic = create_synthetic_training_data(df, n_synthetic_projects=20)
        training = build_training_data(synthetic)

        predictor = DurationPredictor(model_type="random_forest")
        metrics = predictor.train(training)

        assert predictor.is_fitted
        assert "train_mae" in metrics
        assert "train_r2" in metrics
        assert "feature_importance" in metrics
        assert metrics["train_mae"] >= 0

    def test_train_neural_network(self):
        df = make_sample_activities()
        synthetic = create_synthetic_training_data(df, n_synthetic_projects=20)
        training = build_training_data(synthetic)

        predictor = DurationPredictor(model_type="neural_network")
        metrics = predictor.train(training)

        assert predictor.is_fitted
        assert metrics["train_mae"] >= 0

    def test_predict_bias_factors(self):
        df = make_sample_activities()
        synthetic = create_synthetic_training_data(df, n_synthetic_projects=20)
        training = build_training_data(synthetic)

        predictor = DurationPredictor(model_type="random_forest")
        predictor.train(training)

        bias = predictor.predict_bias_factors(df)
        assert len(bias) == 10
        # Bias factors should be reasonable (0.5 - 2.5)
        for aid, factor in bias.items():
            assert 0.5 <= factor <= 2.5, f"Activity {aid}: bias={factor} out of range"

    def test_predict_without_training(self):
        df = make_sample_activities()
        predictor = DurationPredictor()

        bias = predictor.predict_bias_factors(df)
        # Should return 1.0 for all (neutral)
        assert all(v == 1.0 for v in bias.values())

    def test_feature_importance(self):
        df = make_sample_activities()
        synthetic = create_synthetic_training_data(df, n_synthetic_projects=20)
        training = build_training_data(synthetic)

        predictor = DurationPredictor(model_type="random_forest")
        predictor.train(training)

        importance = predictor.get_feature_importance()
        assert importance is not None
        assert len(importance) > 0
        assert "feature" in importance.columns
        assert "importance" in importance.columns
        # Importances should sum to ~1
        assert importance["importance"].sum() == pytest.approx(1.0, abs=0.01)

    def test_cross_validate(self):
        df = make_sample_activities()
        synthetic = create_synthetic_training_data(df, n_synthetic_projects=20)
        training = build_training_data(synthetic)

        predictor = DurationPredictor(model_type="random_forest")
        cv_result = predictor.cross_validate(training, cv_folds=3)

        assert "cv_mae_mean" in cv_result
        assert "cv_scores" in cv_result
        assert cv_result["cv_mae_mean"] >= 0

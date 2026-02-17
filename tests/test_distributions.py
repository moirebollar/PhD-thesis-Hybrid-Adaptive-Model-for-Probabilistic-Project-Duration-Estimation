"""Tests for the distributions module."""

import numpy as np
import pytest

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.distributions import (
    TriangularDist,
    BetaPERT,
    create_distribution,
    fit_triangular_mle,
    fit_betapert_mle,
    auto_select_distribution,
    compute_moments,
)


class TestTriangularDist:
    def test_basic_creation(self):
        dist = TriangularDist(10, 20, 30)
        assert dist.a == 10
        assert dist.m == 20
        assert dist.b == 30

    def test_mean(self):
        dist = TriangularDist(10, 20, 30)
        assert dist.mean == pytest.approx(20.0, abs=0.01)

    def test_symmetric_mean(self):
        dist = TriangularDist(0, 50, 100)
        assert dist.mean == pytest.approx(50.0, abs=0.01)

    def test_sample_bounds(self):
        dist = TriangularDist(10, 20, 30)
        rng = np.random.default_rng(42)
        samples = dist.sample(10000, rng=rng)
        assert np.all(samples >= 10)
        assert np.all(samples <= 30)

    def test_sample_mean_convergence(self):
        dist = TriangularDist(10, 20, 30)
        rng = np.random.default_rng(42)
        samples = dist.sample(50000, rng=rng)
        assert np.mean(samples) == pytest.approx(dist.mean, abs=0.2)

    def test_invalid_params(self):
        with pytest.raises(ValueError):
            TriangularDist(30, 20, 10)  # a > m

    def test_degenerate(self):
        with pytest.raises(ValueError):
            TriangularDist(10, 10, 10)  # a == b

    def test_cdf_properties(self):
        dist = TriangularDist(10, 20, 30)
        assert dist.cdf(np.array([10.0])) == pytest.approx(0.0, abs=0.001)
        assert dist.cdf(np.array([30.0])) == pytest.approx(1.0, abs=0.001)

    def test_percentile(self):
        dist = TriangularDist(10, 20, 30)
        p50 = dist.percentile(0.5)
        assert 10 < p50 < 30

    def test_variance_positive(self):
        dist = TriangularDist(10, 20, 30)
        assert dist.variance > 0
        assert dist.std > 0


class TestBetaPERT:
    def test_basic_creation(self):
        dist = BetaPERT(10, 20, 30)
        assert dist.a == 10
        assert dist.m == 20
        assert dist.b == 30
        assert dist.lam == 4.0

    def test_pert_mean(self):
        """BetaPERT mean should be (a + 4m + b) / 6."""
        dist = BetaPERT(10, 20, 30, lam=4)
        expected_mean = (10 + 4 * 20 + 30) / 6
        assert dist.mean == pytest.approx(expected_mean, abs=0.01)

    def test_sample_bounds(self):
        dist = BetaPERT(10, 20, 30)
        rng = np.random.default_rng(42)
        samples = dist.sample(10000, rng=rng)
        assert np.all(samples >= 10)
        assert np.all(samples <= 30)

    def test_sample_mean_convergence(self):
        dist = BetaPERT(10, 20, 30)
        rng = np.random.default_rng(42)
        samples = dist.sample(50000, rng=rng)
        assert np.mean(samples) == pytest.approx(dist.mean, abs=0.2)

    def test_mode_near_m(self):
        """For symmetric PERT, mode should be near m."""
        dist = BetaPERT(10, 20, 30)
        rng = np.random.default_rng(42)
        samples = dist.sample(100000, rng=rng)
        # Mode estimate from histogram
        hist, edges = np.histogram(samples, bins=100)
        mode_idx = np.argmax(hist)
        mode_est = (edges[mode_idx] + edges[mode_idx + 1]) / 2
        assert mode_est == pytest.approx(20.0, abs=1.5)

    def test_lambda_effect(self):
        """Higher lambda should concentrate distribution more around mode."""
        dist_low = BetaPERT(10, 20, 30, lam=2)
        dist_high = BetaPERT(10, 20, 30, lam=8)
        assert dist_high.std < dist_low.std

    def test_invalid_params(self):
        with pytest.raises(ValueError):
            BetaPERT(30, 20, 10)

    def test_skewed_distribution(self):
        """Asymmetric PERT should produce skewed samples."""
        dist = BetaPERT(10, 15, 50)  # right-skewed
        assert dist.mean > 15  # mean pulled toward b

    def test_alpha_beta_positive(self):
        dist = BetaPERT(10, 20, 30)
        assert dist.alpha > 0
        assert dist.beta_param > 0

    def test_percentile(self):
        dist = BetaPERT(10, 20, 30)
        p50 = dist.percentile(0.5)
        assert 10 < p50 < 30
        # P50 should be close to mean for symmetric PERT
        assert p50 == pytest.approx(dist.mean, abs=1.0)


class TestCreateDistribution:
    def test_create_triangular(self):
        dist = create_distribution("triangular", 10, 20, 30)
        assert isinstance(dist, TriangularDist)

    def test_create_betapert(self):
        dist = create_distribution("betapert", 10, 20, 30)
        assert isinstance(dist, BetaPERT)

    def test_case_insensitive(self):
        dist = create_distribution("BetaPERT", 10, 20, 30)
        assert isinstance(dist, BetaPERT)

    def test_aliases(self):
        assert isinstance(create_distribution("pert", 10, 20, 30), BetaPERT)
        assert isinstance(create_distribution("tri", 10, 20, 30), TriangularDist)

    def test_unknown_type(self):
        with pytest.raises(ValueError):
            create_distribution("gaussian", 10, 20, 30)


class TestFitting:
    def test_fit_triangular(self):
        # Generate data from known triangular
        rng = np.random.default_rng(42)
        true_dist = TriangularDist(10, 20, 30)
        data = true_dist.sample(1000, rng=rng)

        fitted = fit_triangular_mle(data)
        assert fitted.a == pytest.approx(10, abs=2)
        assert fitted.m == pytest.approx(20, abs=3)
        assert fitted.b == pytest.approx(30, abs=2)

    def test_fit_betapert(self):
        rng = np.random.default_rng(42)
        true_dist = BetaPERT(10, 20, 30)
        data = true_dist.sample(1000, rng=rng)

        fitted = fit_betapert_mle(data)
        assert fitted.a < fitted.m < fitted.b
        assert fitted.mean == pytest.approx(true_dist.mean, abs=2)

    def test_auto_select_with_data(self):
        rng = np.random.default_rng(42)
        data = BetaPERT(10, 20, 30).sample(100, rng=rng)
        dist = auto_select_distribution(10, 20, 30, data=data, min_samples=10)
        assert dist is not None

    def test_auto_select_without_data(self):
        dist = auto_select_distribution(10, 20, 30)
        assert isinstance(dist, BetaPERT)


class TestMoments:
    def test_compute_moments(self):
        dist = BetaPERT(10, 20, 30)
        moments = compute_moments(dist)
        assert "mean" in moments
        assert "std" in moments
        assert "p50" in moments
        assert moments["p10"] < moments["p50"] < moments["p90"]

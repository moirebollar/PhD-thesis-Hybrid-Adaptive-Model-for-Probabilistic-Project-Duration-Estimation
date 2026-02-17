"""
Distribution Module.

Implements Triangular and BetaPERT distributions for activity duration
sampling, plus MLE fitting and automatic distribution selection.
"""

from typing import Optional

import numpy as np
from scipy import stats
from scipy.optimize import minimize


class TriangularDist:
    """Triangular distribution parameterized by (a, m, b)."""

    def __init__(self, a: float, m: float, b: float):
        if not (a <= m <= b):
            raise ValueError(f"Requires a <= m <= b, got a={a}, m={m}, b={b}")
        if a == b:
            raise ValueError(f"Degenerate distribution: a == b == {a}")
        self.a = float(a)
        self.m = float(m)
        self.b = float(b)

    @property
    def mean(self) -> float:
        return (self.a + self.m + self.b) / 3.0

    @property
    def variance(self) -> float:
        a, m, b = self.a, self.m, self.b
        return (a**2 + m**2 + b**2 - a*m - a*b - m*b) / 18.0

    @property
    def std(self) -> float:
        return np.sqrt(self.variance)

    def sample(self, n: int = 1, rng: Optional[np.random.Generator] = None) -> np.ndarray:
        """Draw n samples from the triangular distribution."""
        if rng is None:
            rng = np.random.default_rng()
        # scipy uses c = (m - a) / (b - a) as the mode parameter
        c = (self.m - self.a) / (self.b - self.a)
        return stats.triang.rvs(c, loc=self.a, scale=self.b - self.a, size=n,
                                random_state=rng)

    def pdf(self, x: np.ndarray) -> np.ndarray:
        c = (self.m - self.a) / (self.b - self.a)
        return stats.triang.pdf(x, c, loc=self.a, scale=self.b - self.a)

    def cdf(self, x: np.ndarray) -> np.ndarray:
        c = (self.m - self.a) / (self.b - self.a)
        return stats.triang.cdf(x, c, loc=self.a, scale=self.b - self.a)

    def percentile(self, p: float) -> float:
        """Return the value at the p-th percentile (p in 0-1)."""
        c = (self.m - self.a) / (self.b - self.a)
        return stats.triang.ppf(p, c, loc=self.a, scale=self.b - self.a)

    def __repr__(self) -> str:
        return f"Triangular(a={self.a}, m={self.m}, b={self.b})"


class BetaPERT:
    """
    BetaPERT (Program Evaluation and Review Technique) distribution.

    Modified Beta distribution parameterized by (a, m, b) with
    lambda weighting on the mode.

    Mean = (a + lambda*m + b) / (lambda + 2)
    """

    def __init__(self, a: float, m: float, b: float, lam: float = 4.0):
        if not (a <= m <= b):
            raise ValueError(f"Requires a <= m <= b, got a={a}, m={m}, b={b}")
        if a == b:
            raise ValueError(f"Degenerate distribution: a == b == {a}")
        self.a = float(a)
        self.m = float(m)
        self.b = float(b)
        self.lam = float(lam)

        # Calculate Beta distribution parameters
        self._mu = (self.a + self.lam * self.m + self.b) / (self.lam + 2)

        # Standard deviation using PERT formula
        self._sigma = (self.b - self.a) / (self.lam + 2)

        # Convert to standard Beta(alpha, beta) on [a, b]
        if self._sigma > 0:
            # Method: match mean and variance to Beta distribution on [a, b]
            mu_std = (self._mu - self.a) / (self.b - self.a)  # standardize to [0,1]
            var_std = self._sigma**2 / (self.b - self.a)**2

            # Ensure valid parameters
            if mu_std <= 0:
                mu_std = 0.001
            if mu_std >= 1:
                mu_std = 0.999

            # Alpha and Beta from mean and variance of standard Beta
            if var_std > 0 and var_std < mu_std * (1 - mu_std):
                common = mu_std * (1 - mu_std) / var_std - 1
                self.alpha = mu_std * common
                self.beta_param = (1 - mu_std) * common
            else:
                # Fallback: use lambda-based formula
                self.alpha = 1 + self.lam * (self.m - self.a) / (self.b - self.a)
                self.beta_param = 1 + self.lam * (self.b - self.m) / (self.b - self.a)
        else:
            self.alpha = 1.0
            self.beta_param = 1.0

        # Ensure alpha, beta > 0
        self.alpha = max(self.alpha, 0.01)
        self.beta_param = max(self.beta_param, 0.01)

    @property
    def mean(self) -> float:
        return self._mu

    @property
    def variance(self) -> float:
        return self._sigma**2

    @property
    def std(self) -> float:
        return self._sigma

    def sample(self, n: int = 1, rng: Optional[np.random.Generator] = None) -> np.ndarray:
        """Draw n samples from the BetaPERT distribution."""
        if rng is None:
            rng = np.random.default_rng()
        # Sample from Beta(alpha, beta) and scale to [a, b]
        raw = rng.beta(self.alpha, self.beta_param, size=n)
        return self.a + raw * (self.b - self.a)

    def pdf(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        # Standardize to [0, 1]
        x_std = (x - self.a) / (self.b - self.a)
        return stats.beta.pdf(x_std, self.alpha, self.beta_param) / (self.b - self.a)

    def cdf(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        x_std = (x - self.a) / (self.b - self.a)
        return stats.beta.cdf(x_std, self.alpha, self.beta_param)

    def percentile(self, p: float) -> float:
        """Return the value at the p-th percentile (p in 0-1)."""
        x_std = stats.beta.ppf(p, self.alpha, self.beta_param)
        return self.a + x_std * (self.b - self.a)

    def __repr__(self) -> str:
        return (f"BetaPERT(a={self.a}, m={self.m}, b={self.b}, "
                f"lambda={self.lam}, alpha={self.alpha:.3f}, beta={self.beta_param:.3f})")


def create_distribution(dist_type: str, a: float, m: float, b: float,
                        lam: float = 4.0):
    """
    Factory function to create a distribution object.

    Parameters
    ----------
    dist_type : str
        'triangular' or 'betapert'.
    a, m, b : float
        Optimistic, most likely, pessimistic values.
    lam : float
        Lambda parameter for BetaPERT (default: 4).

    Returns
    -------
    TriangularDist or BetaPERT
    """
    dist_type = dist_type.lower().strip()
    if dist_type in ("triangular", "triang", "tri"):
        return TriangularDist(a, m, b)
    elif dist_type in ("betapert", "pert", "beta_pert", "beta"):
        return BetaPERT(a, m, b, lam=lam)
    else:
        raise ValueError(f"Unknown distribution type: {dist_type}. "
                        f"Use 'triangular' or 'betapert'.")


def fit_triangular_mle(data: np.ndarray) -> TriangularDist:
    """
    Fit a Triangular distribution to observed data using MLE.

    Parameters
    ----------
    data : np.ndarray
        Observed duration samples.

    Returns
    -------
    TriangularDist
        Fitted distribution.
    """
    data = np.asarray(data, dtype=float)

    # Use scipy's built-in MLE fitting
    c, loc, scale = stats.triang.fit(data)
    a = loc
    b = loc + scale
    m = a + c * scale

    return TriangularDist(a, m, b)


def fit_betapert_mle(data: np.ndarray, lam: float = 4.0) -> BetaPERT:
    """
    Fit a BetaPERT distribution to observed data using MLE.

    Estimates (a, m, b) by fitting a Beta distribution and
    back-calculating the PERT parameters.

    Parameters
    ----------
    data : np.ndarray
        Observed duration samples.
    lam : float
        Lambda parameter (default: 4).

    Returns
    -------
    BetaPERT
        Fitted distribution.
    """
    data = np.asarray(data, dtype=float)

    a_est = data.min() * 0.95  # slight padding
    b_est = data.max() * 1.05

    # Standardize to [0, 1]
    data_std = (data - a_est) / (b_est - a_est)
    data_std = np.clip(data_std, 0.001, 0.999)

    # Fit Beta distribution
    alpha_fit, beta_fit, _, _ = stats.beta.fit(data_std, floc=0, fscale=1)

    # Back-calculate mode from Beta parameters
    if alpha_fit > 1 and beta_fit > 1:
        mode_std = (alpha_fit - 1) / (alpha_fit + beta_fit - 2)
    else:
        mode_std = alpha_fit / (alpha_fit + beta_fit)

    m_est = a_est + mode_std * (b_est - a_est)

    return BetaPERT(a_est, m_est, b_est, lam=lam)


def auto_select_distribution(a: float, m: float, b: float,
                             data: Optional[np.ndarray] = None,
                             min_samples: int = 10,
                             lam: float = 4.0):
    """
    Automatically select and parameterize the best distribution.

    Strategy:
    - If sufficient historical data exists, fit both distributions and
      select based on log-likelihood.
    - Otherwise, default to BetaPERT (generally better for project estimation).

    Parameters
    ----------
    a, m, b : float
        Three-point estimates.
    data : np.ndarray, optional
        Historical duration observations for this activity.
    min_samples : int
        Minimum samples required for MLE fitting.
    lam : float
        Lambda parameter for BetaPERT.

    Returns
    -------
    distribution object
        Best-fitting distribution.
    """
    if data is not None and len(data) >= min_samples:
        # Fit both and compare
        try:
            tri_fit = fit_triangular_mle(data)
            pert_fit = fit_betapert_mle(data, lam=lam)

            # Compare log-likelihoods
            ll_tri = np.sum(np.log(tri_fit.pdf(data) + 1e-300))
            ll_pert = np.sum(np.log(pert_fit.pdf(data) + 1e-300))

            if ll_tri > ll_pert:
                return tri_fit
            else:
                return pert_fit
        except Exception:
            pass

    # Default to BetaPERT with given parameters
    return BetaPERT(a, m, b, lam=lam)


def compute_moments(dist) -> dict:
    """Compute and return statistical moments for a distribution."""
    samples = dist.sample(100000, rng=np.random.default_rng(0))
    return {
        "mean": dist.mean,
        "std": dist.std,
        "variance": dist.variance,
        "skewness": float(stats.skew(samples)),
        "kurtosis": float(stats.kurtosis(samples)),
        "p10": float(np.percentile(samples, 10)),
        "p50": float(np.percentile(samples, 50)),
        "p90": float(np.percentile(samples, 90)),
    }

"""
Bayesian Updating Module.

When actual progress data arrives during project execution, this module
updates activity duration distributions dynamically using Bayesian
inference (conjugate priors or PyMC MCMC).
"""

import json
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

from .distributions import BetaPERT, TriangularDist, create_distribution


class ConjugateBayesianUpdater:
    """
    Fast Bayesian updating using conjugate priors.

    Uses a Normal-Normal conjugate model:
    - Prior: N(mu_prior, sigma_prior^2) derived from BetaPERT/Triangular parameters
    - Likelihood: N(x_obs, sigma_obs^2) from observed durations
    - Posterior: N(mu_post, sigma_post^2) analytically computed

    The posterior is then mapped back to (a, m, b) parameters for
    re-simulation.
    """

    def __init__(self, activities_df: pd.DataFrame, pert_lambda: float = 4.0):
        """
        Parameters
        ----------
        activities_df : pd.DataFrame
            Activity data with columns: id, a, m, b, dist_type.
        pert_lambda : float
            Lambda for BetaPERT distributions.
        """
        self.activities = activities_df.copy()
        self.pert_lambda = pert_lambda
        self.priors = {}
        self.posteriors = {}
        self.observations = {}

        # Initialize priors from activity estimates
        for _, row in activities_df.iterrows():
            aid = row["id"]
            a, m, b = float(row["a"]), float(row["m"]), float(row["b"])
            dist = create_distribution(
                row.get("dist_type", "betapert"), a, m, b, lam=pert_lambda
            )
            self.priors[aid] = {
                "mu": dist.mean,
                "sigma": dist.std,
                "a": a,
                "m": m,
                "b": b,
                "dist_type": row.get("dist_type", "betapert"),
            }
            self.posteriors[aid] = self.priors[aid].copy()
            self.observations[aid] = []

    def observe(self, activity_id: int, observed_duration: float) -> dict:
        """
        Update the posterior for an activity given an observed duration.

        Parameters
        ----------
        activity_id : int
        observed_duration : float
            Actual observed duration for this activity.

        Returns
        -------
        dict with prior and posterior parameters.
        """
        if activity_id not in self.priors:
            raise ValueError(f"Activity {activity_id} not found in priors.")

        self.observations[activity_id].append(observed_duration)
        obs = np.array(self.observations[activity_id])

        prior = self.priors[activity_id]
        mu_prior = prior["mu"]
        sigma_prior = prior["sigma"]

        # Observation statistics
        n = len(obs)
        x_bar = np.mean(obs)
        # Use prior sigma as observation noise estimate (single observation case)
        sigma_obs = sigma_prior if n == 1 else np.std(obs, ddof=1)
        sigma_obs = max(sigma_obs, 0.1)  # prevent degenerate

        # Conjugate Normal-Normal update
        precision_prior = 1.0 / (sigma_prior ** 2)
        precision_obs = n / (sigma_obs ** 2)
        precision_post = precision_prior + precision_obs

        mu_post = (precision_prior * mu_prior + precision_obs * x_bar) / precision_post
        sigma_post = np.sqrt(1.0 / precision_post)

        # Map posterior back to (a, m, b)
        a_orig, b_orig = prior["a"], prior["b"]
        range_orig = b_orig - a_orig

        # Adjust range based on uncertainty reduction
        uncertainty_ratio = sigma_post / sigma_prior
        range_post = range_orig * uncertainty_ratio

        a_post = mu_post - range_post / 2
        b_post = mu_post + range_post / 2
        m_post = mu_post  # posterior mode ≈ posterior mean for Normal

        # Ensure a <= m <= b and positive
        a_post = max(a_post, 0.1)
        m_post = max(m_post, a_post)
        b_post = max(b_post, m_post)

        self.posteriors[activity_id] = {
            "mu": mu_post,
            "sigma": sigma_post,
            "a": round(a_post, 2),
            "m": round(m_post, 2),
            "b": round(b_post, 2),
            "dist_type": prior["dist_type"],
            "n_observations": n,
            "uncertainty_reduction": 1 - uncertainty_ratio,
        }

        return {
            "activity_id": activity_id,
            "prior": {
                "mu": mu_prior, "sigma": sigma_prior,
                "a": prior["a"], "m": prior["m"], "b": prior["b"],
            },
            "posterior": self.posteriors[activity_id],
            "observed": observed_duration,
            "shift": mu_post - mu_prior,
        }

    def observe_multiple(
        self, observations: dict[int, float]
    ) -> list[dict]:
        """
        Update posteriors for multiple activities at once.

        Parameters
        ----------
        observations : dict
            Mapping of activity_id -> observed_duration.

        Returns
        -------
        list of update result dicts.
        """
        results = []
        for aid, duration in observations.items():
            result = self.observe(aid, duration)
            results.append(result)
        return results

    def get_updated_dataframe(self) -> pd.DataFrame:
        """
        Return an updated activities DataFrame with posterior parameters
        replacing the original (a, m, b).
        """
        df = self.activities.copy()
        # Ensure float dtype for estimate columns
        for col in ["a", "m", "b"]:
            if col in df.columns:
                df[col] = df[col].astype(float)
        for idx, row in df.iterrows():
            aid = row["id"]
            if aid in self.posteriors:
                post = self.posteriors[aid]
                df.at[idx, "a"] = post["a"]
                df.at[idx, "m"] = post["m"]
                df.at[idx, "b"] = post["b"]
        return df

    def summary(self) -> pd.DataFrame:
        """Summary of all priors and posteriors."""
        rows = []
        for aid in self.priors:
            prior = self.priors[aid]
            post = self.posteriors[aid]
            n_obs = len(self.observations[aid])
            rows.append({
                "activity_id": aid,
                "prior_a": prior["a"],
                "prior_m": prior["m"],
                "prior_b": prior["b"],
                "prior_mu": prior["mu"],
                "prior_sigma": prior["sigma"],
                "posterior_a": post["a"],
                "posterior_m": post["m"],
                "posterior_b": post["b"],
                "posterior_mu": post["mu"],
                "posterior_sigma": post["sigma"],
                "n_observations": n_obs,
                "observations": self.observations[aid],
                "uncertainty_reduction": post.get("uncertainty_reduction", 0),
            })
        return pd.DataFrame(rows)


class MCMCBayesianUpdater:
    """
    Full Bayesian updating using PyMC MCMC sampling.

    Uses a hierarchical model where each activity's duration is modeled
    with appropriate priors (LogNormal or Gamma) and updated with
    observed data.
    """

    def __init__(
        self,
        activities_df: pd.DataFrame,
        n_samples: int = 2000,
        n_chains: int = 2,
        n_tune: int = 1000,
        pert_lambda: float = 4.0,
    ):
        self.activities = activities_df.copy()
        self.n_samples = n_samples
        self.n_chains = n_chains
        self.n_tune = n_tune
        self.pert_lambda = pert_lambda
        self.posteriors = {}
        self.traces = {}

    def update_activity(
        self,
        activity_id: int,
        observed_durations: list[float],
    ) -> dict:
        """
        Run MCMC to update the posterior for a single activity.

        Parameters
        ----------
        activity_id : int
        observed_durations : list[float]
            One or more observed durations for this activity.

        Returns
        -------
        dict with posterior statistics and (a, m, b) estimates.
        """
        try:
            import pymc as pm
            import arviz as az
        except ImportError:
            warnings.warn(
                "PyMC not available. Falling back to conjugate updater."
            )
            return self._fallback_update(activity_id, observed_durations)

        row = self.activities[self.activities["id"] == activity_id].iloc[0]
        a_prior, m_prior, b_prior = float(row["a"]), float(row["m"]), float(row["b"])

        # Prior parameters for LogNormal
        mu_prior = (a_prior + self.pert_lambda * m_prior + b_prior) / (self.pert_lambda + 2)
        sigma_prior = (b_prior - a_prior) / (self.pert_lambda + 2)
        log_mu = np.log(mu_prior) - 0.5 * np.log(1 + (sigma_prior / mu_prior) ** 2)
        log_sigma = np.sqrt(np.log(1 + (sigma_prior / mu_prior) ** 2))

        obs = np.array(observed_durations)

        with pm.Model() as model:
            # LogNormal prior on activity duration
            duration = pm.LogNormal("duration", mu=log_mu, sigma=log_sigma)

            # Observation noise
            sigma_obs = pm.HalfNormal("sigma_obs", sigma=sigma_prior * 0.5)

            # Likelihood
            likelihood = pm.Normal("obs", mu=duration, sigma=sigma_obs,
                                   observed=obs)

            # Sample
            trace = pm.sample(
                draws=self.n_samples,
                chains=self.n_chains,
                tune=self.n_tune,
                return_inferencedata=True,
                progressbar=False,
            )

        # Extract posterior statistics
        post_samples = trace.posterior["duration"].values.flatten()
        mu_post = float(np.mean(post_samples))
        sigma_post = float(np.std(post_samples))

        # Map to (a, m, b)
        a_post = float(np.percentile(post_samples, 2.5))
        m_post = float(np.median(post_samples))
        b_post = float(np.percentile(post_samples, 97.5))

        self.posteriors[activity_id] = {
            "mu": mu_post,
            "sigma": sigma_post,
            "a": round(a_post, 2),
            "m": round(m_post, 2),
            "b": round(b_post, 2),
        }
        self.traces[activity_id] = trace

        return {
            "activity_id": activity_id,
            "posterior_mu": mu_post,
            "posterior_sigma": sigma_post,
            "a": a_post,
            "m": m_post,
            "b": b_post,
            "hdi_94": az.hdi(trace, var_names=["duration"], hdi_prob=0.94).to_dict(),
        }

    def _fallback_update(self, activity_id: int, observed_durations: list[float]) -> dict:
        """Fallback to conjugate updating if PyMC is not available."""
        conjugate = ConjugateBayesianUpdater(self.activities, self.pert_lambda)
        for dur in observed_durations:
            result = conjugate.observe(activity_id, dur)
        return result

    def get_updated_dataframe(self) -> pd.DataFrame:
        """Return activities DataFrame with MCMC posteriors."""
        df = self.activities.copy()
        for idx, row in df.iterrows():
            aid = row["id"]
            if aid in self.posteriors:
                post = self.posteriors[aid]
                df.at[idx, "a"] = post["a"]
                df.at[idx, "m"] = post["m"]
                df.at[idx, "b"] = post["b"]
        return df


def create_updater(
    activities_df: pd.DataFrame,
    method: str = "conjugate",
    **kwargs,
):
    """
    Factory function to create a Bayesian updater.

    Parameters
    ----------
    activities_df : pd.DataFrame
    method : str
        'conjugate' (fast, analytical) or 'mcmc' (full PyMC).
    **kwargs
        Passed to the updater constructor.

    Returns
    -------
    ConjugateBayesianUpdater or MCMCBayesianUpdater
    """
    if method == "conjugate":
        return ConjugateBayesianUpdater(activities_df, **kwargs)
    elif method == "mcmc":
        return MCMCBayesianUpdater(activities_df, **kwargs)
    else:
        raise ValueError(f"Unknown method: {method}. Use 'conjugate' or 'mcmc'.")


def adaptive_reestimation(
    activities_df: pd.DataFrame,
    completed_observations: dict[int, float],
    risk_registry=None,
    sim_config=None,
    method: str = "conjugate",
) -> dict:
    """
    Full adaptive re-estimation cycle:
    1. Update priors with observed data
    2. Re-run Monte Carlo with posterior distributions
    3. Return updated predictions

    Parameters
    ----------
    activities_df : pd.DataFrame
    completed_observations : dict
        {activity_id: observed_duration} for completed activities.
    risk_registry : RiskRegistry, optional
    sim_config : SimulationConfig, optional
    method : str
        Bayesian update method ('conjugate' or 'mcmc').

    Returns
    -------
    dict with:
        - 'updater_summary': DataFrame of prior/posterior comparison
        - 'updated_activities': DataFrame for re-simulation
        - 'simulation_results': new SimulationResults
        - 'comparison': before/after statistics
    """
    from .monte_carlo import run_simulation, SimulationConfig

    if sim_config is None:
        sim_config = SimulationConfig()

    # Run original simulation
    original_results = run_simulation(activities_df, risk_registry, sim_config)

    # Bayesian update
    updater = create_updater(activities_df, method=method)
    updater.observe_multiple(completed_observations)

    # Get updated DataFrame
    updated_df = updater.get_updated_dataframe()

    # Mark completed activities as fixed (zero variance)
    for aid, obs_dur in completed_observations.items():
        mask = updated_df["id"] == aid
        updated_df.loc[mask, "a"] = obs_dur
        updated_df.loc[mask, "m"] = obs_dur
        updated_df.loc[mask, "b"] = obs_dur

    # Re-run simulation with posteriors
    updated_results = run_simulation(updated_df, risk_registry, sim_config)

    return {
        "updater_summary": updater.summary(),
        "updated_activities": updated_df,
        "simulation_results": updated_results,
        "comparison": {
            "original": original_results.statistics,
            "updated": updated_results.statistics,
            "delta_P50": (
                updated_results.statistics["P50"] - original_results.statistics["P50"]
            ),
            "delta_P90": (
                updated_results.statistics["P90"] - original_results.statistics["P90"]
            ),
        },
    }

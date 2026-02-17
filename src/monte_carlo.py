"""
Monte Carlo Simulation Engine.

Core simulation loop with activity precedence network (CPM),
configurable iterations, critical path identification per iteration,
and output metrics (P50, P80, P90).
"""

import json
import time
from dataclasses import dataclass, field
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import Optional

import networkx as nx
import numpy as np
import pandas as pd

from .distributions import create_distribution
from .risks import RiskRegistry


@dataclass
class SimulationConfig:
    """Configuration for Monte Carlo simulation."""
    n_iterations: int = 10000
    random_seed: int = 42
    confidence_levels: list = field(default_factory=lambda: [0.50, 0.80, 0.90])
    convergence_check: bool = True
    convergence_tolerance: float = 0.005
    pert_lambda: float = 4.0
    n_parallel: int = 1  # number of parallel processes

    @classmethod
    def from_config_file(cls, config_path: Optional[str] = None) -> "SimulationConfig":
        if config_path is None:
            config_path = Path(__file__).parent.parent / "config.json"
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        sim_cfg = cfg.get("simulation", {})
        dist_cfg = cfg.get("distributions", {})
        return cls(
            n_iterations=sim_cfg.get("n_iterations", 10000),
            random_seed=sim_cfg.get("random_seed", 42),
            confidence_levels=sim_cfg.get("confidence_levels", [0.50, 0.80, 0.90]),
            convergence_check=sim_cfg.get("convergence_check", True),
            convergence_tolerance=sim_cfg.get("convergence_tolerance", 0.005),
            pert_lambda=dist_cfg.get("pert_lambda", 4.0),
        )


@dataclass
class SimulationResults:
    """Results from a Monte Carlo simulation run."""
    total_durations: np.ndarray  # array of total project duration per iteration
    activity_durations: dict  # activity_id -> array of sampled durations
    activity_starts: dict  # activity_id -> array of start times
    activity_finishes: dict  # activity_id -> array of finish times
    critical_path_counts: dict  # activity_id -> count of times on critical path
    risk_impacts: dict  # activity_id -> array of risk-induced delays
    n_iterations: int
    elapsed_seconds: float
    converged: bool
    convergence_at: Optional[int]  # iteration where convergence was achieved

    @property
    def percentiles(self) -> dict:
        """Compute standard percentiles."""
        return {
            "P10": float(np.percentile(self.total_durations, 10)),
            "P25": float(np.percentile(self.total_durations, 25)),
            "P50": float(np.percentile(self.total_durations, 50)),
            "P80": float(np.percentile(self.total_durations, 80)),
            "P90": float(np.percentile(self.total_durations, 90)),
            "P95": float(np.percentile(self.total_durations, 95)),
        }

    @property
    def statistics(self) -> dict:
        """Compute summary statistics."""
        return {
            "mean": float(np.mean(self.total_durations)),
            "median": float(np.median(self.total_durations)),
            "std": float(np.std(self.total_durations)),
            "min": float(np.min(self.total_durations)),
            "max": float(np.max(self.total_durations)),
            "cv": float(np.std(self.total_durations) / np.mean(self.total_durations)),
            **self.percentiles,
        }

    @property
    def criticality_index(self) -> dict:
        """Fraction of iterations where each activity was on the critical path."""
        return {
            aid: count / self.n_iterations
            for aid, count in self.critical_path_counts.items()
        }

    def probability_of_completion(self, target_duration: float) -> float:
        """Probability that project completes within target_duration."""
        return float(np.mean(self.total_durations <= target_duration))

    def duration_at_probability(self, probability: float) -> float:
        """Duration that gives the specified probability of completion."""
        return float(np.percentile(self.total_durations, probability * 100))

    def summary_dataframe(self) -> pd.DataFrame:
        """Return a summary DataFrame of per-activity statistics."""
        rows = []
        for aid in self.activity_durations:
            durations = self.activity_durations[aid]
            rows.append({
                "activity_id": aid,
                "mean_duration": np.mean(durations),
                "std_duration": np.std(durations),
                "p50_duration": np.percentile(durations, 50),
                "p90_duration": np.percentile(durations, 90),
                "mean_start": np.mean(self.activity_starts[aid]),
                "mean_finish": np.mean(self.activity_finishes[aid]),
                "criticality_index": self.critical_path_counts.get(aid, 0) / self.n_iterations,
                "mean_risk_impact": np.mean(self.risk_impacts.get(aid, [0])),
            })
        return pd.DataFrame(rows)


def build_precedence_network(activities_df: pd.DataFrame) -> nx.DiGraph:
    """
    Build a directed acyclic graph (DAG) from activity precedence relationships.

    Parameters
    ----------
    activities_df : pd.DataFrame
        Must have columns: 'id', 'predecessors' (list of ints).

    Returns
    -------
    nx.DiGraph
        DAG with 'START' and 'END' sentinel nodes.
    """
    G = nx.DiGraph()

    # Add START and END sentinel nodes
    G.add_node("START", duration=0)
    G.add_node("END", duration=0)

    activity_ids = set(activities_df["id"].tolist())

    for _, row in activities_df.iterrows():
        aid = row["id"]
        G.add_node(aid)

        predecessors = row.get("predecessors", [])
        if isinstance(predecessors, str):
            predecessors = [int(x) for x in predecessors.split(",") if x.strip()]
        elif not isinstance(predecessors, list):
            predecessors = []

        # Filter to valid predecessor IDs
        valid_preds = [p for p in predecessors if p in activity_ids]

        if not valid_preds:
            G.add_edge("START", aid)
        else:
            for pred_id in valid_preds:
                G.add_edge(pred_id, aid)

    # Connect nodes with no successors to END
    for aid in activity_ids:
        if G.out_degree(aid) == 0:
            G.add_edge(aid, "END")

    # Verify DAG
    if not nx.is_directed_acyclic_graph(G):
        cycles = list(nx.simple_cycles(G))
        raise ValueError(f"Activity network contains cycles: {cycles[:3]}")

    return G


def _forward_pass(
    G: nx.DiGraph,
    durations: dict[int, float],
) -> tuple[dict, dict, float]:
    """
    Forward pass to compute Early Start (ES) and Early Finish (EF).

    Returns
    -------
    early_start, early_finish, project_duration
    """
    topo_order = list(nx.topological_sort(G))

    es = {"START": 0.0}
    ef = {"START": 0.0}

    for node in topo_order:
        if node == "START":
            continue

        # ES = max(EF of all predecessors)
        preds = list(G.predecessors(node))
        es[node] = max(ef[p] for p in preds) if preds else 0.0

        # EF = ES + duration
        dur = durations.get(node, 0.0)
        ef[node] = es[node] + dur

    return es, ef, ef["END"]


def _backward_pass(
    G: nx.DiGraph,
    durations: dict[int, float],
    project_duration: float,
) -> tuple[dict, dict]:
    """
    Backward pass to compute Late Start (LS) and Late Finish (LF).

    Returns
    -------
    late_start, late_finish
    """
    topo_order = list(reversed(list(nx.topological_sort(G))))

    lf = {"END": project_duration}
    ls = {"END": project_duration}

    for node in topo_order:
        if node == "END":
            continue

        # LF = min(LS of all successors)
        succs = list(G.successors(node))
        lf[node] = min(ls[s] for s in succs) if succs else project_duration

        # LS = LF - duration
        dur = durations.get(node, 0.0)
        ls[node] = lf[node] - dur

    return ls, lf


def find_critical_path(
    G: nx.DiGraph,
    durations: dict[int, float],
) -> tuple[list, float]:
    """
    Find the critical path using forward/backward pass.

    Returns
    -------
    critical_activities : list
        Activity IDs on the critical path.
    project_duration : float
        Total project duration.
    """
    es, ef, project_duration = _forward_pass(G, durations)
    ls, lf = _backward_pass(G, durations, project_duration)

    # Critical activities have zero total float (LS - ES == 0)
    critical = []
    for node in G.nodes():
        if node in ("START", "END"):
            continue
        total_float = ls.get(node, 0) - es.get(node, 0)
        if abs(total_float) < 1e-6:
            critical.append(node)

    return critical, project_duration


def run_simulation(
    activities_df: pd.DataFrame,
    risk_registry: Optional[RiskRegistry] = None,
    config: Optional[SimulationConfig] = None,
    ml_bias_factors: Optional[dict] = None,
    progress_callback=None,
) -> SimulationResults:
    """
    Run Monte Carlo simulation.

    Parameters
    ----------
    activities_df : pd.DataFrame
        Activity data with columns: id, name, dist_type, a, m, b, predecessors.
    risk_registry : RiskRegistry, optional
        Risk events to simulate.
    config : SimulationConfig, optional
        Simulation configuration.
    ml_bias_factors : dict, optional
        ML-predicted bias factors per activity_id (multiplied to sampled duration).
    progress_callback : callable, optional
        Function(iteration, n_total) called periodically for progress reporting.

    Returns
    -------
    SimulationResults
    """
    if config is None:
        config = SimulationConfig()

    start_time = time.time()

    # Build precedence network
    G = build_precedence_network(activities_df)
    activity_ids = [row["id"] for _, row in activities_df.iterrows()]
    n_activities = len(activity_ids)
    n_iter = config.n_iterations

    # Pre-create distribution objects (None for fixed-duration activities)
    distributions = {}
    fixed_durations = {}
    for _, row in activities_df.iterrows():
        aid = row["id"]
        a_val, m_val, b_val = float(row["a"]), float(row["m"]), float(row["b"])
        if a_val == b_val:
            # Fixed/completed activity - no sampling needed
            distributions[aid] = None
            fixed_durations[aid] = m_val
        else:
            dist = create_distribution(
                row.get("dist_type", "betapert"),
                a_val, m_val, b_val,
                lam=config.pert_lambda,
            )
            distributions[aid] = dist

    # Initialize result arrays
    total_durations = np.zeros(n_iter)
    activity_dur_samples = {aid: np.zeros(n_iter) for aid in activity_ids}
    activity_start_samples = {aid: np.zeros(n_iter) for aid in activity_ids}
    activity_finish_samples = {aid: np.zeros(n_iter) for aid in activity_ids}
    critical_path_counts = {aid: 0 for aid in activity_ids}
    risk_impact_samples = {aid: np.zeros(n_iter) for aid in activity_ids}

    # Main simulation loop
    rng = np.random.default_rng(config.random_seed)
    converged = False
    convergence_at = None

    for i in range(n_iter):
        # Sample durations for all activities
        durations = {}
        for aid in activity_ids:
            if aid in fixed_durations:
                sampled = fixed_durations[aid]
            else:
                sampled = float(distributions[aid].sample(1, rng=rng)[0])

                # Apply ML bias factor if available
                if ml_bias_factors and aid in ml_bias_factors:
                    sampled *= ml_bias_factors[aid]

            durations[aid] = max(sampled, 0.1)  # prevent zero/negative
            activity_dur_samples[aid][i] = durations[aid]

        # Simulate risk events
        if risk_registry:
            risk_delays = risk_registry.simulate_risks(activity_ids, rng)
            for aid in activity_ids:
                durations[aid] += risk_delays.get(aid, 0.0)
                risk_impact_samples[aid][i] = risk_delays.get(aid, 0.0)

        # Add sentinel durations
        durations["START"] = 0.0
        durations["END"] = 0.0

        # Forward pass to get project duration and starts/finishes
        es, ef, project_dur = _forward_pass(G, durations)
        total_durations[i] = project_dur

        for aid in activity_ids:
            activity_start_samples[aid][i] = es.get(aid, 0.0)
            activity_finish_samples[aid][i] = ef.get(aid, 0.0)

        # Find critical path for this iteration
        critical_activities, _ = find_critical_path(G, durations)
        for aid in critical_activities:
            if aid in critical_path_counts:
                critical_path_counts[aid] += 1

        # Convergence check every 500 iterations
        if config.convergence_check and i > 0 and i % 500 == 0 and not converged:
            if i >= 2000:
                p50_current = np.percentile(total_durations[:i+1], 50)
                p50_prev = np.percentile(total_durations[:i-499], 50)
                if p50_prev > 0:
                    rel_change = abs(p50_current - p50_prev) / p50_prev
                    if rel_change < config.convergence_tolerance:
                        converged = True
                        convergence_at = i

        # Progress callback
        if progress_callback and i % max(1, n_iter // 100) == 0:
            progress_callback(i, n_iter)

    elapsed = time.time() - start_time

    return SimulationResults(
        total_durations=total_durations,
        activity_durations=activity_dur_samples,
        activity_starts=activity_start_samples,
        activity_finishes=activity_finish_samples,
        critical_path_counts=critical_path_counts,
        risk_impacts=risk_impact_samples,
        n_iterations=n_iter,
        elapsed_seconds=elapsed,
        converged=converged,
        convergence_at=convergence_at,
    )


def run_simulation_parallel(
    activities_df: pd.DataFrame,
    risk_registry: Optional[RiskRegistry] = None,
    config: Optional[SimulationConfig] = None,
    ml_bias_factors: Optional[dict] = None,
    n_workers: Optional[int] = None,
) -> SimulationResults:
    """
    Run Monte Carlo simulation using multiple processes.

    Splits iterations across workers and combines results.
    """
    if config is None:
        config = SimulationConfig()

    if n_workers is None:
        n_workers = min(config.n_parallel, cpu_count())
    if n_workers <= 1:
        return run_simulation(activities_df, risk_registry, config, ml_bias_factors)

    iters_per_worker = config.n_iterations // n_workers
    remainder = config.n_iterations % n_workers

    configs = []
    for w in range(n_workers):
        worker_config = SimulationConfig(
            n_iterations=iters_per_worker + (1 if w < remainder else 0),
            random_seed=config.random_seed + w,
            confidence_levels=config.confidence_levels,
            convergence_check=False,
            pert_lambda=config.pert_lambda,
        )
        configs.append((activities_df, risk_registry, worker_config, ml_bias_factors))

    start_time = time.time()

    with Pool(n_workers) as pool:
        results = pool.starmap(_worker_simulate, configs)

    # Combine results
    combined = _merge_results(results)
    combined.elapsed_seconds = time.time() - start_time

    return combined


def _worker_simulate(activities_df, risk_registry, config, ml_bias_factors):
    """Worker function for parallel simulation."""
    return run_simulation(activities_df, risk_registry, config, ml_bias_factors)


def _merge_results(results: list[SimulationResults]) -> SimulationResults:
    """Merge results from multiple simulation runs."""
    total_durations = np.concatenate([r.total_durations for r in results])
    n_iterations = sum(r.n_iterations for r in results)

    # Merge activity-level arrays
    activity_ids = list(results[0].activity_durations.keys())
    activity_durations = {}
    activity_starts = {}
    activity_finishes = {}
    critical_path_counts = {}
    risk_impacts = {}

    for aid in activity_ids:
        activity_durations[aid] = np.concatenate(
            [r.activity_durations[aid] for r in results]
        )
        activity_starts[aid] = np.concatenate(
            [r.activity_starts[aid] for r in results]
        )
        activity_finishes[aid] = np.concatenate(
            [r.activity_finishes[aid] for r in results]
        )
        critical_path_counts[aid] = sum(
            r.critical_path_counts.get(aid, 0) for r in results
        )
        risk_impacts[aid] = np.concatenate(
            [r.risk_impacts.get(aid, np.zeros(r.n_iterations)) for r in results]
        )

    return SimulationResults(
        total_durations=total_durations,
        activity_durations=activity_durations,
        activity_starts=activity_starts,
        activity_finishes=activity_finishes,
        critical_path_counts=critical_path_counts,
        risk_impacts=risk_impacts,
        n_iterations=n_iterations,
        elapsed_seconds=0.0,
        converged=False,
        convergence_at=None,
    )


def deterministic_cpm(activities_df: pd.DataFrame) -> tuple[float, list, pd.DataFrame]:
    """
    Run a deterministic CPM analysis using most-likely durations.

    Returns
    -------
    project_duration : float
    critical_path : list of activity IDs
    schedule : pd.DataFrame with ES, EF, LS, LF, TF columns
    """
    G = build_precedence_network(activities_df)

    durations = {"START": 0.0, "END": 0.0}
    for _, row in activities_df.iterrows():
        durations[row["id"]] = float(row["m"])

    es, ef, project_duration = _forward_pass(G, durations)
    ls, lf = _backward_pass(G, durations, project_duration)
    critical_path, _ = find_critical_path(G, durations)

    rows = []
    for _, row in activities_df.iterrows():
        aid = row["id"]
        tf = ls.get(aid, 0) - es.get(aid, 0)
        rows.append({
            "id": aid,
            "name": row.get("name", ""),
            "duration": durations[aid],
            "ES": es.get(aid, 0),
            "EF": ef.get(aid, 0),
            "LS": ls.get(aid, 0),
            "LF": lf.get(aid, 0),
            "TF": tf,
            "on_critical_path": aid in critical_path,
        })

    schedule_df = pd.DataFrame(rows)
    return project_duration, critical_path, schedule_df

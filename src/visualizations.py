"""
Visualization Module.

Histogram, CDF (S-curve), tornado chart, 3D surface, critical path
network, and comparison charts for Monte Carlo simulation results.
"""

from typing import Optional

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for server use
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from .monte_carlo import SimulationResults


# --- Matplotlib (static) charts ---

def plot_histogram(
    results: SimulationResults,
    title: str = "Project Duration Distribution",
    bins: int = 50,
    percentile_lines: Optional[list] = None,
    actual_duration: Optional[float] = None,
    figsize: tuple = (10, 6),
) -> plt.Figure:
    """
    Histogram of total project duration with percentile markers.

    Parameters
    ----------
    results : SimulationResults
    title : str
    bins : int
    percentile_lines : list of floats (0-1), e.g. [0.50, 0.80, 0.90]
    actual_duration : float, optional
        If provided, draws a vertical line for actual duration.

    Returns
    -------
    matplotlib.Figure
    """
    if percentile_lines is None:
        percentile_lines = [0.50, 0.80, 0.90]

    fig, ax = plt.subplots(figsize=figsize)
    data = results.total_durations

    ax.hist(data, bins=bins, density=True, alpha=0.7, color="#2196F3",
            edgecolor="white", linewidth=0.5)

    colors = {"0.5": "#FF9800", "0.8": "#F44336", "0.9": "#9C27B0"}
    for p in percentile_lines:
        val = np.percentile(data, p * 100)
        color = colors.get(str(p), "#333333")
        label = f"P{int(p*100)} = {val:.1f} days"
        ax.axvline(val, color=color, linestyle="--", linewidth=2, label=label)

    if actual_duration is not None:
        ax.axvline(actual_duration, color="#4CAF50", linestyle="-", linewidth=2.5,
                   label=f"Actual = {actual_duration:.0f} days")

    ax.set_xlabel("Project Duration (days)", fontsize=12)
    ax.set_ylabel("Probability Density", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend(fontsize=10, loc="upper right")

    stats_text = (
        f"Mean: {np.mean(data):.1f}\n"
        f"Std Dev: {np.std(data):.1f}\n"
        f"N = {results.n_iterations:,}"
    )
    ax.text(0.02, 0.95, stats_text, transform=ax.transAxes,
            fontsize=9, verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    fig.tight_layout()
    return fig


def plot_cdf(
    results: SimulationResults,
    title: str = "Cumulative Distribution (S-Curve)",
    percentile_lines: Optional[list] = None,
    actual_duration: Optional[float] = None,
    figsize: tuple = (10, 6),
) -> plt.Figure:
    """
    CDF (S-curve) showing probability of completing before a given date.
    """
    if percentile_lines is None:
        percentile_lines = [0.50, 0.80, 0.90]

    fig, ax = plt.subplots(figsize=figsize)
    data = np.sort(results.total_durations)
    cdf = np.arange(1, len(data) + 1) / len(data)

    ax.plot(data, cdf, color="#2196F3", linewidth=2)
    ax.fill_between(data, cdf, alpha=0.1, color="#2196F3")

    colors = {"0.5": "#FF9800", "0.8": "#F44336", "0.9": "#9C27B0"}
    for p in percentile_lines:
        val = np.percentile(results.total_durations, p * 100)
        color = colors.get(str(p), "#333333")
        ax.axhline(p, color=color, linestyle=":", alpha=0.5)
        ax.axvline(val, color=color, linestyle="--", linewidth=1.5,
                   label=f"P{int(p*100)} = {val:.1f} days")
        ax.plot(val, p, "o", color=color, markersize=8)

    if actual_duration is not None:
        prob = results.probability_of_completion(actual_duration)
        ax.axvline(actual_duration, color="#4CAF50", linestyle="-", linewidth=2,
                   label=f"Actual = {actual_duration:.0f} (P{prob*100:.0f})")

    ax.set_xlabel("Project Duration (days)", fontsize=12)
    ax.set_ylabel("Cumulative Probability", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_ylim(0, 1.02)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    ax.legend(fontsize=10, loc="lower right")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    return fig


def plot_tornado(
    tornado_df: pd.DataFrame,
    title: str = "Sensitivity Analysis - Tornado Diagram",
    top_n: int = 15,
    figsize: tuple = (10, 8),
) -> plt.Figure:
    """
    Tornado diagram showing Spearman rank correlations.

    Parameters
    ----------
    tornado_df : pd.DataFrame
        From sensitivity.tornado_analysis(). Must have 'name', 'correlation'.
    top_n : int
        Show top N activities.
    """
    df = tornado_df.head(top_n).sort_values("abs_correlation", ascending=True)

    fig, ax = plt.subplots(figsize=figsize)
    colors = ["#F44336" if c > 0 else "#2196F3" for c in df["correlation"]]
    bars = ax.barh(df["name"], df["correlation"], color=colors, height=0.6)

    ax.set_xlabel("Spearman Rank Correlation with Project Duration", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.axvline(0, color="black", linewidth=0.8)

    # Add value labels
    for bar, val in zip(bars, df["correlation"]):
        x = bar.get_width()
        ax.text(x + 0.01 * np.sign(x), bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", va="center", fontsize=9)

    fig.tight_layout()
    return fig


def plot_criticality(
    criticality_df: pd.DataFrame,
    title: str = "Criticality Index",
    figsize: tuple = (10, 6),
) -> plt.Figure:
    """Bar chart of criticality index per activity."""
    df = criticality_df.sort_values("criticality_index", ascending=True)

    fig, ax = plt.subplots(figsize=figsize)
    colors = plt.cm.RdYlGn_r(df["criticality_index"])
    ax.barh(df["name"], df["criticality_index"], color=colors, height=0.6)

    ax.set_xlabel("Criticality Index (fraction on critical path)", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlim(0, 1.05)
    ax.xaxis.set_major_formatter(mticker.PercentFormatter(1.0))

    fig.tight_layout()
    return fig


def plot_comparison(
    comparison_data: dict[str, float],
    title: str = "Duration Estimate Comparison",
    figsize: tuple = (10, 6),
) -> plt.Figure:
    """
    Bar chart comparing different estimation methods.

    Parameters
    ----------
    comparison_data : dict
        E.g. {"PERT/CPM": 240, "P50 Monte Carlo": 265, "P80": 290, "Actual": 278}
    """
    fig, ax = plt.subplots(figsize=figsize)
    labels = list(comparison_data.keys())
    values = list(comparison_data.values())

    colors = ["#2196F3", "#FF9800", "#F44336", "#4CAF50", "#9C27B0"][:len(labels)]
    bars = ax.bar(labels, values, color=colors, width=0.5)

    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                f"{val:.0f}", ha="center", fontsize=11, fontweight="bold")

    ax.set_ylabel("Duration (days)", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_ylim(0, max(values) * 1.15)

    fig.tight_layout()
    return fig


def plot_convergence(
    results: SimulationResults,
    percentiles: Optional[list] = None,
    title: str = "Monte Carlo Convergence",
    figsize: tuple = (10, 6),
) -> plt.Figure:
    """
    Plot how percentile estimates stabilize as iterations increase.
    """
    if percentiles is None:
        percentiles = [0.50, 0.80, 0.90]

    fig, ax = plt.subplots(figsize=figsize)
    data = results.total_durations
    checkpoints = np.arange(100, len(data) + 1, max(1, len(data) // 200))

    colors = {"0.5": "#FF9800", "0.8": "#F44336", "0.9": "#9C27B0"}
    for p in percentiles:
        vals = [np.percentile(data[:n], p * 100) for n in checkpoints]
        color = colors.get(str(p), "#333333")
        ax.plot(checkpoints, vals, color=color, linewidth=1.5,
                label=f"P{int(p*100)}")

    if results.convergence_at:
        ax.axvline(results.convergence_at, color="green", linestyle=":",
                   label=f"Converged at {results.convergence_at}")

    ax.set_xlabel("Number of Iterations", fontsize=12)
    ax.set_ylabel("Duration (days)", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    return fig


# --- Plotly (interactive) charts ---

def plotly_histogram(
    results: SimulationResults,
    title: str = "Project Duration Distribution",
    percentile_lines: Optional[list] = None,
) -> go.Figure:
    """Interactive histogram using Plotly."""
    if percentile_lines is None:
        percentile_lines = [0.50, 0.80, 0.90]

    data = results.total_durations
    fig = go.Figure()

    fig.add_trace(go.Histogram(
        x=data, nbinsx=50, name="Duration",
        marker_color="rgba(33, 150, 243, 0.7)",
        histnorm="probability density",
    ))

    colors = {0.50: "orange", 0.80: "red", 0.90: "purple"}
    for p in percentile_lines:
        val = np.percentile(data, p * 100)
        fig.add_vline(
            x=val, line_dash="dash",
            line_color=colors.get(p, "gray"),
            annotation_text=f"P{int(p*100)}={val:.1f}",
        )

    fig.update_layout(
        title=title,
        xaxis_title="Project Duration (days)",
        yaxis_title="Probability Density",
        showlegend=True,
    )
    return fig


def plotly_cdf(
    results: SimulationResults,
    title: str = "Cumulative Distribution (S-Curve)",
) -> go.Figure:
    """Interactive CDF using Plotly."""
    data = np.sort(results.total_durations)
    cdf = np.arange(1, len(data) + 1) / len(data)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=data, y=cdf, mode="lines", name="CDF",
        line=dict(color="royalblue", width=2),
        fill="tozeroy", fillcolor="rgba(65, 105, 225, 0.1)",
    ))

    for p, color in [(0.50, "orange"), (0.80, "red"), (0.90, "purple")]:
        val = np.percentile(results.total_durations, p * 100)
        fig.add_hline(y=p, line_dash="dot", line_color=color, opacity=0.5)
        fig.add_vline(x=val, line_dash="dash", line_color=color,
                      annotation_text=f"P{int(p*100)}={val:.1f}")

    fig.update_layout(
        title=title,
        xaxis_title="Project Duration (days)",
        yaxis_title="Cumulative Probability",
        yaxis=dict(tickformat=".0%"),
    )
    return fig


def plotly_tornado(
    tornado_df: pd.DataFrame,
    title: str = "Sensitivity - Tornado Diagram",
    top_n: int = 15,
) -> go.Figure:
    """Interactive tornado diagram using Plotly."""
    df = tornado_df.head(top_n).sort_values("abs_correlation", ascending=True)

    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=df["name"],
        x=df["correlation"],
        orientation="h",
        marker_color=["red" if c > 0 else "steelblue" for c in df["correlation"]],
        text=[f"{c:.3f}" for c in df["correlation"]],
        textposition="outside",
    ))

    fig.update_layout(
        title=title,
        xaxis_title="Spearman Rank Correlation",
        yaxis_title="",
        showlegend=False,
    )
    return fig


def plotly_3d_surface(
    results: SimulationResults,
    buffer_range: Optional[np.ndarray] = None,
    title: str = "Buffer vs Completion Probability vs Cost Impact",
) -> go.Figure:
    """
    3D surface plot: buffer size vs. completion probability vs. cost impact.

    Parameters
    ----------
    results : SimulationResults
    buffer_range : np.ndarray, optional
        Buffer sizes to evaluate (in days). Defaults to 0-60 days.
    """
    if buffer_range is None:
        p50 = np.percentile(results.total_durations, 50)
        buffer_range = np.linspace(0, p50 * 0.3, 30)

    # Cost impact factor (normalized: 0 = free, 1 = full daily cost)
    cost_factors = np.linspace(0, 1, 20)

    # Compute completion probability for each combination
    p50 = np.percentile(results.total_durations, 50)
    Z = np.zeros((len(buffer_range), len(cost_factors)))

    for i, buffer in enumerate(buffer_range):
        target = p50 + buffer
        prob = results.probability_of_completion(target)
        for j, cost in enumerate(cost_factors):
            # Simple model: probability * (1 - cost_factor * buffer/p50)
            Z[i, j] = prob * (1 - cost * buffer / p50 * 0.5)

    fig = go.Figure(data=[go.Surface(
        x=cost_factors,
        y=buffer_range,
        z=Z,
        colorscale="Viridis",
        colorbar_title="Adjusted Score",
    )])

    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title="Cost Impact Factor",
            yaxis_title="Buffer (days)",
            zaxis_title="Completion Probability Score",
        ),
    )
    return fig


def plotly_network(
    activities_df: pd.DataFrame,
    criticality_index: Optional[dict] = None,
    title: str = "Activity Precedence Network",
) -> go.Figure:
    """
    Interactive network diagram of activity precedences.

    Parameters
    ----------
    activities_df : pd.DataFrame
    criticality_index : dict, optional
        activity_id -> criticality_index (0-1) for color coding.
    """
    import networkx as nx
    from .monte_carlo import build_precedence_network

    G = build_precedence_network(activities_df)

    # Layout
    pos = nx.spring_layout(G, k=2, iterations=50, seed=42)

    # Create edges
    edge_x, edge_y = [], []
    for u, v in G.edges():
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y, mode="lines",
        line=dict(width=1, color="#888"),
        hoverinfo="none",
    )

    # Create nodes
    node_x, node_y, node_text, node_color = [], [], [], []
    name_lookup = dict(zip(activities_df["id"], activities_df["name"]))

    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)

        if node in ("START", "END"):
            node_text.append(str(node))
            node_color.append(0.0)
        else:
            name = name_lookup.get(node, str(node))
            ci = criticality_index.get(node, 0) if criticality_index else 0
            node_text.append(f"{node}: {name}<br>CI: {ci:.2%}")
            node_color.append(ci)

    node_trace = go.Scatter(
        x=node_x, y=node_y, mode="markers+text",
        marker=dict(
            size=20,
            color=node_color,
            colorscale="RdYlGn_r",
            cmin=0, cmax=1,
            colorbar=dict(title="Criticality"),
            line=dict(width=1, color="black"),
        ),
        text=[str(n) for n in G.nodes()],
        textposition="top center",
        hovertext=node_text,
        hoverinfo="text",
    )

    fig = go.Figure(data=[edge_trace, node_trace])
    fig.update_layout(
        title=title,
        showlegend=False,
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
    )
    return fig


def plotly_scenario_comparison(
    scenario_df: pd.DataFrame,
    title: str = "Scenario Comparison",
) -> go.Figure:
    """
    Grouped bar chart comparing scenarios.

    Parameters
    ----------
    scenario_df : pd.DataFrame
        From sensitivity.scenario_analysis(). Columns: scenario, P50, P80, P90.
    """
    fig = go.Figure()
    for metric, color in [("P50", "orange"), ("P80", "red"), ("P90", "purple")]:
        fig.add_trace(go.Bar(
            name=metric,
            x=scenario_df["scenario"],
            y=scenario_df[metric],
            marker_color=color,
            text=[f"{v:.0f}" for v in scenario_df[metric]],
            textposition="outside",
        ))

    fig.update_layout(
        title=title,
        xaxis_title="Scenario",
        yaxis_title="Duration (days)",
        barmode="group",
    )
    return fig


def save_all_figures(
    figures: dict[str, plt.Figure],
    output_dir: str,
    dpi: int = 150,
    formats: tuple = ("png",),
) -> list[str]:
    """Save multiple matplotlib figures to files."""
    from pathlib import Path
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    saved = []

    for name, fig in figures.items():
        for fmt in formats:
            path = output_dir / f"{name}.{fmt}"
            fig.savefig(str(path), dpi=dpi, bbox_inches="tight")
            saved.append(str(path))
        plt.close(fig)

    return saved

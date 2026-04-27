"""Visualization functions for DC-TCO framework.

All matplotlib/seaborn code is isolated here.  These functions are
designed for use in notebooks; the core package never imports this module.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import matplotlib
import numpy as np
import matplotlib.pyplot as plt

from .hardware import GpuSpec
from .simulation import SimulationResult, quarter_label
from .scenarios import MonteCarloResult, SweepResult


def _set_quarter_ticks(
    ax,
    quarters: List[int],
    base_year: int,
    qpy: int = 4
) -> None:
    """Replace raw quarter indices with year-quarter labels on the x-axis.

    Shows one label per year (at Q1) to avoid clutter, with minor ticks
    at each quarter.
    """
    labels = [quarter_label(q, base_year, qpy) for q in quarters]
    # Show every qpy-th label (once per year) to avoid crowding
    step = max(1, qpy)
    q_list = quarters if hasattr(quarters, '__getitem__') else list(quarters)
    tick_positions = q_list[::step]
    tick_labels = [f"{labels[i]} (+{q_list[i] // qpy})" for i in range(0, len(q_list), step)]
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, rotation=45, ha="right", fontsize=8)


# ============================================================================
# GPU roadmap trends
# ============================================================================

def plot_gpu_trends(
    roadmap: List[GpuSpec],
    metrics: Optional[List[str]] = None,
    figsize: Tuple[int, int] = (14, 8),
):
    """Plot GPU metric trends over time (known + projected).

    Parameters
    ----------
    roadmap : list of GpuSpec
        GPU roadmap (from :func:`build_gpu_roadmap`).
    metrics : list of str, optional
        Subset of ``["tdp", "tflops", "mem_bw", "mem_cap", "cost"]``.
    """
    if metrics is None:
        metrics = ["tdp", "tflops", "mem_bw", "mem_cap", "cost"]

    labels = {
        "tdp": "TDP (W)", "tflops": "TFLOPS (FP16)",
        "mem_bw": "Mem BW (GB/s)", "mem_cap": "Mem Cap (GB)",
        "cost": "Cost (USD)",
    }

    n = len(metrics)
    fig, axes = plt.subplots(1, n, figsize=figsize, squeeze=False)

    for idx, metric in enumerate(metrics):
        ax = axes[0, idx]
        years = [g.release_year for g in roadmap]
        vals = [getattr(g, metric) for g in roadmap]
        names = [g.code_name for g in roadmap]

        known_mask = [not n.startswith("Future") for n in names]
        future_mask = [n.startswith("Future") for n in names]

        ax.plot(
            [y for y, m in zip(years, known_mask) if m],
            [v for v, m in zip(vals, known_mask) if m],
            "o-", label="Known", color="tab:blue",
        )
        if any(future_mask):
            ax.plot(
                [y for y, m in zip(years, future_mask) if m],
                [v for v, m in zip(vals, future_mask) if m],
                "s--", label="Projected", color="tab:orange",
            )

        # Annotate known GPUs
        for y, v, n_label, m in zip(years, vals, names, known_mask):
            if m:
                ax.annotate(n_label, (y, v), fontsize=7, ha="center",
                            va="bottom", rotation=30)

        ax.set_xlabel("Release Year (offset)")
        ax.set_ylabel(labels.get(metric, metric))
        ax.set_title(labels.get(metric, metric))
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    return fig


# ============================================================================
# Server timeline (stacked area)
# ============================================================================

def plot_server_timeline(
    result: SimulationResult,
    figsize: Tuple[int, int] = (14, 6),
):
    """Stacked bar chart of servers by GPU generation over time."""
    quarters = [s.quarter for s in result.quarterly_states]
    all_gens = sorted({
        gen for s in result.quarterly_states
        for gen in s.gen_server_counts
    })

    gen_data: Dict[str, List[int]] = {g: [] for g in all_gens}
    for s in result.quarterly_states:
        for g in all_gens:
            gen_data[g].append(s.gen_server_counts.get(g, 0))

    fig, ax = plt.subplots(figsize=figsize)
    bottom = np.zeros(len(quarters))

    colors = matplotlib.colormaps["tab20"](np.linspace(0, 1, max(len(all_gens), 1)))
    bar_width = 0.8
    for idx, gen in enumerate(all_gens):
        vals = np.array(gen_data[gen])
        ax.bar(quarters, vals, bottom=bottom, width=bar_width,
               label=gen, alpha=0.8, color=colors[idx % len(colors)])
        bottom += vals

    ax.set_xlabel("Quarter")
    ax.set_ylabel("Number of Servers")
    ax.set_title("Server Fleet Composition Over Time")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(True, alpha=0.3, axis="y")
    _set_quarter_ticks(ax, quarters, result.config.simulation.base_year,
                       result.config.simulation.quarters_per_year)
    fig.tight_layout()
    return fig


# ============================================================================
# TCO breakdown (stacked bar per year)
# ============================================================================

def plot_tco_breakdown(
    result: SimulationResult,
    figsize: Tuple[int, int] = (14, 6),
    scale: float = 1e9,
    unit: str = "Billion $",
):
    """Stacked bar chart of annual TCO by cost category."""
    base_year = result.config.simulation.base_year
    years = sorted(result.tco_by_year.keys())
    cal_years = [base_year + y for y in years]
    categories = [
        ("Servers", "capex_server"),
        ("Racks", "capex_rack"),
        ("Power Infra", "capex_power_provisioning"),
        ("Cooling Infra", "capex_cooling_provisioning"),
        ("Net CapEx", "capex_network"),
        ("Energy", "opex_energy"),
        ("Cooling", "opex_cooling"),
        ("Maintenance", "opex_maintenance"),
        ("Net OpEx", "opex_network"),
    ]

    fig, ax = plt.subplots(figsize=figsize)
    bottom = np.zeros(len(years))
    capex_colors = matplotlib.colormaps["Blues"](np.linspace(0.3, 0.8, 5))
    opex_colors = matplotlib.colormaps["Oranges"](np.linspace(0.3, 0.8, 4))
    colors = list(capex_colors) + list(opex_colors)

    for idx, (label, attr) in enumerate(categories):
        vals = np.array([getattr(result.tco_by_year[y], attr) for y in years]) / scale
        if np.any(vals > 0):
            ax.bar(cal_years, vals, bottom=bottom[:len(years)],
                   label=label, color=colors[idx % len(colors)])
            bottom[:len(years)] += vals

    ax.set_xlabel("Year")
    ax.set_ylabel(f"Cost ({unit})")
    ax.set_title("Annual TCO Breakdown")
    ax.legend(loc="upper left", fontsize=8, ncol=2)
    tick_labels = [f"{cy} (+{y})" for cy, y in zip(cal_years, years)]
    ax.set_xticks(cal_years)
    ax.set_xticklabels(tick_labels, rotation=45, ha="right", fontsize=8)
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    return fig


# ============================================================================
# Stranded power
# ============================================================================

def plot_stranded_power(
    result: SimulationResult,
    figsize: Tuple[int, int] = (14, 4),
):
    """Line plot of stranded power over time."""
    quarters = [s.quarter for s in result.quarterly_states]
    stranded = [s.stranded_power_kw for s in result.quarterly_states]

    fig, ax = plt.subplots(figsize=figsize)
    ax.fill_between(quarters, stranded, alpha=0.4, color="tab:red")
    ax.plot(quarters, stranded, color="tab:red", linewidth=1.5)
    ax.set_xlabel("Quarter")
    ax.set_ylabel("Stranded Power (kW)")
    ax.set_title("Stranded Power Over Time")
    ax.grid(True, alpha=0.3)
    _set_quarter_ticks(ax, quarters, result.config.simulation.base_year,
                       result.config.simulation.quarters_per_year)
    fig.tight_layout()
    return fig


# ============================================================================
# Policy comparison (sweep results)
# ============================================================================

def plot_policy_comparison(
    sweep_results: List[SweepResult],
    normalize_to: Optional[str] = "baseline",
    figsize: Tuple[int, int] = (16, 6),
    variant: str = "regular",
):
    """Bar chart comparing TCO across policies from a sweep.

    Parameters
    ----------
    normalize_to : str, optional
        Normalize all TCO values to this policy's TCO (shows % difference).
    variant : str
        Filter to "regular" or "disaggregated" results.
    """
    filtered = [r for r in sweep_results if r.variant == variant]
    if not filtered:
        return None

    if normalize_to:
        baseline_tco = next(
            (r.total_tco for r in filtered if r.policy_name == normalize_to),
            filtered[0].total_tco,
        )
        values = [(r.total_tco / baseline_tco - 1) * 100 for r in filtered]
        ylabel = f"TCO Change vs {normalize_to} (%)"
    else:
        values = [r.total_tco / 1e9 for r in filtered]
        ylabel = "TCO (Billion $)"

    names = [r.policy_name for r in filtered]

    fig, ax = plt.subplots(figsize=figsize)
    colors = ["tab:green" if v < 0 else "tab:red" for v in values]
    ax.bar(range(len(names)), values, color=colors, alpha=0.8)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=7)
    ax.set_ylabel(ylabel)
    ax.set_title(f"Policy TCO Comparison ({variant})")
    ax.axhline(0, color="black", linewidth=0.5)
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    return fig


# ============================================================================
# Monte Carlo distribution
# ============================================================================

def plot_monte_carlo_distribution(
    mc_result: MonteCarloResult,
    figsize: Tuple[int, int] = (10, 6),
    bins: int = 50,
    outlier_pct: float = 1.0,
):
    """Histogram of TCO values from Monte Carlo simulation.

    Parameters
    ----------
    outlier_pct : float
        Percentage to trim from each tail (default 1.0 = exclude
        bottom 1% and top 1%).
    """
    fig, ax = plt.subplots(figsize=figsize)

    values = mc_result.tco_values / 1e9
    lo = np.percentile(values, outlier_pct)
    hi = np.percentile(values, 100 - outlier_pct)
    filtered = values[(values >= lo) & (values <= hi)]
    n_excluded = len(values) - len(filtered)

    ax.hist(filtered, bins=bins, alpha=0.7,
            color="tab:blue", edgecolor="black", linewidth=0.5)

    # Mark statistics
    ax.axvline(mc_result.mean / 1e9, color="red", linestyle="--",
               label=f"Mean: ${mc_result.mean/1e9:.2f}B")
    ax.axvline(mc_result.p5 / 1e9, color="orange", linestyle=":",
               label=f"P5: ${mc_result.p5/1e9:.2f}B")
    ax.axvline(mc_result.p95 / 1e9, color="orange", linestyle=":",
               label=f"P95: ${mc_result.p95/1e9:.2f}B")

    ax.set_xlabel("Total Lifecycle TCO (Billion $)")
    ax.set_ylabel("Frequency")
    ax.set_title(f"Monte Carlo TCO Distribution ({mc_result.num_trials} trials, "
                 f"policy={mc_result.policy_name}"
                 f"{f', {n_excluded} outliers excluded' if n_excluded else ''})")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def plot_demand_curve(
    result: SimulationResult,
    figsize: Tuple[int, int] = (14, 4),
):
    """Plot demand and capacity over time."""
    quarters = [s.quarter for s in result.quarterly_states]
    demand = [s.demand for s in result.quarterly_states]
    capacity = [s.total_capacity for s in result.quarterly_states]

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(quarters, np.array(demand) / 1e3, label="Demand (K req/s)",
            color="tab:blue", linewidth=2)
    ax.plot(quarters, np.array(capacity) / 1e3, label="Capacity (K tok/s)",
            color="tab:green", linewidth=2, linestyle="--")
    shortfall = (np.array(demand) > np.array(capacity)).tolist()
    ax.fill_between(quarters, np.array(demand) / 1e3, np.array(capacity) / 1e3,
                    where=shortfall,
                    alpha=0.3, color="tab:red", label="Shortfall")
    ax.set_xlabel("Quarter")
    ax.set_ylabel("Throughput (K/s)")
    ax.set_title("Demand vs. Capacity")
    ax.legend()
    ax.grid(True, alpha=0.3)
    _set_quarter_ticks(ax, quarters, result.config.simulation.base_year,
                       result.config.simulation.quarters_per_year)
    fig.tight_layout()
    return fig

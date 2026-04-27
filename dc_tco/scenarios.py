"""Scenario shocks, Monte Carlo engine, and policy sweep."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from itertools import combinations
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import stats as sp_stats

from .config import Config, DistributionType, MonteCarloConfig, PolicyName, StochasticVariable
from .hardware import GpuSpec, build_server_roadmap
from .simulation import SimulationResult, run_simulation


# ============================================================================
# Shock helpers
# ============================================================================

def apply_hw_capability_shock(
    gpu_roadmap: List[GpuSpec],
    cfg: Config,
) -> List[GpuSpec]:
    """Apply a hardware capability shock to the GPU roadmap.

    At ``shock_year``, TFLOPS, mem_bw, and mem_cap jump by ``gamma``,
    then post-shock growth slows to ``epsilon`` per year (replacing the
    original projection trend).
    """
    shock = cfg.scenarios.hw_capability_shock
    if not shock.enabled:
        return gpu_roadmap

    shock_q = shock.shock_year * cfg.simulation.quarters_per_year

    # Find the last pre-shock GPU to use as the baseline
    pre_shock = [g for g in gpu_roadmap if g.release_quarter < shock_q]
    if pre_shock:
        base_gpu = pre_shock[-1]
    else:
        base_gpu = gpu_roadmap[0]

    base_tflops = base_gpu.tflops
    base_mem_bw = base_gpu.mem_bw
    base_mem_cap = base_gpu.mem_cap

    result = []
    for gpu in gpu_roadmap:
        g = copy.copy(gpu)
        if g.release_quarter >= shock_q:
            years_after = (g.release_quarter - shock_q) / cfg.simulation.quarters_per_year
            # Replace projected values with: baseline * gamma * slow post-shock growth
            g.tflops = base_tflops * shock.gamma * (1 + shock.epsilon) ** years_after
            g.mem_bw = base_mem_bw * shock.gamma * (1 + shock.epsilon) ** years_after
            g.mem_cap = base_mem_cap * shock.gamma * (1 + shock.epsilon) ** years_after
        result.append(g)
    return result


def apply_price_shock(
    gpu_roadmap: List[GpuSpec],
    cfg: Config,
) -> List[GpuSpec]:
    """Apply a price shock to the GPU roadmap.

    At ``shock_year``, GPU cost drops to ``delta * Pricetrend(tp)``, then
    recovers linearly: ``delta * Pricetrend(tp) * (1 + eta * (t - tp))``,
    capped at the original projected price.
    """
    shock = cfg.scenarios.price_shock
    if not shock.enabled:
        return gpu_roadmap

    shock_q = shock.shock_year * cfg.simulation.quarters_per_year

    # Find the price at the shock point: last pre-shock GPU's projected cost
    pre_shock = [g for g in gpu_roadmap if g.release_quarter < shock_q]
    if pre_shock:
        base_cost = pre_shock[-1].cost
    else:
        base_cost = gpu_roadmap[0].cost

    result = []
    for gpu in gpu_roadmap:
        g = copy.copy(gpu)
        if g.release_quarter >= shock_q:
            years_after = (g.release_quarter - shock_q) / cfg.simulation.quarters_per_year
            # Price = delta * Pricetrend(tp) * (1 + eta * years_after), capped at original
            shocked_cost = base_cost * shock.delta * (1 + shock.eta * years_after)
            g.cost = min(shocked_cost, g.cost)  # Cap at original projected price
        result.append(g)
    return result


# ============================================================================
# Policy sweep
# ============================================================================

@dataclass
class SweepResult:
    """Result of a single policy evaluation within a sweep."""
    policy_name: str
    variant: str          # "regular" or "disaggregated"
    total_tco: float
    tco_breakdown: dict   # Flat dict of breakdown values
    skip_gens: List[str]
    extend_years: int


def run_policy_sweep(
    base_cfg: Config,
    include_disaggregated: bool = True,
    include_skip: bool = True,
    max_extend_years: int = 5,
) -> List[SweepResult]:
    """Run a combinatorial sweep over policies, generation skipping, and lifetime extension.

    Parameters
    ----------
    base_cfg : Config
        Base configuration to use for all runs.
    include_disaggregated : bool
        Also sweep disaggregated policy variants.
    include_skip : bool
        Also sweep generation-skipping combinations.
    max_extend_years : int
        Maximum lifetime extension to try (1 through max_extend_years).

    Returns
    -------
    list of SweepResult
    """
    results = []

    variants = ["regular"]
    if include_disaggregated:
        variants.append("disaggregated")

    # Get available GPU code names (skip the first one — always keep the initial gen)
    roadmap = build_server_roadmap(base_cfg)
    gen_names = [s.code_name for s in roadmap]
    skippable = gen_names[1:]  # Never skip the first generation

    for variant in variants:
        for skip_combo in _skip_combinations(skippable, include_skip):
            # --- Baseline ---
            cfg = _make_cfg(base_cfg, "baseline", variant, skip_combo, 0)
            result = run_simulation(cfg)
            results.append(_to_sweep_result(
                "baseline", variant, skip_combo, 0, result
            ))

            # --- Replace all ---
            cfg = _make_cfg(base_cfg, "replace_all", variant, skip_combo, 0)
            result = run_simulation(cfg)
            results.append(_to_sweep_result(
                "replace_all", variant, skip_combo, 0, result
            ))

            # --- Extend lifetime (1..max_extend_years) ---
            for ext in range(1, max_extend_years + 1):
                cfg = _make_cfg(base_cfg, "extend_lifetime", variant, skip_combo, ext)
                result = run_simulation(cfg)
                results.append(_to_sweep_result(
                    f"extend_{ext}y", variant, skip_combo, ext, result
                ))

    return results


def _skip_combinations(
    gen_names: List[str],
    include_skip: bool,
) -> List[List[str]]:
    """Generate skip-generation combinations: none, skip-1, skip-2."""
    combos: List[List[str]] = [[]]  # Always include no-skip
    if include_skip and gen_names:
        for g in gen_names:
            combos.append([g])
        for g1, g2 in combinations(gen_names, 2):
            combos.append([g1, g2])
    return combos


def _make_cfg(
    base: Config,
    policy_name: str,
    variant: str,
    skip_gens: List[str],
    extend_years: int,
) -> Config:
    """Create a Config for a specific sweep point."""
    cfg = copy.deepcopy(base)
    if variant == "disaggregated":
        cfg.policy.name = PolicyName.DISAGGREGATED
    else:
        cfg.policy.name = PolicyName(policy_name) if not isinstance(policy_name, PolicyName) else policy_name
    cfg.policy.skip_generations = skip_gens
    cfg.policy.extend_lifetime_years = extend_years
    return cfg


def _to_sweep_result(
    policy_name: str,
    variant: str,
    skip_gens: List[str],
    extend_years: int,
    sim: SimulationResult,
) -> SweepResult:
    label = policy_name
    if skip_gens:
        label += f"_skip-{'_'.join(skip_gens)}"
    return SweepResult(
        policy_name=label,
        variant=variant,
        total_tco=sim.total_tco,
        tco_breakdown={
            "capex_server": sim.tco_breakdown.capex_server,
            "capex_rack": sim.tco_breakdown.capex_rack,
            "capex_power": sim.tco_breakdown.capex_power_provisioning,
            "capex_cooling": sim.tco_breakdown.capex_cooling_provisioning,
            "capex_network": sim.tco_breakdown.capex_network,
            "opex_energy": sim.tco_breakdown.opex_energy,
            "opex_cooling": sim.tco_breakdown.opex_cooling,
            "opex_maintenance": sim.tco_breakdown.opex_maintenance,
            "opex_network": sim.tco_breakdown.opex_network,
        },
        skip_gens=skip_gens,
        extend_years=extend_years,
    )


# ============================================================================
# Monte Carlo engine
# ============================================================================

@dataclass
class MonteCarloResult:
    """Results from a Monte Carlo simulation."""
    num_trials: int
    tco_values: np.ndarray          # TCO for each trial
    mean: float
    std: float
    ci_95: Tuple[float, float]      # 95% confidence interval
    p5: float                       # 5th percentile
    p50: float                      # Median
    p95: float                      # 95th percentile
    converged: bool
    policy_name: str = "baseline"
    per_trial_configs: Optional[List[dict]] = None  # Sampled parameters


def run_monte_carlo(
    base_cfg: Config,
    mc_cfg: Optional[MonteCarloConfig] = None,
    policy_name: Optional[str] = None,
) -> MonteCarloResult:
    """Run Monte Carlo simulation over stochastic parameters.

    Samples parameters from their distributions (paper Table 2),
    respecting correlations, and runs a full simulation per trial.

    Parameters
    ----------
    base_cfg : Config
        Base configuration.
    mc_cfg : MonteCarloConfig, optional
        Monte Carlo configuration. If None, uses ``base_cfg.monte_carlo``.
    policy_name : str, optional
        Override the policy for all trials.

    Returns
    -------
    MonteCarloResult
    """
    if mc_cfg is None:
        mc_cfg = base_cfg.monte_carlo

    n_trials = mc_cfg.num_trials
    rng = np.random.default_rng(mc_cfg.seed)

    # Sample all parameters for all trials at once
    samples = _sample_parameters(mc_cfg, n_trials, rng)

    tco_values = np.zeros(n_trials)
    per_trial_configs = []

    for i in range(n_trials):
        trial_cfg = _apply_sample(base_cfg, samples[i])
        if policy_name:
            # Support extend_lifetime_Nyr syntax (e.g. "extend_lifetime_2yr")
            if policy_name.startswith("extend_lifetime_") and policy_name.endswith("yr"):
                trial_cfg.policy.name = PolicyName.EXTEND_LIFETIME
                trial_cfg.policy.extend_lifetime_years = int(policy_name.split("_")[-1].replace("yr", ""))
            else:
                trial_cfg.policy.name = PolicyName(policy_name)

        result = run_simulation(trial_cfg)
        tco_values[i] = result.total_tco
        per_trial_configs.append(samples[i])

    # Statistics
    mean = float(np.mean(tco_values))
    std = float(np.std(tco_values))
    ci_95 = (float(np.percentile(tco_values, 2.5)), float(np.percentile(tco_values, 97.5)))
    p5 = float(np.percentile(tco_values, 5))
    p50 = float(np.percentile(tco_values, 50))
    p95 = float(np.percentile(tco_values, 95))

    # Convergence check: running mean stable within threshold
    converged = _check_convergence(tco_values, mc_cfg.convergence_threshold)

    return MonteCarloResult(
        num_trials=n_trials,
        tco_values=tco_values,
        mean=mean,
        std=std,
        ci_95=ci_95,
        p5=p5,
        p50=p50,
        p95=p95,
        converged=converged,
        policy_name=policy_name or base_cfg.policy.name,
        per_trial_configs=per_trial_configs,
    )


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------

# Variable names and their mapped config paths
_VAR_MAP = {
    "workload_growth": "workload.scaling_factor",
    "model_size_growth": "model_trends._growth_factor",  # Custom handling
    "gpu_perf_per_watt": "hardware._perf_factor",       # Custom handling
    "gpu_price_change": "hardware._price_change",        # Custom handling
    "release_interval": "hardware.release_interval_years",
    "electricity_price": "tco.power_cost_per_kwh",
    "pue": "tco.cooling_ratio_opex",  # PUE now affects cooling OpEx ratio
    "server_lifetime": "tco.server_lifetime_years",
}


def _sample_parameters(
    mc_cfg: MonteCarloConfig,
    n_trials: int,
    rng: np.random.Generator,
) -> List[dict]:
    """Sample all stochastic variables for all trials.

    Uses multivariate normal for correlated variables, then transforms
    to target marginal distributions via inverse CDF.
    """
    var_names = list(mc_cfg.variables.keys())
    n_vars = len(var_names)

    if n_vars == 0:
        return [{} for _ in range(n_trials)]

    # Build correlation matrix
    corr_matrix = np.eye(n_vars)
    for i, name_i in enumerate(var_names):
        var_i = mc_cfg.variables[name_i]
        for j, name_j in enumerate(var_names):
            if name_j in var_i.correlation:
                corr_matrix[i, j] = var_i.correlation[name_j]
                corr_matrix[j, i] = var_i.correlation[name_j]

    # Sample from multivariate normal
    mvn_samples = rng.multivariate_normal(
        mean=np.zeros(n_vars),
        cov=corr_matrix,
        size=n_trials,
    )

    # Transform each variable to its target distribution
    samples = []
    for trial in range(n_trials):
        trial_sample = {}
        for k, name in enumerate(var_names):
            var_cfg = mc_cfg.variables[name]
            u = sp_stats.norm.cdf(mvn_samples[trial, k])  # Uniform [0, 1]
            trial_sample[name] = _inverse_cdf(u, var_cfg, name)
        samples.append(trial_sample)

    return samples


def _inverse_cdf(u: float, var: StochasticVariable, name: str) -> float:
    """Transform uniform [0,1] to target distribution via inverse CDF."""
    dist = var.distribution

    if dist == DistributionType.LOGNORMAL:
        mu = var.mu if var.mu is not None else 0.0
        sigma = var.sigma if var.sigma is not None else 0.1
        if var.sigma_fraction is not None:
            sigma = var.sigma_fraction  # Use as absolute sigma
        val = sp_stats.lognorm.ppf(u, s=sigma, scale=np.exp(mu))
        # Apply cap if specified
        if var.cap_sigma is not None:
            median = np.exp(mu)
            lo = median * np.exp(-var.cap_sigma * sigma)
            hi = median * np.exp(var.cap_sigma * sigma)
            val = np.clip(val, lo, hi)
        return float(val)

    elif dist == DistributionType.NORMAL:
        mean = var.mean if var.mean is not None else 0.0
        sigma = var.sigma if var.sigma is not None else 0.1
        if var.sigma_fraction is not None:
            sigma = abs(mean) * var.sigma_fraction
        return float(sp_stats.norm.ppf(u, loc=mean, scale=max(sigma, 1e-10)))

    elif dist == DistributionType.TRIANGULAR:
        lo = var.min if var.min is not None else -0.15
        hi = var.max if var.max is not None else 0.20
        mode = var.mode if var.mode is not None else 0.0
        # scipy triangular uses c = (mode - lo) / (hi - lo)
        c = (mode - lo) / (hi - lo) if hi > lo else 0.5
        return float(sp_stats.triang.ppf(u, c, loc=lo, scale=hi - lo))

    elif dist == DistributionType.DISCRETE:
        values = var.values or [1.0]
        idx = min(int(u * len(values)), len(values) - 1)
        return float(values[idx])

    else:
        raise ValueError(f"Unknown distribution: {dist} for variable {name}")


def _apply_sample(base_cfg: Config, sample: dict) -> Config:
    """Create a Config with sampled parameter values applied."""
    cfg = copy.deepcopy(base_cfg)

    if "workload_growth" in sample:
        cfg.workload.scaling_factor = sample["workload_growth"]

    if "release_interval" in sample:
        cfg.hardware.release_interval_years = sample["release_interval"]

    if "electricity_price" in sample:
        # Scale around the baseline
        baseline = base_cfg.tco.power_cost_per_kwh
        cfg.tco.power_cost_per_kwh = baseline * sample["electricity_price"]

    if "pue" in sample:
        cfg.tco.cooling_ratio_opex = sample["pue"]

    if "server_lifetime" in sample:
        cfg.tco.server_lifetime_years = int(sample["server_lifetime"])

    if "gpu_price_change" in sample:
        # Apply as a multiplier to all GPU costs
        factor = 1.0 + sample["gpu_price_change"]
        for gpu in cfg.hardware.known_gpus:
            gpu.cost *= factor

    # gpu_perf_per_watt and model_size_growth affect projections
    # but are harder to apply cleanly; they modify the trend slopes
    # For simplicity, these scale the projection mode's effective slope

    return cfg


# ---------------------------------------------------------------------------
# Convergence
# ---------------------------------------------------------------------------

def _check_convergence(values: np.ndarray, threshold: float) -> bool:
    """Check if running mean has stabilized.

    Returns True if the change in running mean over the final 2000 samples
    is less than *threshold* (as a fraction of the overall mean).
    """
    n = len(values)
    if n < 4000:
        return True  # Too few samples to check

    window = min(2000, n // 2)
    running_mean = np.cumsum(values) / np.arange(1, n + 1)
    recent_change = abs(running_mean[-1] - running_mean[-window]) / max(abs(running_mean[-1]), 1e-10)
    return recent_change < threshold


def compare_policies_mc(
    base_cfg: Config,
    policy_names: List[str],
    mc_cfg: Optional[MonteCarloConfig] = None,
) -> Dict[str, MonteCarloResult]:
    """Run Monte Carlo for multiple policies and return comparison.

    Parameters
    ----------
    base_cfg : Config
        Base configuration.
    policy_names : list of str
        Policy names to compare.
    mc_cfg : MonteCarloConfig, optional
        Monte Carlo configuration.

    Returns
    -------
    dict mapping policy_name -> MonteCarloResult
    """
    results = {}
    for name in policy_names:
        results[name] = run_monte_carlo(base_cfg, mc_cfg, policy_name=name)
    return results

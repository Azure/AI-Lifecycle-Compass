"""Configuration loading and management for DC-TCO framework."""

from __future__ import annotations

import yaml
import dataclasses

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional


@dataclass
class SimulationConfig:
    base_year: int = 2016           # Calendar year of simulation start (aligned with first known GPU)
    total_years: int = 15
    quarters_per_year: int = 4

    @property
    def num_quarters(self) -> int:
        return self.total_years * self.quarters_per_year


@dataclass
class GpuEntry:
    code_name: str = ""
    quarter: str = ""
    tdp: float = 0.0
    tflops: float = 0.0
    mem_bw: float = 0.0
    mem_cap: float = 0.0
    net_bw: float = 0.0    # NVLink bisection bandwidth (GB/s per GPU)
    cost: float = 0.0


# ---------------------------------------------------------------------------
# Enums for constrained string fields
# ---------------------------------------------------------------------------

class PerformanceModelType(str, Enum):
    """Throughput prediction model type."""
    THEORETICAL = "theoretical"
    SIMPLIFIED = "simplified"
    CSV = "csv"


class ProjectionMode(str, Enum):
    """Hardware metric projection mode."""
    FLAT = "flat"
    LINEAR = "linear"
    EXPONENTIAL = "exponential"


class ScalingMode(str, Enum):
    """Workload/model-trend scaling mode."""
    FLAT = "flat"
    LINEAR = "linear"
    EXPONENTIAL = "exponential"


class NetworkType(str, Enum):
    """Network interconnect type."""
    ETHERNET = "ethernet"
    INFINIBAND = "infiniband"
    NVLINK = "nvlink"


class PolicyName(str, Enum):
    """Server refresh policy."""
    BASELINE = "baseline"
    REPLACE_ALL = "replace_all"
    EXTEND_LIFETIME = "extend_lifetime"
    SKIP_GENERATION = "skip_generation"
    DISAGGREGATED = "disaggregated"
    DISAGGREGATED_REPLACE = "disaggregated_replace"


class DistributionType(str, Enum):
    """Monte Carlo stochastic distribution type."""
    NORMAL = "normal"
    LOGNORMAL = "lognormal"
    TRIANGULAR = "triangular"
    DISCRETE = "discrete"


class GoodputMode(str, Enum):
    """Rental goodput recovery strategy."""
    CHECKPOINT_COLD = "checkpoint_cold"
    CHECKPOINT_HOT = "checkpoint_hot"
    FAULT_TOLERANT = "fault_tolerant"


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class HardwareProjection:
    tdp: ProjectionMode = ProjectionMode.LINEAR
    tflops: ProjectionMode = ProjectionMode.LINEAR
    mem_bw: ProjectionMode = ProjectionMode.LINEAR
    mem_cap: ProjectionMode = ProjectionMode.LINEAR
    net_bw: ProjectionMode = ProjectionMode.LINEAR
    cost: ProjectionMode = ProjectionMode.LINEAR


@dataclass
class HardwareConfig:
    gpus_per_server: int = 8
    power_overhead_factor: float = 0.5  # 50% power GPU, rest overhead
    release_interval_years: float = 1.0
    projection: HardwareProjection = field(default_factory=HardwareProjection)
    known_gpus: List[GpuEntry] = field(default_factory=list)


@dataclass
class ModelTrendsConfig:
    trend: ProjectionMode = ProjectionMode.LINEAR
    notable_models_csv: Optional[str] = None
    # Column mapping — set these when using a custom CSV with different headers.
    # When null/empty, auto-detection tries common column names.
    csv_col_params: Optional[str] = None    # Column containing parameter counts
    csv_col_date: Optional[str] = None      # Column containing publication date or year
    csv_col_name: Optional[str] = None      # Column containing model name
    csv_min_year: int = 2015                 # Only include models from this year onward
    distilled_model_size_fraction: float = 0.1


@dataclass
class PerformanceConfig:
    model_type: PerformanceModelType = PerformanceModelType.THEORETICAL
    max_ttft_s: float = 0.4
    max_tbt_s: float = 0.1
    compute_efficiency: float = 0.2
    memory_efficiency: float = 0.6
    network_efficiency: float = 1.0
    throughput_csv: Optional[str] = None  # Path to CSV for model_type=csv


@dataclass
class WorkloadConfig:
    scaling: ScalingMode = ScalingMode.EXPONENTIAL
    scaling_factor: float = 1.05
    initial_demand: float = 100_000
    initial_migration_fraction: float = 0.1
    offload_fraction: float = 0.2
    prefill_fraction: float = 0.4


@dataclass
class TcoConfig:
    server_lifetime_years: int = 5
    amortization_years: int = 5
    utilization: float = 0.75
    power_cost_per_kwh: float = 0.04
    cooling_ratio_opex: float = 0.2
    cooling_ratio_capex: float = 0.3
    facility_cost: float = 1_000_000
    facility_power_capacity_w: float = 8_000_000
    rack_power_w: float = 50_000
    rack_cost: float = 2_000
    row_power_limit_w: float = 300_000
    maintenance_cost_per_server_per_year: float = 5_200
    power_provisioning_cost_per_watt: float = 7.0
    cooling_provisioning_cost_per_watt: float = 2.5
    building_lifetime_years: int = 15
    hours_per_year: float = 365 * 24

    @property
    def server_lifetime_quarters(self) -> int:
        return self.server_lifetime_years * 4


@dataclass
class NetworkTypeConfig:
    capex_per_server: float = 0.0
    opex_per_server_per_year: float = 0.0
    bandwidth_gbps: float = 0.0
    latency_us: float = 0.0
    performance_multiplier: float = 1.0


@dataclass
class NetworkingConfig:
    type: NetworkType = NetworkType.ETHERNET
    configs: Dict[str, NetworkTypeConfig] = field(default_factory=dict)

    @property
    def active(self) -> NetworkTypeConfig:
        return self.configs.get(self.type, NetworkTypeConfig())


@dataclass
class PolicyConfig:
    name: PolicyName = PolicyName.BASELINE
    extend_lifetime_years: int = 0  # Number of years to extend server lifetime
    maintenance_increase: float = 1.15  # 15% increase in maintenance cost when extending lifetime
    skip_generations: List[str] = field(default_factory=list)
    oversubscription_factor: float = 1.0


@dataclass
class DemandShockConfig:
    enabled: bool = False
    shock_year: int = 5  # Year of shock (e.g., 5 = shock occurs at q=20)
    alpha: float = 3.0  # Shock magnitude multiplier (e.g., 3.0 = 3x demand at shock)
    post_shock_growth: float = 0.0  # Additional linear growth per quarter after shock (e.g., 0.01 = +1% per quarter)


@dataclass
class ModelSizeContractionConfig:
    enabled: bool = False
    contraction_year: int = 5  # Year of contraction (e.g., 5 = contraction occurs at q=20)
    beta: float = 0.8  # Per-task compute intensity multiplier (<1 = contraction)


@dataclass
class HwCapabilityShockConfig:
    enabled: bool = False
    shock_year: int = 5  # Year of shock (e.g., 5 = shock occurs at q=20)
    gamma: float = 3.0  # Shock magnitude multiplier (e.g., 3.0 = 3x capability at shock)
    epsilon: float = 0.05  # Additional linear growth per quarter after shock (e.g., 0.01 = +1% per quarter)


@dataclass
class PriceShockConfig:
    enabled: bool = False
    shock_year: int = 5  # Year of shock (e.g., 5 = shock occurs at q=20)
    delta: float = 0.6  # Shock magnitude multiplier (e.g., 0.6 = 60% of original price at shock)
    eta: float = 0.10  # Additional linear growth per quarter after shock (e.g., 0.01 = +1% per quarter)


@dataclass
class ScenariosConfig:
    demand_shock: DemandShockConfig = field(default_factory=DemandShockConfig)
    model_size_contraction: ModelSizeContractionConfig = field(default_factory=ModelSizeContractionConfig)
    hw_capability_shock: HwCapabilityShockConfig = field(default_factory=HwCapabilityShockConfig)
    price_shock: PriceShockConfig = field(default_factory=PriceShockConfig)


@dataclass
class StochasticVariable:
    distribution: DistributionType = DistributionType.NORMAL
    mu: Optional[float] = None
    sigma: Optional[float] = None
    sigma_fraction: Optional[float] = None
    min: Optional[float] = None
    mode: Optional[float] = None
    max: Optional[float] = None
    values: Optional[List[float]] = None
    cap_sigma: Optional[float] = None
    mean: Optional[float] = None
    correlation: Dict[str, float] = field(default_factory=dict)


@dataclass
class MonteCarloConfig:
    num_trials: int = 10_000
    seed: Optional[int] = None
    convergence_threshold: float = 0.01
    variables: Dict[str, StochasticVariable] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Rental (cloud) TCO configuration — SemiAnalysis-style 8-component model
# ---------------------------------------------------------------------------

@dataclass
class RentalStorageConfig:
    hot_cost_per_gb_mo: float = 0.12  # $/GB/month for hot storage (e.g., NVMe)
    warm_cost_per_gb_mo: float = 0.023  # $/GB/month for warm storage (e.g., HDD)
    cold_cost_per_gb_mo: float = 0.004  # $/GB/month for cold storage (e.g., Azure Blob Cool tier)
    tb_per_gpu: float = 5.0  # TB of storage needed per GPU (for model data, checkpoints, etc.)
    hot_fraction: float = 0.3  # Fraction of storage in hot tier (rest split between warm/cold)
    warm_fraction: float = 0.5  # Fraction of storage in warm tier (rest is cold)


@dataclass
class RentalNetworkingConfig:
    cost_per_month: float = 500.0  # $/month for network egress and interconnect
    egress_per_gb: float = 0.09  # $/GB for data egress from cloud
    egress_gb_per_month: float = 1000.0  # GB/month of egress per GPU (for model updates, data transfer, etc.)


@dataclass
class RentalControlPlaneConfig:
    cost_per_hour: float = 5.0  # $ per hour for control plane services (orchestration, monitoring, etc.)
    num_nodes: int = 4


@dataclass
class RentalSupportConfig:
    uplift_pct: float = 0.05


@dataclass
class RentalSetupConfig:
    engineering_hours: float = 160.0
    engineering_rate: float = 200.0
    cluster_hours: float = 80.0


@dataclass
class RentalDebuggingConfig:
    engineering_hours_per_month: float = 40.0
    engineering_rate: float = 200.0
    cluster_hours_per_month: float = 20.0


@dataclass
class RentalGoodputConfig:
    mode: GoodputMode = GoodputMode.CHECKPOINT_HOT
    num_failures_per_month: float = 4.0
    time_to_identify_min: float = 5.0
    checkpoint_interval_min: float = 30.0
    job_init_time_min: float = 10.0
    repair_time_min: float = 60.0
    failover_time_min: float = 2.0
    blast_radius_gpus: int = 8
    avg_job_size_gpus: int = 256
    network_overhead_pct: float = 0.0
    memory_overhead_pct: float = 0.0


@dataclass
class RentalConfig:
    gpu_cost_per_hour: float = 2.85
    num_gpus: int = 1024
    contract_months: int = 12
    discount_pct: float = 0.0
    spot_fraction: float = 0.0
    spot_discount_pct: float = 0.50
    orchestration_premium_pct: float = 0.0
    storage: RentalStorageConfig = field(
        default_factory=RentalStorageConfig)
    networking: RentalNetworkingConfig = field(
        default_factory=RentalNetworkingConfig)
    control_plane: RentalControlPlaneConfig = field(
        default_factory=RentalControlPlaneConfig)
    support: RentalSupportConfig = field(
        default_factory=RentalSupportConfig)
    setup: RentalSetupConfig = field(
        default_factory=RentalSetupConfig)
    debugging: RentalDebuggingConfig = field(
        default_factory=RentalDebuggingConfig)
    goodput: RentalGoodputConfig = field(
        default_factory=RentalGoodputConfig)


@dataclass
class Config:
    """Top-level configuration container."""
    simulation: SimulationConfig = field(default_factory=SimulationConfig)
    hardware: HardwareConfig = field(default_factory=HardwareConfig)
    model_trends: ModelTrendsConfig = field(default_factory=ModelTrendsConfig)
    performance: PerformanceConfig = field(default_factory=PerformanceConfig)
    workload: WorkloadConfig = field(default_factory=WorkloadConfig)
    tco: TcoConfig = field(default_factory=TcoConfig)
    networking: NetworkingConfig = field(default_factory=NetworkingConfig)
    policy: PolicyConfig = field(default_factory=PolicyConfig)
    scenarios: ScenariosConfig = field(default_factory=ScenariosConfig)
    monte_carlo: MonteCarloConfig = field(default_factory=MonteCarloConfig)
    rental: RentalConfig = field(default_factory=RentalConfig)


# ---------------------------------------------------------------------------
# YAML loading helpers
# ---------------------------------------------------------------------------

def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge *override* into *base* (in-place). Returns *base*."""
    for key, val in override.items():
        if key in base and isinstance(base[key], dict) and isinstance(val, dict):
            _deep_merge(base[key], val)
        else:
            base[key] = val
    return base


def _dict_to_dataclass(cls, data):
    if data is None:
        return cls()

    if isinstance(data, cls):
        return data

    if not isinstance(data, dict):
        raise TypeError(f"Expected dict or {cls.__name__}, got {type(data).__name__}")

    field_types = {f.name: f.type for f in cls.__dataclass_fields__.values()}
    kwargs = {}

    for fname, ftype in field_types.items():
        if fname not in data:
            continue
        val = data[fname]

        actual_type = ftype
        if isinstance(ftype, str):
            # Handle forward references
            actual_type = eval(ftype, globals(), locals())

        # Handle Optional types
        origin = getattr(actual_type, '__origin__', None)
        if origin is type(None):
            kwargs[fname] = val
            continue

        # Handle Dict[str, <dataclass>]
        if origin is dict and val is not None:
            args = getattr(actual_type, '__args__', ())
            if len(args) == 2 and hasattr(args[1], '__dataclass_fields__'):
                val = {k: _dict_to_dataclass(args[1], v) for k, v in val.items()}
            elif len(args) == 2 and hasattr(args[1], '__origin__'):
                pass  # dict of primitives
            elif len(args) == 2:
                # Check if it's StochasticVariable
                pass
            kwargs[fname] = val
            continue

        # Handle List[<dataclass>]
        if origin is list and val is not None:
            args = getattr(actual_type, '__args__', ())
            if args and hasattr(args[0], '__dataclass_fields__'):
                val = [_dict_to_dataclass(args[0], item) for item in val]
            kwargs[fname] = val
            continue

        # Handle Enum types (e.g., PerformanceModelType)
        if isinstance(actual_type, type) and issubclass(actual_type, Enum):
            if not isinstance(val, actual_type):
                val = actual_type(val)
            kwargs[fname] = val
            continue

        # Handle nested dataclass
        if hasattr(actual_type, '__dataclass_fields__') and isinstance(val, dict):
            val = _dict_to_dataclass(actual_type, val)

        kwargs[fname] = val

    return cls(**kwargs)


def _build_config_from_dict(raw: dict) -> Config:
    """Build a Config dataclass from a raw YAML dict, handling special types."""
    # Special handling for networking configs (Dict[str, NetworkTypeConfig])
    if 'networking' in raw and 'configs' in raw['networking']:
        net_cfgs = {}
        for name, ncfg in raw['networking']['configs'].items():
            net_cfgs[name] = _dict_to_dataclass(NetworkTypeConfig, ncfg)
        raw['networking']['configs'] = net_cfgs

    # Special handling for monte_carlo variables (Dict[str, StochasticVariable])
    if 'monte_carlo' in raw and 'variables' in raw.get('monte_carlo', {}):
        mc_vars = {}
        for name, vcfg in raw['monte_carlo']['variables'].items():
            mc_vars[name] = _dict_to_dataclass(StochasticVariable, vcfg)
        raw['monte_carlo']['variables'] = mc_vars

    # Special handling for known_gpus (List[GpuEntry])
    if 'hardware' in raw and 'known_gpus' in raw.get('hardware', {}):
        gpus = [_dict_to_dataclass(GpuEntry, g) for g in raw['hardware']['known_gpus']]
        raw['hardware']['known_gpus'] = gpus

    # Handle projection
    if 'hardware' in raw and 'projection' in raw.get('hardware', {}):
        raw['hardware']['projection'] = _dict_to_dataclass(
            HardwareProjection, raw['hardware']['projection']
        )

    cfg_kwargs = {}
    for fname in Config.__dataclass_fields__:
        if fname in raw:
            val = raw[fname]
            ftype = Config.__dataclass_fields__[fname].type
            if isinstance(ftype, str):
                ftype = eval(ftype, globals(), locals())
            if hasattr(ftype, '__dataclass_fields__') and isinstance(val, dict):
                cfg_kwargs[fname] = _dict_to_dataclass(ftype, val)
            else:
                cfg_kwargs[fname] = val
        # else: use default

    return Config(**cfg_kwargs)


def load_config(
    path: str | Path | None = None,
    overrides: str | Path | None = None,
) -> Config:
    """Load configuration from YAML file(s).

    Parameters
    ----------
    path : str or Path, optional
        Path to the base YAML config.  If *None*, uses built-in defaults.
    overrides : str or Path, optional
        Path to an override YAML (e.g., a scenario file).  Values in this
        file are deep-merged on top of *path*.

    Returns
    -------
    Config
        Fully populated configuration dataclass.
    """
    if path is None:
        # Use the bundled default config
        default_path = Path(__file__).resolve().parent.parent / "configs" / "default.yaml"
        if default_path.exists():
            path = default_path
        else:
            return Config()

    path = Path(path)
    with open(path, "r", encoding="utf-8") as f:
        raw: dict = yaml.safe_load(f) or {}

    # Process includes: load and merge each referenced file (relative to path)
    includes = raw.pop("includes", None)
    if includes:
        base_dir = path.parent
        for inc_file in includes:
            inc_path = base_dir / inc_file
            with open(inc_path, "r", encoding="utf-8") as f:
                inc_raw: dict = yaml.safe_load(f) or {}
            _deep_merge(raw, inc_raw)

    if overrides is not None:
        with open(Path(overrides), "r", encoding="utf-8") as f:
            override_raw: dict = yaml.safe_load(f) or {}
        _deep_merge(raw, override_raw)

    return _build_config_from_dict(raw)


def config_to_dict(cfg: Config) -> dict:
    """Convert a Config back to a plain dict (for serialization)."""
    def _convert(obj):
        if dataclasses.is_dataclass(obj):
            return {
                k: _convert(v)
                for k, v in dataclasses.asdict(obj).items()  # type: ignore
            }
        return obj
    return _convert(cfg)

"""Lifecycle policy strategies for server fleet management.

Each policy decides, at each quarter, whether to decommission existing
servers and what generation to buy when provisioning new capacity.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .config import Config, PolicyName
from .hardware import ServerSpec
from .tco import ServerBatch


# ---------------------------------------------------------------------------
# Policy decision
# ---------------------------------------------------------------------------

@dataclass
class PolicyDecision:
    """Decision made by a policy at a given quarter."""
    # Batch IDs to decommission immediately this quarter
    force_decommission: List[int] = field(default_factory=list)
    # Index into server_roadmap for new server purchases (None = use latest available)
    buy_gen_index: Optional[int] = None
    # Override decom quarter for specific batches (batch_id -> new_decom_q)
    decom_overrides: Dict[int, int] = field(default_factory=dict)
    # Maintenance multiplier for specific batches (batch_id -> multiplier)
    maintenance_overrides: Dict[int, float] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Base policy
# ---------------------------------------------------------------------------

class BasePolicy(ABC):
    """Abstract base class for lifecycle policies."""

    @abstractmethod
    def on_quarter(
        self,
        q: int,
        active_batches: List[ServerBatch],
        available_gens: List[ServerSpec],
        new_gen_released: bool,
        cfg: Config,
    ) -> PolicyDecision:
        """Make fleet management decisions for quarter *q*.

        Parameters
        ----------
        q : int
            Current quarter index.
        active_batches : list of ServerBatch
            Currently active server batches.
        available_gens : list of ServerSpec
            Server generations available for purchase (released by quarter q).
        new_gen_released : bool
            True if a new GPU generation becomes available this quarter.
        cfg : Config
            Configuration.

        Returns
        -------
        PolicyDecision
        """
        ...


# ---------------------------------------------------------------------------
# Concrete policies
# ---------------------------------------------------------------------------

class BaselinePolicy(BasePolicy):
    """Default policy: buy latest gen to cover shortfall, decommission at natural EOL.

    Old servers remain active until their natural lifetime expires.
    When new hardware becomes available, it is used for new purchases only.
    """

    def on_quarter(self, q, active_batches, available_gens, new_gen_released, cfg):
        # No forced decommissions, buy latest available gen
        return PolicyDecision(
            buy_gen_index=len(available_gens) - 1 if available_gens else None,
        )


class ReplaceAllPolicy(BasePolicy):
    """When new hardware arrives, immediately decommission ALL older servers.

    All demand is re-provisioned on the newest generation.
    """

    def on_quarter(self, q, active_batches, available_gens, new_gen_released, cfg):
        decision = PolicyDecision(
            buy_gen_index=len(available_gens) - 1 if available_gens else None,
        )

        if new_gen_released and len(available_gens) >= 2:
            newest = available_gens[-1]
            for batch in active_batches:
                if batch.server.code_name != newest.code_name:
                    decision.force_decommission.append(batch.batch_id)

        return decision


class ExtendLifetimePolicy(BasePolicy):
    """Extend server lifetime beyond the default, with increased maintenance costs.

    Parameters
    ----------
    extend_years : int
        Additional years of lifetime beyond ``server_lifetime_years``.
    maintenance_increase : float
        Maintenance cost multiplier for servers past their normal lifetime
        (e.g., 1.15 = 15% increase).
    """

    def __init__(self, extend_years: int = 1, maintenance_increase: float = 1.15):
        self.extend_years = extend_years
        self.maintenance_increase = maintenance_increase

    def on_quarter(self, q, active_batches, available_gens, new_gen_released, cfg):
        qpy = cfg.simulation.quarters_per_year
        normal_lifetime_q = cfg.tco.server_lifetime_years * qpy
        extended_lifetime_q = normal_lifetime_q + self.extend_years * qpy

        decision = PolicyDecision(
            buy_gen_index=len(available_gens) - 1 if available_gens else None,
        )

        for batch in active_batches:
            age = q - batch.start_q
            # Extend decommission date
            new_decom = batch.start_q + extended_lifetime_q
            if new_decom > batch.decom_q:
                decision.decom_overrides[batch.batch_id] = new_decom
            # Increase maintenance after normal lifetime
            if age >= normal_lifetime_q:
                decision.maintenance_overrides[batch.batch_id] = self.maintenance_increase

        return decision


class SkipGenerationPolicy(BasePolicy):
    """Same as baseline, but specified GPU generations are never purchased.

    The ``skip_generations`` list in the config controls which code_names
    are excluded.  The simulation's server roadmap is pre-filtered, so this
    policy just delegates to baseline behavior.
    """

    def on_quarter(self, q, active_batches, available_gens, new_gen_released, cfg):
        return PolicyDecision(
            buy_gen_index=len(available_gens) - 1 if available_gens else None,
        )


class DisaggregatedPolicy(BasePolicy):
    """Disaggregated LLM serving: prefill on newer GPUs, decode on older GPUs.

    When a new generation arrives, newer servers handle prefill (compute-
    intensive) while older servers handle decode (memory-bound).

    This policy is implemented at the simulation level rather than purely
    in the policy decision, because it requires splitting demand into
    prefill and decode components.  The policy signals that disaggregation
    is active; the simulation engine handles the split.
    """

    def __init__(self, replace_old: bool = False):
        self.replace_old = replace_old

    def on_quarter(self, q, active_batches, available_gens, new_gen_released, cfg):
        decision = PolicyDecision(
            buy_gen_index=len(available_gens) - 1 if available_gens else None,
        )

        if self.replace_old and new_gen_released and len(available_gens) >= 2:
            newest = available_gens[-1]
            for batch in active_batches:
                if batch.server.code_name != newest.code_name:
                    decision.force_decommission.append(batch.batch_id)

        return decision


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def get_policy(cfg: Config) -> BasePolicy:
    """Create a policy instance based on configuration.

    Parameters
    ----------
    cfg : Config
        Configuration with ``policy.name`` and related parameters.

    Returns
    -------
    BasePolicy
    """
    name = cfg.policy.name

    if name == PolicyName.BASELINE:
        return BaselinePolicy()

    elif name == PolicyName.REPLACE_ALL:
        return ReplaceAllPolicy()

    elif name == PolicyName.EXTEND_LIFETIME:
        return ExtendLifetimePolicy(
            extend_years=cfg.policy.extend_lifetime_years,
            maintenance_increase=cfg.policy.maintenance_increase,
        )

    elif name == PolicyName.SKIP_GENERATION:
        return SkipGenerationPolicy()

    elif name == PolicyName.DISAGGREGATED:
        return DisaggregatedPolicy(replace_old=False)

    elif name == PolicyName.DISAGGREGATED_REPLACE:
        return DisaggregatedPolicy(replace_old=True)

    else:
        raise ValueError(f"Unknown policy: {cfg.policy.name}")

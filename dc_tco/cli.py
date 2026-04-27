"""Command-line interface for DC-TCO framework."""

from __future__ import annotations

import argparse
import json

from pathlib import Path

from typing import Optional

import numpy as np
import pandas as pd

from .config import load_config
from .simulation import run_simulation
from .simulation import quarter_label
from .scenarios import run_monte_carlo
from .scenarios import run_policy_sweep
from .rental import compute_rental_tco
from .rental import RentalTcoResult


def main(
    argv: Optional[list[str]] = None
) -> None:
    parser = argparse.ArgumentParser(
        prog="dc-tco",
        description="TCO-driven AI datacenter lifecycle evaluation framework",
    )
    sub = parser.add_subparsers(dest="command", help="Available commands")

    # --- run ---
    run_p = sub.add_parser("run", help="Run a single simulation")
    run_p.add_argument("--config", "-c", default=None, help="Path to YAML config")
    run_p.add_argument("--scenario", "-s", default=None, help="Path to scenario override YAML")
    run_p.add_argument("--policy", "-p", default=None, help="Override policy name")
    run_p.add_argument("--output", "-o", default=None, help="Output directory for results")

    # --- sweep ---
    sweep_p = sub.add_parser("sweep", help="Run policy sweep")
    sweep_p.add_argument("--config", "-c", default=None, help="Path to YAML config")
    sweep_p.add_argument("--scenario", "-s", default=None, help="Scenario override")
    sweep_p.add_argument("--output", "-o", default="results", help="Output directory")
    sweep_p.add_argument("--no-disagg", action="store_true", help="Skip disaggregated variants")
    sweep_p.add_argument("--no-skip", action="store_true", help="Skip generation-skipping combos")

    # --- monte-carlo ---
    mc_p = sub.add_parser("monte-carlo", help="Run Monte Carlo simulation")
    mc_p.add_argument("--config", "-c", default=None, help="Path to YAML config")
    mc_p.add_argument("--trials", "-n", type=int, default=None, help="Number of trials")
    mc_p.add_argument("--policy", "-p", default=None, help="Policy name")
    mc_p.add_argument("--seed", type=int, default=None, help="Random seed")
    mc_p.add_argument("--output", "-o", default="results", help="Output directory")

    # --- rental ---
    rental_p = sub.add_parser("rental", help="Compute rental/cloud TCO")
    rental_p.add_argument("--config", "-c", default=None,
                          help="Path to YAML config")
    rental_p.add_argument("--months", "-m", type=int, default=None,
                          help="Override contract months")
    rental_p.add_argument("--gpus", "-g", type=int, default=None,
                          help="Override number of GPUs")
    rental_p.add_argument("--output", "-o", default=None,
                          help="Output directory for results")

    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help()
        return

    if args.command == "run":
        _cmd_run(args)
    elif args.command == "sweep":
        _cmd_sweep(args)
    elif args.command == "monte-carlo":
        _cmd_monte_carlo(args)
    elif args.command == "rental":
        _cmd_rental(args)


def _cmd_run(
    args: argparse.Namespace
) -> None:
    cfg = load_config(args.config, args.scenario)
    if args.policy:
        cfg.policy.name = args.policy

    print(f"Running simulation: policy={cfg.policy.name}, "
          f"years={cfg.simulation.total_years}")

    result = run_simulation(cfg)

    print(f"\n{'='*50}")
    print(f"Total TCO: ${result.total_tco / 1e9:.3f} B")
    print(f"  CapEx:   ${result.tco_breakdown.total_capex / 1e9:.3f} B")
    print(f"  OpEx:    ${result.tco_breakdown.total_opex / 1e9:.3f} B")
    print(f"{'='*50}")

    # Final state
    final = result.quarterly_states[-1]
    base_year = cfg.simulation.base_year
    qpy = cfg.simulation.quarters_per_year
    print(f"\nFinal state ({quarter_label(final.quarter, base_year, qpy)}):")
    print(f"  Servers: {final.total_servers:,}")
    print(f"  Racks:   {final.total_racks:,}")
    print(f"  Demand:  {final.demand:,.0f} req/s")

    if args.output:
        _save_result(result, args.output)


def _cmd_sweep(
    args: argparse.Namespace
) -> None:
    cfg = load_config(args.config, args.scenario)

    print(f"Running policy sweep (disaggregated={not args.no_disagg}, "
          f"skip-gen={not args.no_skip})")

    results = run_policy_sweep(
        cfg,
        include_disaggregated=not args.no_disagg,
        include_skip=not args.no_skip,
    )

    # Print summary
    print(f"\n{'Policy':<40s} {'Variant':<15s} {'TCO ($B)':>12s}")
    print("-" * 70)

    baseline_tco = next(
        (r.total_tco for r in results if r.policy_name == "baseline" and r.variant == "regular"),
        results[0].total_tco if results else 1.0,
    )

    for r in sorted(results, key=lambda x: x.total_tco):
        pct = (r.total_tco / baseline_tco - 1) * 100
        print(f"  {r.policy_name:<38s} {r.variant:<15s} "
              f"${r.total_tco/1e9:>10.3f}  ({pct:+.1f}%)")

    if args.output:
        _save_sweep(results, args.output)


def _cmd_monte_carlo(
    args: argparse.Namespace
) -> None:
    cfg = load_config(args.config)
    if args.trials:
        cfg.monte_carlo.num_trials = args.trials
    if args.seed is not None:
        cfg.monte_carlo.seed = args.seed

    policy = args.policy or cfg.policy.name
    print(f"Running Monte Carlo: {cfg.monte_carlo.num_trials} trials, "
          f"policy={policy}")

    mc_result = run_monte_carlo(cfg, policy_name=policy)

    print(f"\n{'='*50}")
    print(f"Monte Carlo Results ({mc_result.num_trials} trials)")
    print(f"  Mean TCO:    ${mc_result.mean / 1e9:.3f} B")
    print(f"  Std Dev:     ${mc_result.std / 1e9:.3f} B")
    print(f"  95% CI:      ${mc_result.ci_95[0]/1e9:.3f} - ${mc_result.ci_95[1]/1e9:.3f} B")
    print(f"  P5 / P50 / P95: ${mc_result.p5/1e9:.3f} / ${mc_result.p50/1e9:.3f} / ${mc_result.p95/1e9:.3f} B")
    print(f"  Converged:   {mc_result.converged}")
    print(f"{'='*50}")

    if args.output:
        Path(args.output).mkdir(parents=True, exist_ok=True)
        np.save(Path(args.output) / "mc_tco_values.npy", mc_result.tco_values)
        print(f"\nResults saved to {args.output}/")


def _cmd_rental(
    args: argparse.Namespace
) -> None:
    cfg = load_config(args.config)
    if args.months:
        cfg.rental.contract_months = args.months
    if args.gpus:
        cfg.rental.num_gpus = args.gpus

    rc = cfg.rental
    print(f"Computing rental TCO: {rc.num_gpus} GPUs × "
          f"{rc.contract_months} months")

    result = compute_rental_tco(cfg)

    print(f"\n{'='*50}")
    print("Rental TCO Summary")
    print(f"{'='*50}")
    m = result.monthly
    print(f"  Monthly total:     ${m.total:>12,.0f}")
    print(f"    GPU compute:     ${m.gpu:>12,.0f}")
    print(f"    Storage:         ${m.storage:>12,.0f}")
    print(f"    Networking:      ${m.networking:>12,.0f}")
    print(f"    Control plane:   ${m.control_plane:>12,.0f}")
    print(f"    Support:         ${m.support:>12,.0f}")
    print(f"    Goodput loss:    ${m.goodput:>12,.0f}")
    print(f"    Setup amort.:    ${m.setup:>12,.0f}")
    print(f"    Debugging:       ${m.debugging:>12,.0f}")
    gm = result.goodput_metrics
    print(f"  Goodput expense:   ${gm.goodput_expense:>12,.0f}/mo")
    print(f"  Goodput util.:     {gm.utilization:>11.1%}")
    print(f"  Contract total:    ${result.contract_total.total:>12,.0f}")
    eff = result.cost_per_gpu_hour_effective
    print(f"  Eff. $/GPU-hr:     ${eff:>12.2f}")
    print(f"{'='*50}")

    if args.output:
        _save_rental(result, args.output)


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

def _save_result(result, output_dir: str) -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    base_year = result.config.simulation.base_year
    qpy = result.config.simulation.quarters_per_year

    # Quarterly states to CSV
    rows = []
    for s in result.quarterly_states:
        rows.append({
            "quarter": s.quarter,
            "quarter_label": quarter_label(s.quarter, base_year, qpy),
            "year": s.year,
            "demand": s.demand,
            "model_size_B": s.model_size_B,
            "total_servers": s.total_servers,
            "total_racks": s.total_racks,
            "total_capacity": s.total_capacity,
            "shortfall": s.shortfall,
            "servers_added": s.servers_added,
            "stranded_power_kw": s.stranded_power_kw,
            "tco_capex": s.tco.total_capex,
            "tco_opex": s.tco.total_opex,
            "tco_total": s.tco.total,
        })
    pd.DataFrame(rows).to_csv(out / "quarterly_states.csv", index=False)

    # Summary JSON
    summary = {
        "total_tco": result.total_tco,
        "total_capex": result.tco_breakdown.total_capex,
        "total_opex": result.tco_breakdown.total_opex,
        "policy": result.config.policy.name,
        "total_years": result.config.simulation.total_years,
    }
    with open(out / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nResults saved to {out}/")


def _save_sweep(results, output_dir: str) -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    rows = []
    for r in results:
        row = {
            "policy": r.policy_name,
            "variant": r.variant,
            "total_tco": r.total_tco,
            "extend_years": r.extend_years,
            "skip_gens": ",".join(r.skip_gens),
        }
        row.update(r.tco_breakdown)
        rows.append(row)

    pd.DataFrame(rows).to_csv(out / "policy_sweep.csv", index=False)
    print(f"\nSweep results saved to {out}/policy_sweep.csv")


def _save_rental(
    result: RentalTcoResult,
    output_dir: str
) -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    m = result.monthly
    gm = result.goodput_metrics
    summary = {
        "monthly_total": m.total,
        "contract_total": result.contract_total.total,
        "contract_months": len(result.monthly_series),
        "cost_per_gpu_hour_effective": result.cost_per_gpu_hour_effective,
        "goodput_expense": gm.goodput_expense,
        "goodput_utilization": gm.utilization,
        "monthly_breakdown": {
            "gpu": m.gpu,
            "storage": m.storage,
            "networking": m.networking,
            "control_plane": m.control_plane,
            "support": m.support,
            "goodput": m.goodput,
            "setup": m.setup,
            "debugging": m.debugging,
        },
    }
    with open(out / "rental_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nRental results saved to {out}/rental_summary.json")


if __name__ == "__main__":
    main()

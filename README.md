# 🧭 AI Datacenter Lifecycle Compass

A TCO-driven evaluation framework for AI datacenter lifecycle management, spanning build, IT provisioning, and operation stages over a 15-year planning horizon.

## 🔎 Overview

DC-TCO provides a parameterized simulation engine that models the total cost of ownership (TCO) of an AI datacenter across its full lifecycle. Key capabilities:

- **GPU hardware roadmap projection** — known GPUs (P100 through B200) with NVLink interconnect specs and configurable future generation extrapolation
- **AI model size trend modeling** — based on the Epoch AI dataset, with distillation and migration scheduling
- **Roofline-based performance model** — compute, memory, and interconnect-bound throughput estimation for LLM inference (theoretical, simplified, or CSV-based)
- **Lifecycle simulation** — quarterly granularity over 15 years (60 quarters) with demand-driven provisioning
- **5 lifecycle policies** — baseline, replace-all, extend-lifetime, skip-generation, disaggregated serving
- **Scenario shocks** — demand plateau, model size contraction, hardware capability disruption, GPU price shocks
- **Monte Carlo engine** — correlated stochastic sampling with convergence validation
- **Policy sweep** — combinatorial evaluation across policies, generation skipping, and lifetime extension
- **Ownership vs rental comparison** — cloud GPU rental TCO calculator for build-vs-buy analysis

## 🛠️ Installation

```bash
pip install -e .
```

With plotting support:

```bash
pip install -e ".[plot]"
```

## 🚀 Quick Start

### 🐍 Python API

```python
from dc_tco.config import load_config
from dc_tco.simulation import run_simulation

# Load default configuration
cfg = load_config()

# Run a baseline simulation
result = run_simulation(cfg)

print(f"Total TCO: ${result.total_tco / 1e9:.3f} B")
print(f"  CapEx: ${result.tco_breakdown.total_capex / 1e9:.3f} B")
print(f"  OpEx:  ${result.tco_breakdown.total_opex / 1e9:.3f} B")
```

### 💻 CLI

```bash
# Single simulation
dc-tco run --config configs/default.yaml --policy baseline

# Policy sweep
dc-tco sweep --config configs/default.yaml --output results/

# Monte Carlo
dc-tco monte-carlo --config configs/default.yaml --trials 10000 --output results/
```

### ⚡ Scenario Overrides

Apply scenario shocks by passing an override YAML:

```python
cfg = load_config("configs/default.yaml", "configs/scenarios/demand_shock.yaml")
result = run_simulation(cfg)
```

```bash
dc-tco run -c configs/default.yaml -s configs/scenarios/demand_shock.yaml
```

## ⚙️ Configuration

Configuration is modular — `configs/default.yaml` assembles domain-specific files via `includes`:

| File | Contents |
| ---- | -------- |
| `configs/default.yaml` | Hub file with simulation timeline + includes |
| `configs/hardware.yaml` | GPU specs (P100–B200), NVLink bandwidth, projection modes, performance model |
| `configs/workload.yaml` | AI model size trends, workload demand (scaling, initial demand) |
| `configs/tco.yaml` | TCO parameters (power, cooling, facility, racks) and networking configs |
| `configs/policy.yaml` | Lifecycle policy selection and scenario shocks |
| `configs/monte_carlo.yaml` | Stochastic variables, distributions, and correlations |
| `configs/rental.yaml` | Cloud GPU rental pricing for ownership-vs-rental analysis |
| `configs/scenarios/` | Override files for individual shock scenarios |

Key parameters:

| Section | Key Parameters |
| ------- | -------------- |
| `simulation` | `base_year` (2016), `total_years` (15), `quarters_per_year` (4) |
| `hardware` | 7 known GPUs (P100–B200) with TDP, TFLOPS, mem BW/cap, NVLink BW, cost |
| `workload` | `scaling` (exponential), `scaling_factor` (1.0355/quarter ≈ 15%/year) |
| `tco` | `power_cost_per_kwh`, facility/power/cooling provisioning costs, `building_lifetime_years` (15) |
| `policy` | `name`, `skip_generations`, `extend_lifetime_years`, `oversubscription_factor` |
| `scenarios` | Demand shock, model size contraction, HW capability shock, price shock |
| `monte_carlo` | 8 stochastic variables with distributions and correlations |

## 🧮 TCO Model

The TCO model captures:

**CapEx** — servers, racks, power provisioning, cooling provisioning, facility, networking
**OpEx** — energy, cooling, maintenance, networking

- Power and cooling provisioning are per-watt one-time costs amortized over the building lifetime (15 years)
- Facility cost is proportional to power usage relative to facility capacity (default to 8 MW), amortized over building lifetime
- Server CapEx is amortized over the server amortization period (5 years)
- OpEx accrues only while servers are active

## 📐 Lifecycle Policies

| Policy | Behavior |
| ------ | -------- |
| `baseline` | Buy latest gen on shortfall, decommission at natural EOL |
| `replace_all` | Decommission all older servers when new gen arrives |
| `extend_lifetime` | Extend server lifetime by N years with increased maintenance |
| `skip_generation` | Skip specified GPU generations entirely |
| `disaggregated` | Prefill on newer GPUs, decode on older GPUs |

## 🎲 Monte Carlo

The Monte Carlo engine samples 8 stochastic variables with correlated distributions:

- Workload growth (log-normal)
- Model size growth (log-normal)
- GPU perf/watt improvement (normal)
- GPU price change (triangular)
- Release interval (discrete)
- Electricity price (log-normal)
- PUE (normal)
- Server lifetime (discrete)

Correlations (e.g., workload growth ~ model size growth, GPU perf ~ price) are modeled via multivariate normal sampling with inverse CDF transforms.

For full methodology details, see [doc/MONTE_CARLO.md](doc/MONTE_CARLO.md).

## 🗂️ Project Structure

```text
dc-tco-framework/
├── configs/
│   ├── default.yaml              # Hub file — includes domain configs below
│   ├── hardware.yaml             # GPU specs, NVLink BW, projections, performance model
│   ├── workload.yaml             # AI model trends, workload demand
│   ├── tco.yaml                  # TCO parameters, networking configs
│   ├── policy.yaml               # Lifecycle policy, scenario shocks
│   ├── monte_carlo.yaml          # Monte Carlo stochastic variables
│   ├── rental.yaml               # Cloud GPU rental pricing
│   └── scenarios/                # Override files for shock scenarios
├── data/
│   └── notable_ai_models.csv     # Epoch AI notable models dataset
├── dc_tco/
│   ├── config.py                 # YAML loading with includes, dataclass config hierarchy
│   ├── hardware.py               # GPU specs, projection, server roadmap
│   ├── models.py                 # AI model size trends, distillation
│   ├── performance.py            # Roofline model (theoretical + simplified + CSV)
│   ├── demand.py                 # Workload demand curves
│   ├── networking.py             # Network cost configurations
│   ├── tco.py                    # CapEx/OpEx computation (incl. facility)
│   ├── policies.py               # 5 lifecycle policy strategies
│   ├── simulation.py             # Core quarterly lifecycle loop
│   ├── scenarios.py              # Shocks, Monte Carlo, policy sweep
│   ├── rental.py                 # Ownership vs rental TCO calculator
│   ├── plotting.py               # Visualization functions
│   └── cli.py                    # Command-line interface
├── notebooks/
│   ├── 01_hardware_trends.ipynb  # GPU roadmap visualization
│   ├── 02_model_trends.ipynb     # AI model size projections
│   ├── 03_tco_breakdown.ipynb    # TCO breakdown analysis
│   ├── 04_policy_comparison.ipynb # Policy comparison & sweep
│   ├── 05_scenario_analysis.ipynb # Scenario shocks & Monte Carlo
│   └── 06_ownership_vs_rental.ipynb # Build vs rent analysis
├── tests/                        # Unit and integration tests
├── scripts/                      # Utility scripts
└── docs/                         # Additional documentation
```

## 📄 Citation

If you use the AI Datacenter Lifecycle Compass in research, please cite:

```bibtex
@inproceedings{stojkovic2026compass,
  title={{Rearchitecting the Datacenter Lifecycle for AI}},
  author={Jovan Stojkovic, Chaojie Zhang, Íñigo Goiri, Ricardo Bianchini},
  booktitle = {ISCA},
  year={2026}
}
```

## 🤝 Contributing

Pull requests are welcome! Please open an issue for major changes.
Please ensure that your contributions adhere to the [Microsoft Open Source Code of Conduct](CODE_OF_CONDUCT.md).
More details in the [contributing guidelines](CONTRIBUTING.md).

## 📜 License

[MIT License](LICENSE).

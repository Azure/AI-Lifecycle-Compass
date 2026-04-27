# TCO Models: Ownership vs Rental

This framework implements two complementary TCO (Total Cost of Ownership)
calculators for GPU infrastructure.
They answer different questions for different audiences.

## DC Ownership Model (Original)

The ownership model simulates the **full lifecycle of an AI datacenter**
over 15 years, from the perspective of a cloud provider or hyperscaler
who builds, provisions, and operates the facility.

### Ownership — What It Models

- **Building stage** — Facility construction, power provisioning,
  cooling infrastructure
- **IT provisioning stage** — Server procurement across multiple GPU
  generations (H100 → B200 → future), rack deployment,
  network topology (Ethernet, InfiniBand, OCS)
- **Operation stage** — Energy consumption (PUE-based), cooling OpEx,
  hardware maintenance, server decommissioning and refresh cycles

### Cost Structure

```text
CapEx (one-time, amortized)          OpEx (recurring)
├── Server purchase                  ├── Energy (electricity × PUE)
├── Rack infrastructure              ├── Cooling operations
├── Power provisioning ($/watt)      ├── Maintenance (per server/yr)
├── Cooling provisioning ($/watt)    └── Network operations
├── Facility construction
└── Network (switches, cabling)
```

### Key Characteristics

- **Dynamic demand** — Workload grows over time following model-size
  scaling laws (Epoch AI data)
- **Hardware refresh** — Multiple GPU generations are deployed and
  retired; fleet composition changes every few years
- **Policy-driven** — Supports different refresh strategies (baseline,
  extend lifetime, skip generations, disaggregated prefill/decode)
- **Monte Carlo** — Uncertainty analysis over key parameters (GPU cost,
  power price, demand growth)

### Ownership — When to Use

You are a **cloud provider** planning a datacenter investment.
You need to decide: when to buy new GPUs, how long to keep old ones,
whether to skip a generation, and how power/cooling provisioning
affects 15-year TCO.

---

## Rental Model (New)

The rental model computes the **monthly cost of renting a GPU cluster**
from a cloud provider, from the perspective of a customer (enterprise,
startup, or research lab).

### Why "Rental"?

We call it "rental" rather than "cloud" because the model applies to any
scenario where you **pay for GPU access without owning the hardware**:

- **Public cloud** — AWS, Azure, GCP GPU instances (on-demand or reserved)
- **GPU cloud specialists** — CoreWeave, Lambda, Together, Crusoe
- **Bare-metal rental** — Dedicated servers from OVH, Hetzner, Vultr
- **Colocation with leased hardware** — You rent rack space and lease GPUs
- **Internal chargeback** — Enterprise IT charging business units per GPU-hour

The common thread is: you don't build the datacenter, you don't buy the
GPUs, and you don't manage the power/cooling. You pay a recurring fee
($/GPU-hr) and your costs are purely operational. That's a **rental**
relationship, regardless of whether the provider calls it "cloud",
"bare metal", or "reserved capacity".

This model follows the methodology from
[SemiAnalysis: "How Much Do GPU Clusters Really Cost?"](https://newsletter.semianalysis.com/p/how-much-do-gpu-clusters-really-cost)
and their [ClusterMax TCO calculator](https://www.clustermax.ai/tco).

### Rental — What It Models

Eight cost components that make up the true cost of running a rented
GPU cluster:

```text
1. GPU compute       — $/GPU-hr × hours × GPUs (spot/discount options)
2. Storage           — Hot (NVMe) + warm (S3) + cold (archive) tiered
3. Networking        — Fixed costs + egress fees
4. Control plane     — Login nodes, schedulers, monitoring (CPU instances)
5. Support           — Provider support tier (% of base bill)
6. Goodput loss      — Wasted spend from hardware failures and recovery
7. Setup             — Engineering time + GPU-hours for bring-up (amortized)
8. Debugging         — Ongoing engineering + GPU-hours lost to debug jobs
```

### Goodput: The Hidden Cost

The key insight from the SemiAnalysis model is that **GPU failures waste
real money**, and the recovery strategy determines how much. The model
implements three formulas from their "Grand Unifying Theory of Goodput":

| Mode                | Recovery strategy            | Waste per failure                                    |
| ------------------- | ---------------------------- | ---------------------------------------------------- |
| **Checkpoint Cold** | Wait for repair, reload ckpt | `[max(t_detect, t_ckpt/2) + t_init] * j + t_rep * b` |
| **Checkpoint Hot**  | Fail over to hot spare       | `[max(t_detect, t_ckpt/2) + t_init + t_rep] * j`     |
| **Fault Tolerant**  | Framework handles it         | `[t_detect + t_failover] * j + t_rep * b + overhead` |

Where `j` = job size (GPUs), `b` = blast radius (GPUs),
`t_rep` = repair time.

Each formula outputs a **goodput utilization** (0–1) representing what
fraction of GPU-hours are actually productive.

### Rental — When to Use

You are a **customer** evaluating GPU cluster options. You want to know:
what's the real monthly cost beyond the sticker price? How does goodput
loss affect your effective $/GPU-hr? At what rental price should you
build your own cluster instead?

---

## Side-by-Side Comparison

| Aspect             | DC Ownership                          | Rental                            |
| ------------------ | ------------------------------------- | --------------------------------- |
| **Perspective**    | Cloud provider / hyperscaler          | Customer / tenant                 |
| **Time horizon**   | 15 years (multi-generation lifecycle) | 1–36 months (contract term)       |
| **Cost structure** | CapEx-heavy (60–70% of TCO)           | 100% OpEx                         |
| **Hardware**       | Multi-generation fleet, evolving      | Single GPU type, fixed cluster    |
| **Demand**         | Dynamic (grows with AI trends)        | Static (fixed GPU count)          |
| **Refresh cycles** | Explicit (buy, amortize, decommission)| Implicit (provider's problem)     |
| **Power/cooling**  | Detailed PUE model, provisioning CapEx| Bundled in GPU $/hr               |
| **Networking**     | Topology-aware (Ethernet/IB/OCS)      | Flat monthly + egress             |
| **Goodput**        | Single utilization factor             | 3 failure-mode formulas           |
| **Output**         | Quarterly time series over lifecycle  | Monthly breakdown + contract total|

## Running the Models

```bash
# Ownership: run 15-year lifecycle simulation
dc-tco run --config configs/default.yaml

# Rental: compute monthly cluster TCO
dc-tco rental --config configs/default.yaml --gpus 1024 --months 12

# Compare in notebook
jupyter notebook notebooks/06_ownership_vs_rental.ipynb
```

## Configuration

Both models are configured through YAML files in `configs/`:

- **Ownership**: `hardware.yaml`, `tco.yaml`, `workload.yaml`, `policy.yaml`
- **Rental**: `rental.yaml` (8 sections matching the 8 cost components)
- **Shared**: `default.yaml` includes all config files

See `configs/rental.yaml` for annotated rental parameters with source
references.

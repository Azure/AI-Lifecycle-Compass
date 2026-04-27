# 🎲 Monte Carlo Methodology

To account for uncertainty in TCO, instead of using fixed parameters, we run Monte Carlo
simulations where we model inputs as random variables. Each trial represents a plausible future
trajectory of workload growth, model scaling, hardware evolution, pricing, and energy costs. The
simulator then deterministically computes capacity planning, server acquisitions/decommissions, and
annual CapEx/OpEx, producing a single TCO outcome. Repeating this yields a distribution over
lifecycle costs.

## 🔀 Randomized Variables and Distributions

The table below summarizes the stochastic inputs, their distributions, parameterization, and
rationale. Distribution choices are guided by historical AI model scaling trends, GPU release data,
and publicly reported datacenter cost variability. In most experiments, bounds are chosen
conservatively to avoid unrealistic tail behavior. We separately study a few extreme cases by
significantly diverging some variables from trends.

| Variable | Distribution | Parameters / Bounds | Notes / Correlations |
| -------- | ------------ | ------------------- | -------------------- |
| Workload growth factor | Log-normal | $\mu=\log(1.05)$, $\sigma=0.05$ | Positive-only growth; correlated with model size growth ($\rho=0.4$) |
| Model size annual growth | Log-normal | Fit to historical P50 trend; capped at $\pm 2\sigma$ | Captures uncertainty in scaling-law extrapolation |
| GPU Perf/W improvement | Normal | Mean from regression, $\sigma=0.1\mu$ | Correlated with GPU cost improvement ($\rho=-0.5$) |
| GPU price per generation | Triangular | min = $-15\%$, mode = $0\%$, max = $+20\%$ | Reflects supply-chain variability |
| Release interval | Discrete | $\{1,\ 1.5,\ 2\}$ years | Uniform sampling |
| Electricity price ($/kWh) | Log-normal | Mean = regional avg., $\sigma=15\%$ | Independent across trials |
| Cooling efficiency (PUE) | Normal | Mean = baseline, $\sigma=0.05$ | Affects total energy cost |
| Server lifetime | Discrete | $\{4,\ 5,\ 6\}$ years | Uniform sampling |

## 🔗 Correlation Modeling

We draw samples from a multivariate normal distribution with covariance matrix $\Sigma$, whose
entries encode empirically derived pairwise correlations (see table above). We then map these
samples to the desired marginals via inverse CDF transforms (e.g., log-normal, triangular). All
remaining variables are sampled independently.

## 🔢 Number of Trials

Unless otherwise stated, results are based on 10,000 independent trials. We verified that
increasing to 20,000 trials changes the mean and 95% confidence interval of total TCO by less than
1%, indicating statistical stability.

## ✅ Convergence Validation

We validate convergence using three checks:

1. Running mean stabilization (change < 1% over final 2,000 samples)
2. Stabilization of 5th/95th percentile estimates
3. Bootstrap confidence intervals over batches of 1,000 samples

All reported figures use the full converged sample set.

## 📊 Outputs

For each policy (e.g., aggressive vs. delayed refresh), our framework reports:

1. Expected lifecycle TCO
2. Variance and 95% confidence intervals
3. Probability that one policy outperforms another
4. Sensitivity (Sobol-style first-order effects via regression-based decomposition)

By modeling full distributions instead of point estimates, our approach provides distributional
robustness and explicitly quantifies the option value associated with flexible hardware refresh
timing under uncertainty.

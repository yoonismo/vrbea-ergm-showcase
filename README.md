# vrbea-ergm-showcase

Monte Carlo study of two standard ERGM estimators, the maximum pseudo-likelihood
estimator (MPLE) and MCMC-based maximum likelihood (MCMC-MLE), run from Python.
Networks are simulated and estimated with the R package `ergm` through `rpy2`;
replications run in parallel with `joblib`, and results are written to CSV as
each replication finishes.

## What it does

1. **Simulate networks** (`ergm_showcase.utils.graph_generator`): `nsims` networks
   from an ERGM with `edges + triangles`, drawn from one MCMC chain with
   `MCMC.burnin = 1e7` and `MCMC.interval = 1e4`, with a fixed seed.
2. **Estimate each network** (`ergm_showcase.utils.ergm_estimator`):
   - MPLE: `ergm(..., estimate = "MPLE")`
   - MCMC-MLE: `ergm(..., estimate = "MLE")` with
     `main.method = "Stochastic-Approximation"`, `MCMC.burnin = 1e5`,
     `MCMC.interval = 1e3`
   - Starting value for MCMC-MLE: truth + U[-perturb, perturb], drawn with seed
     `seed_num + g` for replication `g`.
3. **Run in parallel** (`experiments/run_simulations.py`): one worker process per
   replication, each with its own embedded R session.
4. **Summarize** (`experiments/summarize_showcase.py`): bias, mean, median, SE,
   5% and 95% quantiles of each estimator, and computation time.

## Parameterization

`param` in `config.py` is θ = (θ_edges, θ_triangles). The coefficients passed to
ergm are θ × (2, 1/n), and estimates are divided by the same factors before
being saved.

## Repository structure

    experiments/
      config.py               simulation settings
      run_simulations.py      parallel Monte Carlo driver
      summarize_showcase.py   summary table (printed and saved as CSV)
    ergm_showcase/
      utils.py                network generation (graph_generator), estimation
                              (ergm_estimator), summary statistics (summary_stat)
      gradients.py            custom autograd functions (torch.autograd.Function)
                              for a mean-field ERGM objective; not used by this
                              simulation
      __init__.py
    docs/
      R_packages.md           installing the R packages for rpy2, incl. on HPC

## Requirements

- Python: numpy, pandas, torch, joblib, rpy2
- R: network, sna, ergm

The R packages are called through rpy2. Install them from the same environment
you run the code in, and build them from source. On an HPC cluster where RStudio
and Jupyter run in different containers, give each its own R library directory;
otherwise packages compiled in one container may fail to load in the other
(e.g. `GLIBC_2.38 not found`). See [docs/R_packages.md](docs/R_packages.md).

## Usage

Run from the repository root.

    python -m experiments.run_simulations
    python -m experiments.summarize_showcase              # newest result file
    python -m experiments.summarize_showcase path/to.csv  # a specific file

Settings are in `experiments/config.py` (`n_obs`, `nsims`, `seed_num`, `param`,
`active`, `perturb`). Set `n_jobs` in `run_simulations.py` to the number of CPUs
allocated to the job.

## Output

`results/<date>_obs<n>_MPLE_MCMCMLE_<terms>_<param>_perturb<perturb>.csv`,
one row per replication:

| column | content |
|---|---|
| `init` | MCMC-MLE starting value |
| `theta_mple`, `time_mple` | MPLE estimate and seconds |
| `theta_mle`, `time_mle` | MCMC-MLE estimate and seconds |

## Acknowledgements and citation

I am deeply grateful to Angelo Mele and Lingjiong Zhu, whose work on approximate
variational estimation of network formation models laid the foundation for this
project. I thank them sincerely for making their research and code publicly
available, and I hold their contributions in the highest regard.

Uses the R package ergm.
Based on 'statnet' project software (statnet.org).
For license and citation information see statnet.org/attribution.

- Hunter, Handcock, Butts, Goodreau and Morris (2008). ergm: A Package to Fit,
  Simulate and Diagnose Exponential-Family Models for Networks. *Journal of
  Statistical Software* 24(3).
- Krivitsky, Hunter, Morris and Klumb (2023). ergm 4: New Features for Analyzing
  Exponential-Family Random Graph Models. *Journal of Statistical Software* 105(6).
- Mele and Zhu (2023). Approximate Variational Estimation for a Model of Network
  Formation. *Review of Economics and Statistics* 105(1).

## Author

Yoon Choi, Department of Economics, University of Washington

## License

Copyright (c) 2026 Yoon Choi. All rights reserved.
This repository is shared for viewing. No permission is granted to copy,
modify or distribute the code. For replication, please contact the author.

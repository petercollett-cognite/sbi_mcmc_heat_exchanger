# Fast Bayesian Inference for Heat Exchanger Condition Monitoring

Companion code for the paper: *"Fast Bayesian inference for equipment condition monitoring via simulation-based inference: applications to heat exchangers health"*

## Overview

This repository provides a probabilistic framework for diagnosing failure modes in shell-and-tube heat exchangers using both traditional MCMC sampling and amortized Simulation-Based Inference (SBI). The model identifies the onset and type of degradation — progressive fouling or internal leakage — from noisy sensor measurements (temperatures, mass flow rates).

## Repository Structure

```
sbi_mcmc_heat_exchanger/
├── src/hx_models/          # Core library
│   ├── heat_exchanger.py   # NumPyro physical model (NTU-effectiveness)
│   ├── inference.py        # MCMC and SBI inference utilities
│   ├── metrics.py          # Comparison metrics (CRPS, Wasserstein, coverage)
│   ├── plotting.py         # Publication-quality plotting functions
│   └── style.py            # Consistent plot styling and paper notation
├── notebooks/
│   ├── 01_model_demonstration.ipynb   # Physical model behavior
│   ├── 02_data_generation.ipynb       # Generate scenario observations
│   ├── 03_prior_comparison.ipynb      # Dirichlet vs Categorical4 priors
│   ├── 04_multi_sample_study.ipynb    # Main MCMC vs SBI comparison
│   └── 05_resource_analysis.ipynb     # Computational cost analysis
├── data/                   # Observation data
├── results/                # Inference results (.npz via Git LFS; see below)
└── figures/                # Publication-ready figures (tracked in git)
```

## Notebook Execution Order

Notebooks are numbered by recommended execution order. Dependencies:

1. **01_model_demonstration** — standalone, no prerequisites
2. **02_data_generation** — generates `data/observations.npz` (or loads cached)
3. **03_prior_comparison** — standalone (generates its own observations for the comparison)
4. **04_multi_sample_study** — requires `data/observations.npz` from notebook 02 and `results/mcmc_posteriors.npz` (see below). Generates `results/sbi_posteriors.npz` on first run.
5. **05_resource_analysis** — requires outputs from notebook 04

## Data Files

| File | How it is stored |
|------|------------------|
| **`data/observations.npz`** | Normal git (~7 MB). Notebook 02 can regenerate if needed. |
| **`results/mcmc_posteriors.npz`** | **Git LFS** (~400 MB). All MCMC repeats in one archive. |
| **`results/sbi_posteriors.npz`** | **Git LFS** (~800 MB+). Produced by notebook 04 (or commit your trained run). |
| **`results/metrics_summary.json`** | Normal git (small). |


**Cloners need `git lfs install` on their machine**; `git clone` then pulls LFS files automatically.

## Installation

```bash
# Using Poetry
poetry install

# Or using pip
pip install -e .
```

## Citation

If you use this code, please cite our paper:

```
@article{stasik2026fast,
  title={Fast Bayesian inference for equipment condition monitoring via simulation-based inference: applications to heat exchangers health},
  author={Stasik, Alexander Johannes and Casolo, Simone and Collett, Peter and Riemer-S{\o}rensen, Signe},
  year={2026}
}
```

# cosmo-bh-emulator

A PCA+Gaussian Process emulator for black hole physics in cosmological simulations, with MCMC-based parameter inference.

## What it does

Trains a fast emulator on a suite of 49 cosmological simulations that vary 5 black hole feedback parameters. The emulator predicts three global galaxy–halo scaling relations as a function of parameter inputs, enabling inference in seconds rather than hours per simulation.

Pipeline:
1. Compress each relation with PCA (k components chosen by explained variance + LOO RMSE criteria)
2. Fit one GP per PC score as a function of the 5 input parameters
3. Validate with leave-one-out cross-validation
4. Run MCMC inference using the emulator likelihood

## Requirements

Python 3.9+ with:

```bash
pip install numpy pandas scikit-learn scipy emcee matplotlib corner streamlit
```

## Usage

Run scripts in order from the repo root:

```bash
python scripts/00_load_data.py       # load + save results/data.npz
python scripts/01_sanity_plots.py    # visual data checks
python scripts/02_pca_validation.py  # choose k, save results/pca_k.json
python scripts/03_emulator.py        # fit + validate emulator
python scripts/04_oneparam_plots.py  # 1-parameter variation plots
python scripts/05_mcmc.py            # MCMC demo on 3 mock observations
```

Intermediate results are cached in `results/` and figures in `figures/`.

## Dataset

`param_variations/` contains 50 simulation runs (run 27 missing/crashed → 49 usable), sampled via Latin hypercube over 5 parameters:

| Parameter | Range |
|---|---|
| BlackHoleFeedbackFactor | 0.03 – 0.3 |
| BHTorqueLimitedAccretionBondiAccretionFactor | 0.1 – 10 |
| BHTorqueLimitedAccretionNormalizationFactor | 0.1 – 10 |
| QuasarThreshold | 0.0006 – 0.006 |
| RadioFeedbackMinDensityFactor | 0.002 – 0.05 |

Each run folder contains:
- `shmr.txt` — stellar-to-halo mass ratio vs halo mass (20 bins)
- `gasfractions.txt` — gas fraction vs halo mass (20 bins)
- `bhstellar.txt` — BH mass vs stellar mass (20 bins)
- `shmr_1.txt` … `shmr_64.txt` — jackknife resamples
- `STHM_1.txt` … `STHM_8.txt` — reduced-volume subcube resamples

Parameter sampling is in `param_variations/param_samples.csv` (semicolon-delimited, row order matches folder indices).

"""
Load cosmological simulation data from param_variations/ directory.
Outputs results/data.npz with arrays for all three relations and parameter matrix.
"""

import numpy as np
import pandas as pd
import json
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "param_variations"
RESULTS_DIR = Path(__file__).parent.parent / "results"

N_BINS = 20
SKIP_RUN = 27


def get_param_names():
    """Return list of parameter column names in order."""
    csv_path = DATA_DIR / "param_samples.csv"
    df = pd.read_csv(csv_path, sep=";")
    return list(df.columns)


def load_relation(folder, filename):
    """Load a 2-column relation file; return (x_axis, y_values)."""
    path = folder / filename
    data = np.loadtxt(path)
    return data[:, 0], data[:, 1]


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Load parameter samples
    csv_path = DATA_DIR / "param_samples.csv"
    params_df = pd.read_csv(csv_path, sep=";")
    print(f"Parameter columns: {list(params_df.columns)}")
    print(f"Total runs in CSV: {len(params_df)}")

    # Collect valid run indices (skip run 27)
    all_indices = list(range(len(params_df)))
    valid_indices = [i for i in all_indices if i != SKIP_RUN]
    print(f"Skipping run {SKIP_RUN}. Using {len(valid_indices)} runs.")

    N = len(valid_indices)

    # Initialize arrays
    shmr_data = np.full((N, N_BINS), np.nan)
    gasfrac_data = np.full((N, N_BINS), np.nan)
    bhstellar_data = np.full((N, N_BINS), np.nan)
    shmr_x = None
    gasfrac_x = None
    bhstellar_x = None
    theta = np.zeros((N, 5))

    for arr_idx, run_idx in enumerate(valid_indices):
        folder = DATA_DIR / f"param_variation_{run_idx}"
        if not folder.exists():
            print(f"  WARNING: folder {folder} not found, filling with NaN")
            continue

        # Load SHMR
        x, y = load_relation(folder, "shmr.txt")
        if shmr_x is None:
            shmr_x = x
        shmr_data[arr_idx] = y

        # Load gas fractions
        x, y = load_relation(folder, "gasfractions.txt")
        if gasfrac_x is None:
            gasfrac_x = x
        gasfrac_data[arr_idx] = y

        # Load BH-stellar
        x, y = load_relation(folder, "bhstellar.txt")
        if bhstellar_x is None:
            bhstellar_x = x
        bhstellar_data[arr_idx] = y

        # Parameter vector
        theta[arr_idx] = params_df.iloc[run_idx].values

    # Log10 transform x-axes (masses) and BH/stellar masses
    shmr_x_log = np.log10(shmr_x)
    gasfrac_x_log = np.log10(gasfrac_x)
    bhstellar_x_log = np.log10(bhstellar_x)
    bhstellar_data_log = np.log10(np.where(bhstellar_data > 0, bhstellar_data, np.nan))

    # NaN summary
    print("\nNaN summary:")
    print(f"  SHMR:      {np.sum(np.isnan(shmr_data))} NaN values")
    print(f"  GasFrac:   {np.sum(np.isnan(gasfrac_data))} NaN values")
    print(f"  BHStellar: {np.sum(np.isnan(bhstellar_data_log))} NaN values")

    nan_bins_shmr = np.where(np.any(np.isnan(shmr_data), axis=0))[0]
    nan_bins_gas = np.where(np.any(np.isnan(gasfrac_data), axis=0))[0]
    nan_bins_bh = np.where(np.any(np.isnan(bhstellar_data_log), axis=0))[0]
    if len(nan_bins_shmr): print(f"  SHMR NaN bins: {nan_bins_shmr}")
    if len(nan_bins_gas): print(f"  GasFrac NaN bins: {nan_bins_gas}")
    if len(nan_bins_bh): print(f"  BHStellar NaN bins: {nan_bins_bh}")

    # Save
    np.savez(
        RESULTS_DIR / "data.npz",
        shmr_data=shmr_data,
        gasfrac_data=gasfrac_data,
        bhstellar_data=bhstellar_data_log,  # log10 transformed
        shmr_x=shmr_x_log,
        gasfrac_x=gasfrac_x_log,
        bhstellar_x=bhstellar_x_log,
        theta=theta,
        param_names=np.array(get_param_names()),
        valid_indices=np.array(valid_indices),
    )

    print("\nArray shapes:")
    print(f"  theta:         {theta.shape}")
    print(f"  shmr_data:     {shmr_data.shape}")
    print(f"  gasfrac_data:  {gasfrac_data.shape}")
    print(f"  bhstellar_data:{bhstellar_data_log.shape}")
    print(f"  shmr_x:        {shmr_x_log.shape}")
    print(f"  gasfrac_x:     {gasfrac_x_log.shape}")
    print(f"  bhstellar_x:   {bhstellar_x_log.shape}")
    print("\nSaved to results/data.npz")


if __name__ == "__main__":
    main()

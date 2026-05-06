"""
PCA+GP emulator for each cosmological relation.
Fits PCA on full dataset, then fits one GP per PC component.
LOO cross-validation for emulator quality assessment.
"""

import numpy as np
import json
import pickle
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "results"
FIG_DIR = Path(__file__).parent.parent / "figures" / "emulator"

plt.style.use("seaborn-v0_8-whitegrid")

RELATIONS = ["shmr", "gasfrac", "bhstellar"]
DATA_KEYS = {
    "shmr": "shmr_data",
    "gasfrac": "gasfrac_data",
    "bhstellar": "bhstellar_data",
}

PRIOR_RANGES = np.array([
    [0.03, 0.3],
    [0.1, 10.0],
    [0.1, 10.0],
    [0.0006, 0.006],
    [0.002, 0.05],
])


def normalize_theta(theta):
    """Normalize parameter vectors to [0, 1] using prior ranges."""
    return (theta - PRIOR_RANGES[:, 0]) / (PRIOR_RANGES[:, 1] - PRIOR_RANGES[:, 0])


def get_valid_mask(data):
    return ~np.any(np.isnan(data), axis=0)


def build_kernel():
    return RBF(length_scale_bounds=(1e-2, 1e2)) + WhiteKernel()


def fit_emulator(data_valid, theta_norm, k):
    """Fit PCA + k GPs. Returns (pca, gp_list)."""
    pca = PCA(n_components=k)
    scores = pca.fit_transform(data_valid)  # [N, k]
    gps = []
    for j in range(k):
        print(f"    Fitting GP for PC {j+1}/{k}...", end="\r")
        gp = GaussianProcessRegressor(
            kernel=build_kernel(),
            normalize_y=True,
            n_restarts_optimizer=5,
        )
        gp.fit(theta_norm, scores[:, j])
        gps.append(gp)
    print()
    return pca, gps


def predict_emulator(pca, gps, theta_norm_single):
    """Predict relation (mean + variance per bin) for one parameter vector."""
    k = len(gps)
    mu_z = np.zeros(k)
    var_z = np.zeros(k)
    for j, gp in enumerate(gps):
        m, s = gp.predict(theta_norm_single.reshape(1, -1), return_std=True)
        mu_z[j] = m[0]
        var_z[j] = s[0] ** 2
    mu_rel = pca.inverse_transform(mu_z.reshape(1, -1))[0]
    # Variance propagation: sum_j (var_z[j] * components_[j]^2)
    var_rel = np.sum(var_z[:, None] * pca.components_ ** 2, axis=0)
    return mu_rel, var_rel


def loo_cv(data_valid, theta_norm, k, diag_var):
    """LOO cross-validation. Returns arrays of rmse, chi2, r2 per sim."""
    N = data_valid.shape[0]
    rmse_list, chi2_list, r2_list = [], [], []

    for i in range(N):
        print(f"    LOO sim {i+1}/{N}...", end="\r")
        train_data = np.delete(data_valid, i, axis=0)
        train_theta = np.delete(theta_norm, i, axis=0)

        pca_loo, gps_loo = fit_emulator(train_data, train_theta, k)
        # Suppress inner print
        mu_pred, var_pred = predict_emulator(pca_loo, gps_loo, theta_norm[i])
        truth = data_valid[i]

        rmse = np.sqrt(np.mean((mu_pred - truth) ** 2))
        chi2 = np.sum((mu_pred - truth) ** 2 / (diag_var + 1e-12))
        ss_res = np.sum((mu_pred - truth) ** 2)
        ss_tot = np.sum((truth - np.mean(truth)) ** 2)
        r2 = 1.0 - ss_res / (ss_tot + 1e-12)

        rmse_list.append(rmse)
        chi2_list.append(chi2)
        r2_list.append(r2)

    print()
    return np.array(rmse_list), np.array(chi2_list), np.array(r2_list)


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    npz = np.load(DATA_DIR / "data.npz", allow_pickle=True)
    with open(DATA_DIR / "pca_k.json") as f:
        pca_k = json.load(f)

    theta = npz["theta"]
    theta_norm = normalize_theta(theta)
    summary_lines = []

    print("=== EMULATOR FITTING ===\n")

    for rel in RELATIONS:
        data = npz[DATA_KEYS[rel]]
        valid_mask = get_valid_mask(data)
        data_valid = data[:, valid_mask]
        k = pca_k[rel]
        diag_var = np.var(data_valid, axis=0)

        print(f"\n{rel}: k={k}, {data_valid.shape[0]} sims, {data_valid.shape[1]} bins")

        # 3a: Fit emulator on full dataset
        print(f"  Fitting full emulator...")
        pca, gps = fit_emulator(data_valid, theta_norm, k)

        # Save
        emulator = {
            "pca": pca,
            "gps": gps,
            "valid_mask": valid_mask,
            "k": k,
            "relation": rel,
        }
        out_pkl = DATA_DIR / f"emulator_{rel}.pkl"
        with open(out_pkl, "wb") as f:
            pickle.dump(emulator, f)
        print(f"  Saved to {out_pkl}")

        # 3b: LOO cross-validation
        print(f"  Running LOO cross-validation...")
        rmse_arr, chi2_arr, r2_arr = loo_cv(data_valid, theta_norm, k, diag_var)

        med_rmse = float(np.median(rmse_arr))
        med_chi2 = float(np.median(chi2_arr))
        med_r2 = float(np.median(r2_arr))
        summary_lines.append(
            f"  {rel:12s}: median RMSE={med_rmse:.4f}, median chi2={med_chi2:.2f}, median R2={med_r2:.3f}"
        )

        # 3c: Predicted vs. true scatter
        # Collect all LOO predictions
        N = data_valid.shape[0]
        all_pred = np.zeros_like(data_valid)
        for i in range(N):
            train_data = np.delete(data_valid, i, axis=0)
            train_theta = np.delete(theta_norm, i, axis=0)
            pca_loo, gps_loo = fit_emulator(train_data, train_theta, k)
            mu_pred, _ = predict_emulator(pca_loo, gps_loo, theta_norm[i])
            all_pred[i] = mu_pred
            print(f"    Collecting predictions {i+1}/{N}...", end="\r")
        print()

        fig, ax = plt.subplots(figsize=(6, 6))
        cmap = plt.cm.viridis
        for i in range(N):
            color = cmap(i / N)
            ax.scatter(data_valid[i], all_pred[i], c=[color], s=5, alpha=0.4)
        vmin = min(data_valid.min(), all_pred.min())
        vmax = max(data_valid.max(), all_pred.max())
        ax.plot([vmin, vmax], [vmin, vmax], "k--", lw=1, label="1:1")
        ax.set_xlabel("True")
        ax.set_ylabel("Predicted")
        ax.set_title(f"{rel} — LOO Predicted vs True\nR²={med_r2:.3f}, median RMSE={med_rmse:.4f}")
        ax.legend()
        fig.tight_layout()
        fig.savefig(FIG_DIR / f"{rel}_predicted_vs_true.png", dpi=150)
        plt.close(fig)

        # LOO residuals
        residuals = all_pred - data_valid
        x = np.arange(data_valid.shape[1])
        fig, ax = plt.subplots(figsize=(8, 4))
        for i in range(N):
            ax.plot(x, residuals[i], color="steelblue", alpha=0.2, lw=0.7)
        ax.plot(x, np.median(residuals, axis=0), color="red", lw=2, label="Median")
        ax.axhline(0, color="black", lw=0.8)
        ax.set_xlabel("Mass bin index")
        ax.set_ylabel("Residual (pred - true)")
        ax.set_title(f"{rel} — LOO Residuals")
        ax.legend()
        fig.tight_layout()
        fig.savefig(FIG_DIR / f"{rel}_loo_residuals.png", dpi=150)
        plt.close(fig)

    print("\n=== EMULATOR VALIDATION SUMMARY ===")
    for line in summary_lines:
        print(line)
    print("\nStep 3 complete.")


if __name__ == "__main__":
    main()

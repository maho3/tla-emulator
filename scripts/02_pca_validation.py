"""
PCA validation for each cosmological relation.
Chooses number of PCA components (k) and saves to results/pca_k.json.
"""

import numpy as np
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "results"
FIG_DIR = Path(__file__).parent.parent / "figures" / "pca"

plt.style.use("seaborn-v0_8-whitegrid")

RELATIONS = ["shmr", "gasfrac", "bhstellar"]
DATA_KEYS = {
    "shmr": ("shmr_data", "shmr_x"),
    "gasfrac": ("gasfrac_data", "gasfrac_x"),
    "bhstellar": ("bhstellar_data", "bhstellar_x"),
}


def get_valid_bins(data):
    """Return mask of bins that are not NaN in any simulation."""
    return ~np.any(np.isnan(data), axis=0)


def fit_pca_full(data_valid):
    """Fit PCA on data_valid [N, M_valid] with all components."""
    pca = PCA(n_components=None)
    pca.fit(data_valid)
    return pca


def loo_reconstruction_error(data_valid, k):
    """Leave-one-out RMSE for k-component PCA reconstruction."""
    N = data_valid.shape[0]
    rmse_list = []
    for i in range(N):
        train = np.delete(data_valid, i, axis=0)
        pca = PCA(n_components=k)
        pca.fit(train)
        score_i = pca.transform(data_valid[i:i+1])
        recon_i = pca.inverse_transform(score_i)[0]
        rmse = np.sqrt(np.mean((recon_i - data_valid[i]) ** 2))
        rmse_list.append(rmse)
    return np.mean(rmse_list)


def choose_k(pca, data_valid, max_k=10):
    """Choose k satisfying >=95% variance AND LOO RMSE < 1% of data std."""
    cum_var = np.cumsum(pca.explained_variance_ratio_)
    data_std = np.std(data_valid)
    rmse_threshold = 0.05 * data_std

    # Compute LOO RMSE for k=1..max_k
    loo_rmse = []
    for k in range(1, max_k + 1):
        print(f"    LOO k={k}/{max_k}...", end="\r")
        rmse = loo_reconstruction_error(data_valid, k)
        loo_rmse.append(rmse)
    print()

    # k_var: first k reaching >=95% explained variance
    k_var = next((k+1 for k, cv in enumerate(cum_var) if cv >= 0.95), max_k)

    # k_abs: first k where LOO RMSE drops below 1% of data std
    k_abs = max_k
    for k in range(1, max_k + 1):
        if loo_rmse[k - 1] < rmse_threshold:
            k_abs = k
            break

    print(f"    data std={data_std:.4f}, threshold (5%)={rmse_threshold:.4f}")
    print(f"    k_var={k_var}, k_abs={k_abs}")

    k_chosen = max(k_var, k_abs)
    k_chosen = min(k_chosen, max_k)
    return k_chosen, loo_rmse, cum_var


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    npz = np.load(DATA_DIR / "data.npz", allow_pickle=True)
    pca_k = {}
    summary_lines = []

    print("=== PCA VALIDATION ===\n")

    for rel in RELATIONS:
        data_key, x_key = DATA_KEYS[rel]
        data = npz[data_key]
        x = npz[x_key]

        # Mask out NaN bins
        valid_mask = get_valid_bins(data)
        data_valid = data[:, valid_mask]
        x_valid = x[valid_mask]
        print(f"{rel}: {data_valid.shape[0]} sims, {data_valid.shape[1]} valid bins")

        # 2a: Full PCA
        pca = fit_pca_full(data_valid)
        n_comp = len(pca.explained_variance_ratio_)

        # 2b: Explained variance plot
        cum_var = np.cumsum(pca.explained_variance_ratio_)
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(np.arange(1, n_comp+1), cum_var * 100, "o-", color="steelblue")
        for thresh, ls in [(90, "--"), (95, "-"), (99, ":")]:
            ax.axhline(thresh, color="gray", linestyle=ls, label=f"{thresh}%")
        ax.set_xlabel("Number of PCA components")
        ax.set_ylabel("Cumulative explained variance [%]")
        ax.set_title(f"{rel} — Explained Variance")
        ax.legend()
        ax.set_xlim(0.5, min(n_comp, 20) + 0.5)
        ax.set_ylim(0, 105)
        fig.tight_layout()
        fig.savefig(FIG_DIR / f"{rel}_explained_variance.png", dpi=150)
        plt.close(fig)

        # 2c: PC mode vectors (first 4)
        n_modes = min(4, n_comp)
        fig, axes = plt.subplots(1, n_modes, figsize=(4 * n_modes, 4), sharey=False)
        if n_modes == 1:
            axes = [axes]
        for j, ax in enumerate(axes):
            ax.plot(x_valid, pca.components_[j], color=f"C{j}", lw=1.5)
            ax.axhline(0, color="gray", lw=0.5)
            ax.set_title(f"PC {j+1} ({pca.explained_variance_ratio_[j]*100:.1f}%)")
            ax.set_xlabel("Mass [log10]")
        fig.suptitle(f"{rel} — PC Mode Vectors", fontsize=12)
        fig.tight_layout()
        fig.savefig(FIG_DIR / f"{rel}_pc_modes.png", dpi=150)
        plt.close(fig)

        # 2d: LOO reconstruction error vs k
        print(f"  Computing LOO for {rel}...")
        k_chosen, loo_rmse, cum_var_arr = choose_k(pca, data_valid, max_k=min(10, data_valid.shape[1]))

        data_std = np.std(data_valid)
        rmse_threshold = 0.05 * data_std
        ks = np.arange(1, len(loo_rmse) + 1)
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(ks, loo_rmse, "s-", color="darkorange")
        ax.axhline(rmse_threshold, color="steelblue", linestyle="--",
                   label=f"5% data std = {rmse_threshold:.4f}")
        ax.axvline(k_chosen, color="red", linestyle="--", label=f"Chosen k={k_chosen}")
        ax.set_xlabel("Number of PCA components (k)")
        ax.set_ylabel("Mean LOO RMSE")
        ax.set_title(f"{rel} — LOO Reconstruction Error")
        ax.legend()
        fig.tight_layout()
        fig.savefig(FIG_DIR / f"{rel}_loo_reconstruction.png", dpi=150)
        plt.close(fig)

        explained_at_k = float(cum_var_arr[k_chosen - 1]) * 100
        loo_at_k = loo_rmse[k_chosen - 1]
        pca_k[rel] = int(k_chosen)

        if k_chosen > 6:
            print(f"  WARNING: {rel} requires k={k_chosen} > 6. PCA may be a poor description!")

        summary_lines.append(
            f"  {rel:12s}: k={k_chosen}, explained variance={explained_at_k:.1f}%, LOO RMSE={loo_at_k:.4f}"
        )

    print("\n=== PCA VALIDATION SUMMARY ===")
    for line in summary_lines:
        print(line)

    out_path = DATA_DIR / "pca_k.json"
    with open(out_path, "w") as f:
        json.dump(pca_k, f, indent=2)
    print(f"\nSaved k values to {out_path}")
    print("Step 2 complete.")


if __name__ == "__main__":
    main()

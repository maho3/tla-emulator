"""
One-parameter variation plots showing how each parameter affects each relation.
Fixes other parameters at their median values and sweeps the target parameter.
Also generates textual descriptions and saves to results/param_descriptions.json.
"""

import numpy as np
import json
import pickle
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "results"
FIG_DIR = Path(__file__).parent.parent / "figures" / "oneparam"
FIG_DIR_UNC = Path(__file__).parent.parent / "figures" / "oneparam_uncertainty"

plt.style.use("seaborn-v0_8-whitegrid")

PRIOR_RANGES = np.array([
    [0.03, 0.3],
    [0.1, 10.0],
    [0.1, 10.0],
    [0.0006, 0.006],
    [0.002, 0.05],
])

# Log-spaced parameters (those spanning orders of magnitude)
LOG_SPACED_PARAMS = {1, 2, 3, 4}  # indices into PRIOR_RANGES

RELATIONS = ["shmr", "gasfrac", "bhstellar"]
REL_LABELS = {
    "shmr": {"xlabel": "Halo Mass [log10]", "ylabel": "M*/M_halo", "title": "SHMR"},
    "gasfrac": {"xlabel": "Halo Mass [log10]", "ylabel": "Gas Fraction", "title": "Gas Fractions"},
    "bhstellar": {"xlabel": "Stellar Mass [log10]", "ylabel": "BH Mass [log10]", "title": "BH-Stellar"},
}
DATA_KEYS = {
    "shmr": ("shmr_data", "shmr_x"),
    "gasfrac": ("gasfrac_data", "gasfrac_x"),
    "bhstellar": ("bhstellar_data", "bhstellar_x"),
}


def normalize_theta(theta):
    return (theta - PRIOR_RANGES[:, 0]) / (PRIOR_RANGES[:, 1] - PRIOR_RANGES[:, 0])


def predict_emulator(emulator, theta_norm_single):
    """Predict relation mean and variance for one normalized parameter vector."""
    pca = emulator["pca"]
    gps = emulator["gps"]
    k = emulator["k"]

    mu_z = np.zeros(k)
    var_z = np.zeros(k)
    for j, gp in enumerate(gps):
        m, s = gp.predict(theta_norm_single.reshape(1, -1), return_std=True)
        mu_z[j] = m[0]
        var_z[j] = s[0] ** 2

    mu_rel = pca.inverse_transform(mu_z.reshape(1, -1))[0]
    var_rel = np.sum(var_z[:, None] * pca.components_ ** 2, axis=0)
    return mu_rel, var_rel


def describe_effect(param_name, rel_name, sweep_vals, mean_curves, x_vals):
    """
    Generate a textual description of the effect of a parameter on a relation.
    Based on analysis of how mean_curves change with sweep_vals.
    """
    n_sweep = len(sweep_vals)
    low_curve = mean_curves[0]
    high_curve = mean_curves[-1]
    diff = high_curve - low_curve
    max_diff = np.max(np.abs(diff))
    mean_diff = np.mean(diff)
    max_bin = np.argmax(np.abs(diff))

    if max_diff < 0.005:
        effect_strength = "negligible"
    elif max_diff < 0.05:
        effect_strength = "weak"
    elif max_diff < 0.15:
        effect_strength = "moderate"
    else:
        effect_strength = "strong"

    direction = "increases" if mean_diff > 0 else "decreases"
    mass_loc = "low-mass" if max_bin < n_sweep // 3 else ("high-mass" if max_bin > 2 * n_sweep // 3 else "intermediate-mass")

    if rel_name == "shmr":
        desc = (
            f"Increasing {param_name} {direction} the stellar-to-halo mass ratio with "
            f"{effect_strength} effect, most pronounced at {mass_loc} scales (bin {max_bin}). "
            f"Peak deviation between lowest and highest parameter value is {max_diff:.3f} dex."
        )
    elif rel_name == "gasfrac":
        desc = (
            f"Increasing {param_name} {direction} the gas fraction with {effect_strength} effect, "
            f"strongest at {mass_loc} halo masses (bin {max_bin}). "
            f"The change reaches {max_diff:.3f} between extreme parameter values."
        )
    else:  # bhstellar
        desc = (
            f"Increasing {param_name} {direction} the BH mass at fixed stellar mass with "
            f"{effect_strength} effect, most significant at {mass_loc} galaxy masses (bin {max_bin}). "
            f"The peak log10 BH mass shift is {max_diff:.2f} dex across the parameter range."
        )
    return desc


def plot_uncertainty_7pt(p_idx, p_name, sweep_vals_7, theta_median, emulators, npz, param_descriptions):
    """Plot 7 sweep points, each with its own ±1σ uncertainty band."""
    cmap = plt.cm.viridis
    norm_color = plt.Normalize(sweep_vals_7.min(), sweep_vals_7.max())

    fig, axes = plt.subplots(3, 1, figsize=(8, 12))

    for rel_idx, rel in enumerate(RELATIONS):
        emulator = emulators[rel]
        valid_mask = emulator["valid_mask"]
        data_key, x_key = DATA_KEYS[rel]
        x = npz[x_key][valid_mask]
        meta = REL_LABELS[rel]
        ax = axes[rel_idx]

        for val in sweep_vals_7:
            theta_test = theta_median.copy()
            theta_test[p_idx] = val
            theta_norm = normalize_theta(theta_test)
            mu_rel, var_rel = predict_emulator(emulator, theta_norm)
            sigma_rel = np.sqrt(var_rel)
            color = cmap(norm_color(val))
            ax.plot(x, mu_rel, color=color, lw=1.5, alpha=0.9)
            ax.fill_between(x, mu_rel - sigma_rel, mu_rel + sigma_rel,
                            color=color, alpha=0.25)

        ax.set_xlabel(meta["xlabel"], fontsize=10)
        ax.set_ylabel(meta["ylabel"], fontsize=10)
        ax.set_title(meta["title"], fontsize=10)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm_color)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes, fraction=0.02, pad=0.04)
    cbar.set_label(p_name, fontsize=11)

    fig.suptitle(f"1P Variation (±1σ per curve): {p_name}", fontsize=13)
    fig.tight_layout(rect=[0, 0, 0.95, 1])
    fname = FIG_DIR_UNC / f"vary_{p_name}_uncertainty.png"
    fig.savefig(fname, dpi=150)
    plt.close(fig)
    print(f"  Saved {fname}")


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR_UNC.mkdir(parents=True, exist_ok=True)

    npz = np.load(DATA_DIR / "data.npz", allow_pickle=True)
    theta = npz["theta"]
    param_names = list(npz["param_names"])

    # Load emulators
    emulators = {}
    for rel in RELATIONS:
        with open(DATA_DIR / f"emulator_{rel}.pkl", "rb") as f:
            emulators[rel] = pickle.load(f)

    # Median parameter vector (in original space)
    theta_median = np.median(theta, axis=0)
    print(f"Median parameter vector: {dict(zip(param_names, theta_median))}")

    N_SWEEP = 50
    param_descriptions = {p: {} for p in param_names}

    for p_idx, p_name in enumerate(param_names):
        print(f"\nVarying {p_name}...")
        lo, hi = PRIOR_RANGES[p_idx]

        if p_idx in LOG_SPACED_PARAMS:
            sweep_vals = np.logspace(np.log10(lo), np.log10(hi), N_SWEEP)
        else:
            sweep_vals = np.linspace(lo, hi, N_SWEEP)

        cmap = plt.cm.viridis
        norm_color = plt.Normalize(sweep_vals.min(), sweep_vals.max())

        fig, axes = plt.subplots(3, 1, figsize=(8, 12))

        for rel_idx, rel in enumerate(RELATIONS):
            emulator = emulators[rel]
            valid_mask = emulator["valid_mask"]
            data_key, x_key = DATA_KEYS[rel]
            x = npz[x_key][valid_mask]
            meta = REL_LABELS[rel]
            ax = axes[rel_idx]

            mean_curves = []
            for s_idx, val in enumerate(sweep_vals):
                theta_test = theta_median.copy()
                theta_test[p_idx] = val
                theta_norm = normalize_theta(theta_test)
                mu_rel, var_rel = predict_emulator(emulator, theta_norm)
                mean_curves.append(mu_rel)
                color = cmap(norm_color(val))
                ax.plot(x, mu_rel, color=color, alpha=0.7, lw=0.9)

            mean_curves = np.array(mean_curves)

            # ±1σ band for the median value
            theta_test = theta_median.copy()
            theta_norm = normalize_theta(theta_test)
            mu_mid, var_mid = predict_emulator(emulator, theta_norm)
            sigma_mid = np.sqrt(var_mid)
            ax.fill_between(x, mu_mid - sigma_mid, mu_mid + sigma_mid,
                            alpha=0.25, color="gray", label="±1σ (median θ)")

            ax.set_xlabel(meta["xlabel"], fontsize=10)
            ax.set_ylabel(meta["ylabel"], fontsize=10)
            ax.set_title(meta["title"], fontsize=10)
            ax.legend(fontsize=8)

            # Generate description
            desc = describe_effect(p_name, rel, sweep_vals, mean_curves, x)
            param_descriptions[p_name][rel] = desc

        # Colorbar
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm_color)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=axes, fraction=0.02, pad=0.04)
        cbar.set_label(p_name, fontsize=11)

        fig.suptitle(f"1P Variation: {p_name}", fontsize=13)
        fig.tight_layout(rect=[0, 0, 0.95, 1])
        fname = FIG_DIR / f"vary_{p_name}.png"
        fig.savefig(fname, dpi=150)
        plt.close(fig)
        print(f"  Saved {fname}")

        # 7-point uncertainty plot
        if p_idx in LOG_SPACED_PARAMS:
            sweep_vals_7 = np.logspace(np.log10(lo), np.log10(hi), 7)
        else:
            sweep_vals_7 = np.linspace(lo, hi, 7)
        plot_uncertainty_7pt(p_idx, p_name, sweep_vals_7, theta_median, emulators, npz, param_descriptions)

    # Save descriptions
    out_path = DATA_DIR / "param_descriptions.json"
    with open(out_path, "w") as f:
        json.dump(param_descriptions, f, indent=2)
    print(f"\nSaved param descriptions to {out_path}")
    print("Step 4 complete.")


if __name__ == "__main__":
    main()

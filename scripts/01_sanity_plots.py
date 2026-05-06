"""
Sanity plots for all three cosmological relations.
Produces 15 per-parameter colored plots + 1 overview panel.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "results"
FIG_DIR = Path(__file__).parent.parent / "figures" / "sanity"

plt.style.use("seaborn-v0_8-whitegrid")

RELATIONS = {
    "shmr": {
        "data_key": "shmr_data",
        "x_key": "shmr_x",
        "xlabel": "Halo Mass [log₁₀ M☉]",
        "ylabel": "M★/M_halo",
        "title": "SHMR",
    },
    "gasfrac": {
        "data_key": "gasfrac_data",
        "x_key": "gasfrac_x",
        "xlabel": "Halo Mass [log₁₀ M☉]",
        "ylabel": "Gas Fraction",
        "title": "Gas Fractions",
    },
    "bhstellar": {
        "data_key": "bhstellar_data",
        "x_key": "bhstellar_x",
        "xlabel": "Stellar Mass [log₁₀ M☉]",
        "ylabel": "BH Mass [log₁₀ M☉]",
        "title": "BH-Stellar",
    },
}


def plot_relation_colored(x, data, param_vals, param_name, relation_name, meta):
    """One plot: all simulation curves colored by one parameter value."""
    fig, ax = plt.subplots(figsize=(7, 5))
    cmap = plt.cm.viridis
    norm = plt.Normalize(param_vals.min(), param_vals.max())

    for i in range(data.shape[0]):
        ax.plot(x, data[i], color=cmap(norm(param_vals[i])), alpha=0.7, lw=1.2)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax)
    cbar.set_label(param_name, fontsize=10)

    ax.set_xlabel(meta["xlabel"], fontsize=11)
    ax.set_ylabel(meta["ylabel"], fontsize=11)
    ax.set_title(f"{meta['title']} colored by {param_name}", fontsize=11)
    fig.tight_layout()
    fname = FIG_DIR / f"{relation_name}_{param_name}.png"
    fig.savefig(fname, dpi=150)
    plt.close(fig)


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    npz = np.load(DATA_DIR / "data.npz", allow_pickle=True)
    theta = npz["theta"]
    param_names = list(npz["param_names"])

    print(f"Loaded {theta.shape[0]} simulations, {theta.shape[1]} parameters")

    # 1a: 15 colored plots
    for rel_key, meta in RELATIONS.items():
        data = npz[meta["data_key"]]
        x = npz[meta["x_key"]]
        for p_idx, p_name in enumerate(param_names):
            print(f"  Plotting {rel_key} colored by {p_name}...")
            plot_relation_colored(x, data, theta[:, p_idx], p_name, rel_key, meta)

    # 1b: Overview panel (3x1)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, (rel_key, meta) in zip(axes, RELATIONS.items()):
        data = npz[meta["data_key"]]
        x = npz[meta["x_key"]]
        for i in range(data.shape[0]):
            ax.plot(x, data[i], color="steelblue", alpha=0.4, lw=0.8)
        ax.set_xlabel(meta["xlabel"], fontsize=10)
        ax.set_ylabel(meta["ylabel"], fontsize=10)
        ax.set_title(meta["title"], fontsize=11)

    fig.suptitle("All simulations — spread overview", fontsize=13)
    fig.tight_layout()
    fname = FIG_DIR / "all_relations_overview.png"
    fig.savefig(fname, dpi=150)
    plt.close(fig)
    print(f"\nSaved overview to {fname}")
    print("Step 1 complete.")


if __name__ == "__main__":
    main()

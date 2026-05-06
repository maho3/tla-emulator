"""
Streamlit interactive emulator app.
Sliders for all 5 parameters; plots all 3 relations with ±1σ uncertainty bands.
Run: conda run -n ili-torch streamlit run scripts/07_streamlit_app.py
"""

import numpy as np
import pickle
import json
import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "results"

PRIOR_RANGES = np.array([
    [0.03, 0.3],
    [0.1, 10.0],
    [0.1, 10.0],
    [0.0006, 0.006],
    [0.002, 0.05],
])

LOG_SPACED_PARAMS = {1, 2, 3, 4}

RELATIONS = ["shmr", "gasfrac", "bhstellar"]
REL_LABELS = {
    "shmr": {"xlabel": "Halo Mass [log10 M_sun]", "ylabel": "M*/M_halo", "title": "SHMR"},
    "gasfrac": {"xlabel": "Halo Mass [log10 M_sun]", "ylabel": "Gas Fraction", "title": "Gas Fractions"},
    "bhstellar": {"xlabel": "Stellar Mass [log10 M_sun]", "ylabel": "BH Mass [log10 M_sun]", "title": "BH-Stellar"},
}
DATA_KEYS = {
    "shmr": ("shmr_data", "shmr_x"),
    "gasfrac": ("gasfrac_data", "gasfrac_x"),
    "bhstellar": ("bhstellar_data", "bhstellar_x"),
}


@st.cache_resource
def load_emulators():
    emulators = {}
    for rel in RELATIONS:
        with open(DATA_DIR / f"emulator_{rel}.pkl", "rb") as f:
            emulators[rel] = pickle.load(f)
    return emulators


@st.cache_resource
def load_data():
    return np.load(DATA_DIR / "data.npz", allow_pickle=True)


def normalize_theta(theta):
    return (theta - PRIOR_RANGES[:, 0]) / (PRIOR_RANGES[:, 1] - PRIOR_RANGES[:, 0])


def predict_emulator(emulator, theta_norm):
    pca = emulator["pca"]
    gps = emulator["gps"]
    k = emulator["k"]
    mu_z = np.zeros(k)
    var_z = np.zeros(k)
    for j, gp in enumerate(gps):
        m, s = gp.predict(theta_norm.reshape(1, -1), return_std=True)
        mu_z[j] = m[0]
        var_z[j] = s[0] ** 2
    mu_rel = pca.inverse_transform(mu_z.reshape(1, -1))[0]
    var_rel = np.sum(var_z[:, None] * pca.components_ ** 2, axis=0)
    return mu_rel, var_rel


def main():
    st.title("Cosmological Simulation Emulator")
    st.markdown("Interactive emulator: adjust the 5 BH physics parameters and see the predicted relations.")

    emulators = load_emulators()
    npz = load_data()
    param_names = list(npz["param_names"])
    theta_train = npz["theta"]

    # Sidebar sliders
    st.sidebar.header("Parameter Controls")
    theta_user = np.zeros(5)

    for p_idx, p_name in enumerate(param_names):
        lo, hi = PRIOR_RANGES[p_idx]
        default = float(np.median(theta_train[:, p_idx]))

        if p_idx in LOG_SPACED_PARAMS:
            log_lo, log_hi = np.log10(lo), np.log10(hi)
            log_val = st.sidebar.slider(
                p_name,
                min_value=float(log_lo),
                max_value=float(log_hi),
                value=float(np.log10(default)),
                step=float((log_hi - log_lo) / 100),
                format="%.3f (log10)",
                key=f"slider_{p_idx}",
            )
            theta_user[p_idx] = 10 ** log_val
        else:
            theta_user[p_idx] = st.sidebar.slider(
                p_name,
                min_value=float(lo),
                max_value=float(hi),
                value=float(default),
                step=float((hi - lo) / 100),
                format="%.4f",
                key=f"slider_{p_idx}",
            )

    theta_norm = normalize_theta(theta_user)

    # Main plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, rel in zip(axes, RELATIONS):
        emulator = emulators[rel]
        valid_mask = emulator["valid_mask"]
        data_key, x_key = DATA_KEYS[rel]
        x = npz[x_key][valid_mask]
        meta = REL_LABELS[rel]

        # Training data spread (faint)
        data_train = npz[data_key][:, valid_mask]
        for i in range(data_train.shape[0]):
            ax.plot(x, data_train[i], color="lightgray", alpha=0.4, lw=0.6)

        # Emulator prediction
        mu_rel, var_rel = predict_emulator(emulator, theta_norm)
        sigma_rel = np.sqrt(var_rel)
        ax.plot(x, mu_rel, color="steelblue", lw=2, label="Emulator mean")
        ax.fill_between(x, mu_rel - sigma_rel, mu_rel + sigma_rel,
                        alpha=0.3, color="steelblue", label="±1σ")

        ax.set_xlabel(meta["xlabel"])
        ax.set_ylabel(meta["ylabel"])
        ax.set_title(meta["title"])
        ax.legend(fontsize=8)

    fig.suptitle("Emulator Predictions vs Training Spread", fontsize=13)
    fig.tight_layout()
    st.pyplot(fig)

    # Sidebar scatter: current theta vs training distribution
    st.sidebar.markdown("---")
    st.sidebar.subheader("Parameter vs Training Set")
    for p_idx, p_name in enumerate(param_names):
        train_vals = theta_train[:, p_idx]
        current_val = theta_user[p_idx]
        percentile = np.mean(train_vals <= current_val) * 100
        st.sidebar.write(f"{p_name[:20]}: {current_val:.4f} (pct: {percentile:.0f}%)")


if __name__ == "__main__":
    main()

"""
MCMC with PC-space likelihood (compare to 05_mcmc.py).

Instead of projecting GP uncertainty back into data space and computing a
diagonal 20-bin likelihood, we project the observation into PC space and
evaluate the likelihood there directly.

For relation r with k_r PCA components P_r (shape [k, 20]):
  z_obs  = P_r @ (y_obs - pca.mean_)          # project observation to PC space
  mu_z   = GP predictions (mean)               # emulator output
  var_z  = GP predictions (variance)           # emulator uncertainty per PC

Observational noise (bin-diagonal in data space) maps to PC space as:
  Cov_obs_pc = P_r @ diag(sigma_obs_sq) @ P_r.T   (full k×k matrix)

Total covariance in PC space:
  Cov_total = Cov_obs_pc + diag(var_z)

Log-likelihood (multivariate Gaussian in k dimensions):
  log L = -0.5 * (z_obs - mu_z).T @ inv(Cov_total) @ (z_obs - mu_z)
          - 0.5 * log det(Cov_total)

This is more self-consistent than the data-space approach because:
- The GP models exactly k dimensions; computing there is honest about that
- PC scores are uncorrelated by construction (no off-diagonal approximation needed
  within the GP part; only Cov_obs_pc is off-diagonal)
- Avoids the rank-k covariance misspecification in data space

Produces corner plots alongside the data-space results for direct comparison.
"""

import numpy as np
import json
import pickle
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import emcee
import corner
from pathlib import Path
from functools import partial
from sklearn.decomposition import PCA
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel

DATA_DIR = Path(__file__).parent.parent / "results"
FIG_DIR  = Path(__file__).parent.parent / "figures" / "mcmc"

PRIOR_RANGES = np.array([
    [0.03,   0.3  ],
    [0.1,   10.0  ],
    [0.1,   10.0  ],
    [0.0006, 0.006],
    [0.002,  0.05 ],
])

RELATIONS = ["shmr", "gasfrac", "bhstellar"]
DATA_KEYS = {"shmr": "shmr_data", "gasfrac": "gasfrac_data", "bhstellar": "bhstellar_data"}


def normalize_theta(theta):
    return (theta - PRIOR_RANGES[:, 0]) / (PRIOR_RANGES[:, 1] - PRIOR_RANGES[:, 0])


def fit_loo_emulators(npz, full_emulators, theta_norm, mock_idx, pca_k):
    """Refit PCA + GPs on the N-1 simulations excluding mock_idx."""
    loo_emulators = {}
    for rel in RELATIONS:
        data       = npz[DATA_KEYS[rel]]
        valid_mask = full_emulators[rel]["valid_mask"]
        data_valid = data[:, valid_mask]
        k          = pca_k[rel]

        train_data  = np.delete(data_valid, mock_idx, axis=0)
        train_theta = np.delete(theta_norm,  mock_idx, axis=0)

        pca    = PCA(n_components=k)
        scores = pca.fit_transform(train_data)

        gps = []
        for j in range(k):
            print(f"    [{rel}] fitting GP {j+1}/{k}...", end="\r")
            gp = GaussianProcessRegressor(
                kernel=RBF(length_scale_bounds=(1e-2, 1e2)) + WhiteKernel(),
                normalize_y=True, n_restarts_optimizer=5,
            )
            gp.fit(train_theta, scores[:, j])
            gps.append(gp)
        print()

        loo_emulators[rel] = {"pca": pca, "gps": gps, "k": k, "valid_mask": valid_mask}
    return loo_emulators


def gp_predict(emulator, theta_norm_single):
    """Return GP mean and variance in PC space (not propagated to data space)."""
    gps = emulator["gps"]
    k   = emulator["k"]
    mu_z  = np.zeros(k)
    var_z = np.zeros(k)
    for j, gp in enumerate(gps):
        m, s = gp.predict(theta_norm_single.reshape(1, -1), return_std=True)
        mu_z[j]  = m[0]
        var_z[j] = s[0] ** 2
    return mu_z, var_z



def log_likelihood_pcspace(theta, emulators, z_obs_dict):
    """
    Joint log-likelihood in PC space. Noise = GP variance only
    (zero observational noise — exact simulation outputs).
    Covariance is diagonal (diag(var_z)), so this reduces to a sum over PC components,
    plus the log-det term which penalises uncertain regions.
    """
    theta_norm = normalize_theta(theta)
    log_L = 0.0
    for rel in RELATIONS:
        mu_z, var_z  = gp_predict(emulators[rel], theta_norm)
        z_obs        = z_obs_dict[rel]
        residual     = z_obs - mu_z
        # diagonal covariance → simple sum; log_det = sum(log(var_z))
        log_L += -0.5 * np.sum(residual**2 / (var_z + 1e-12))
        log_L += -0.5 * np.sum(np.log(var_z + 1e-12))
    return log_L


def log_prior(theta):
    if np.all(theta >= PRIOR_RANGES[:, 0]) and np.all(theta <= PRIOR_RANGES[:, 1]):
        return 0.0
    return -np.inf


def log_posterior_pcspace(theta, emulators, z_obs_dict):
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood_pcspace(theta, emulators, z_obs_dict)


def run_mcmc(log_post_fn, true_theta, n_walkers=32, n_steps=3000, n_burn=1000):
    n_dim  = len(true_theta)
    sigma0 = 0.01 * (PRIOR_RANGES[:, 1] - PRIOR_RANGES[:, 0])
    p0     = np.clip(
        true_theta + sigma0 * np.random.randn(n_walkers, n_dim),
        PRIOR_RANGES[:, 0], PRIOR_RANGES[:, 1]
    )
    sampler = emcee.EnsembleSampler(n_walkers, n_dim, log_post_fn)
    print(f"    Running {n_steps} steps, {n_walkers} walkers...")
    sampler.run_mcmc(p0, n_steps, progress=False)
    acc = float(np.mean(sampler.acceptance_fraction))
    print(f"    Acceptance fraction: {acc:.3f}", "" if 0.2 <= acc <= 0.5 else "  WARNING: outside [0.2, 0.5]")
    return sampler.get_chain(discard=n_burn, flat=True), acc


def select_mock_observations(theta):
    theta_norm = normalize_theta(theta)
    median_norm = np.median(theta_norm, axis=0)
    dists = np.linalg.norm(theta_norm - median_norm, axis=1)
    idx_median = int(np.argmin(dists))
    idx_lo     = int(np.argmin(np.linalg.norm(theta_norm, axis=1)))
    idx_hi     = int(np.argmax(np.linalg.norm(theta_norm, axis=1)))
    return [idx_median, idx_lo, idx_hi]


def check_credible_intervals(samples, true_theta, param_names):
    results = {}
    for i, p in enumerate(param_names):
        s = samples[:, i]
        lo68, hi68 = np.percentile(s, [16, 84])
        lo95, hi95 = np.percentile(s, [2.5, 97.5])
        results[p] = {"in68": bool(lo68 <= true_theta[i] <= hi68),
                      "in95": bool(lo95 <= true_theta[i] <= hi95)}
    return results


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    npz = np.load(DATA_DIR / "data.npz", allow_pickle=True)
    theta      = npz["theta"]
    param_names = list(npz["param_names"])
    theta_norm  = (theta - PRIOR_RANGES[:, 0]) / (PRIOR_RANGES[:, 1] - PRIOR_RANGES[:, 0])

    with open(DATA_DIR / "pca_k.json") as f:
        pca_k = json.load(f)

    # Load full emulators (used only for valid_mask metadata)
    full_emulators = {}
    for rel in RELATIONS:
        with open(DATA_DIR / f"emulator_{rel}.pkl", "rb") as f:
            full_emulators[rel] = pickle.load(f)

    mock_indices = select_mock_observations(theta)
    labels = ["Closest to median", "Smallest L2 from prior min", "Largest L2 from prior min"]

    print("=== MOCK OBSERVATIONS ===")
    for label, idx in zip(labels, mock_indices):
        print(f"  {label}: sim {idx}  {dict(zip(param_names, theta[idx]))}")

    all_ci = []

    for mock_num, mock_idx in enumerate(mock_indices, start=1):
        print(f"\n=== PC-space MCMC: mock {mock_num} (sim {mock_idx}) — refitting LOO emulator... ===")
        true_theta = theta[mock_idx]

        # Refit emulator excluding this mock
        loo_emulators = fit_loo_emulators(npz, full_emulators, theta_norm, mock_idx, pca_k)

        # Project observation into PC space using LOO PCA
        z_obs_dict = {}
        for rel in RELATIONS:
            valid_mask = loo_emulators[rel]["valid_mask"]
            y_obs = npz[DATA_KEYS[rel]][mock_idx][valid_mask]
            z_obs_dict[rel] = loo_emulators[rel]["pca"].transform(y_obs.reshape(1, -1))[0]

        log_post_fn = partial(
            log_posterior_pcspace,
            emulators=loo_emulators,
            z_obs_dict=z_obs_dict,
        )

        flat_samples, acc = run_mcmc(log_post_fn, true_theta)
        print(f"    Samples shape: {flat_samples.shape}")

        fig = corner.corner(
            flat_samples,
            labels=param_names,
            truths=true_theta,
            quantiles=[0.16, 0.5, 0.84],
            show_titles=True,
            title_fmt=".3g",
            plot_datapoints=False,
            range=list(map(tuple, PRIOR_RANGES)),
            smooth=1.5,
            smooth1d=1.5,
        )
        fig.suptitle(f"PC-space likelihood — Mock {mock_num}: {labels[mock_num-1]}", fontsize=11)
        fname = FIG_DIR / f"corner_pcspace_mock{mock_num}.png"
        fig.savefig(fname, dpi=150)
        plt.close(fig)
        print(f"    Saved {fname}")

        ci = check_credible_intervals(flat_samples, true_theta, param_names)
        all_ci.append(ci)
        for p, r in ci.items():
            print(f"      {p}: 68%={'yes' if r['in68'] else 'NO'}, 95%={'yes' if r['in95'] else 'NO'}")

    print("\n=== PC-SPACE CREDIBLE INTERVAL SUMMARY ===")
    print(f"{'Parameter':<45} {'M1 68%':>6} {'M1 95%':>6} {'M2 68%':>6} {'M2 95%':>6} {'M3 68%':>6} {'M3 95%':>6}")
    for p in param_names:
        row = f"{p:<45}"
        for ci in all_ci:
            row += f"  {'yes' if ci[p]['in68'] else 'NO':>4}  {'yes' if ci[p]['in95'] else 'NO':>4}"
        print(row)

    # Save results
    with open(DATA_DIR / "mcmc_pcspace_ci_results.json", "w") as f:
        json.dump({"mock_indices": mock_indices, "mock_labels": labels, "ci_results": all_ci}, f, indent=2)

    print("\nStep 5b complete.")


if __name__ == "__main__":
    main()

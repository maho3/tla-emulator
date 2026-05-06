"""
MCMC demo inference using the PCA+GP emulator.
Runs emcee on 3 mock observations and produces corner plots.
"""

import numpy as np
import json
import pickle
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import emcee
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "results"
FIG_DIR = Path(__file__).parent.parent / "figures" / "mcmc"

PRIOR_RANGES = np.array([
    [0.03, 0.3],
    [0.1, 10.0],
    [0.1, 10.0],
    [0.0006, 0.006],
    [0.002, 0.05],
])

RELATIONS = ["shmr", "gasfrac", "bhstellar"]
DATA_KEYS = {
    "shmr": "shmr_data",
    "gasfrac": "gasfrac_data",
    "bhstellar": "bhstellar_data",
}


def normalize_theta(theta):
    return (theta - PRIOR_RANGES[:, 0]) / (PRIOR_RANGES[:, 1] - PRIOR_RANGES[:, 0])


def predict_emulator(emulator, theta_norm_single):
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


def log_prior(theta):
    if np.all(theta >= PRIOR_RANGES[:, 0]) and np.all(theta <= PRIOR_RANGES[:, 1]):
        return 0.0
    return -np.inf


def log_likelihood(theta, emulators, obs_data, sigma_obs_sq):
    """Joint log-likelihood over all 3 relations."""
    theta_norm = normalize_theta(theta)
    log_L = 0.0
    for rel, obs, sig_sq in zip(RELATIONS, obs_data, sigma_obs_sq):
        emulator = emulators[rel]
        mu_pred, var_pred = predict_emulator(emulator, theta_norm)
        total_var = sig_sq + var_pred
        log_L += -0.5 * np.sum((obs - mu_pred) ** 2 / (total_var + 1e-12))
    return log_L


def log_posterior(theta, emulators, obs_data, sigma_obs_sq):
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(theta, emulators, obs_data, sigma_obs_sq)


def select_mock_observations(theta, valid_indices):
    """Select 3 mock observations."""
    theta_norm = normalize_theta(theta)
    median_norm = np.median(theta_norm, axis=0)

    # 1. Closest to median in normalized space
    dists_to_median = np.linalg.norm(theta_norm - median_norm, axis=1)
    idx_median = int(np.argmin(dists_to_median))

    # 2. Smallest L2 norm from prior minimum (theta_norm closest to origin)
    dists_to_corner_lo = np.linalg.norm(theta_norm, axis=1)
    idx_lo = int(np.argmin(dists_to_corner_lo))

    # 3. Largest L2 norm (theta_norm furthest from origin = closest to upper corner)
    idx_hi = int(np.argmax(dists_to_corner_lo))

    return [idx_median, idx_lo, idx_hi]


def run_mcmc(log_post_fn, true_theta, n_walkers=32, n_steps=3000, n_burn=1000):
    """Run emcee sampler initialized around true_theta."""
    n_dim = len(true_theta)
    # Initial ball: 1% of prior range per parameter
    prior_width = PRIOR_RANGES[:, 1] - PRIOR_RANGES[:, 0]
    sigma_init = 0.01 * prior_width
    p0 = true_theta + sigma_init * np.random.randn(n_walkers, n_dim)
    # Clip to prior
    p0 = np.clip(p0, PRIOR_RANGES[:, 0], PRIOR_RANGES[:, 1])

    sampler = emcee.EnsembleSampler(n_walkers, n_dim, log_post_fn)
    print(f"    Running {n_steps} steps with {n_walkers} walkers...")
    sampler.run_mcmc(p0, n_steps, progress=False)

    acc_frac = np.mean(sampler.acceptance_fraction)
    print(f"    Acceptance fraction: {acc_frac:.3f}")
    if not (0.2 <= acc_frac <= 0.5):
        print(f"    WARNING: Acceptance fraction {acc_frac:.3f} outside [0.2, 0.5]!")

    flat_samples = sampler.get_chain(discard=n_burn, flat=True)
    return flat_samples, acc_frac


def check_credible_intervals(samples, true_theta, param_names):
    """Check if true parameters fall within 68% and 95% CI."""
    results = {}
    for i, p_name in enumerate(param_names):
        s = samples[:, i]
        lo68, hi68 = np.percentile(s, [16, 84])
        lo95, hi95 = np.percentile(s, [2.5, 97.5])
        in68 = bool(lo68 <= true_theta[i] <= hi68)
        in95 = bool(lo95 <= true_theta[i] <= hi95)
        results[p_name] = {"in68": in68, "in95": in95}
    return results


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    # Install corner if needed
    try:
        import corner
    except ImportError:
        import subprocess
        subprocess.run(["conda", "run", "-n", "ili-torch", "pip", "install", "corner"], check=True)
        import corner

    npz = np.load(DATA_DIR / "data.npz", allow_pickle=True)
    theta = npz["theta"]
    param_names = list(npz["param_names"])
    valid_indices = list(npz["valid_indices"])

    # Load emulators
    emulators = {}
    for rel in RELATIONS:
        with open(DATA_DIR / f"emulator_{rel}.pkl", "rb") as f:
            emulators[rel] = pickle.load(f)

    # Data variance for likelihood (diagonal variance across all sims per bin)
    sigma_obs_sq_list = []
    for rel in RELATIONS:
        data = npz[DATA_KEYS[rel]]
        emulator = emulators[rel]
        valid_mask = emulator["valid_mask"]
        data_valid = data[:, valid_mask]
        sigma_obs_sq_list.append(np.var(data_valid, axis=0))

    # Select mock observations
    mock_indices = select_mock_observations(theta, valid_indices)
    print("=== MOCK OBSERVATIONS ===")
    labels = ["Closest to median", "Smallest L2 from prior min", "Largest L2 from prior min"]
    for label, idx in zip(labels, mock_indices):
        print(f"  {label}: sim index {idx}")
        print(f"    {dict(zip(param_names, theta[idx]))}")

    all_ci_results = []

    for mock_num, mock_idx in enumerate(mock_indices, start=1):
        print(f"\n=== MCMC for mock {mock_num} (sim {mock_idx}) ===")
        true_theta = theta[mock_idx]

        # Observed data = true simulation values
        obs_data = []
        for rel in RELATIONS:
            data = npz[DATA_KEYS[rel]]
            emulator = emulators[rel]
            valid_mask = emulator["valid_mask"]
            obs_data.append(data[mock_idx][valid_mask])

        from functools import partial
        log_post_fn = partial(
            log_posterior,
            emulators=emulators,
            obs_data=obs_data,
            sigma_obs_sq=sigma_obs_sq_list,
        )

        flat_samples, acc_frac = run_mcmc(log_post_fn, true_theta)
        print(f"    Posterior samples shape: {flat_samples.shape}")

        # Corner plot
        import corner
        fig = corner.corner(
            flat_samples,
            labels=param_names,
            truths=true_theta,
            quantiles=[0.16, 0.5, 0.84],
            show_titles=True,
            title_fmt=".3g",
            plot_datapoints=False,
        )
        fig.suptitle(f"Mock {mock_num}: {labels[mock_num-1]}", fontsize=11)
        fname = FIG_DIR / f"corner_mock{mock_num}.png"
        fig.savefig(fname, dpi=150)
        plt.close(fig)
        print(f"    Saved corner plot to {fname}")

        # Check credible intervals
        ci_results = check_credible_intervals(flat_samples, true_theta, param_names)
        all_ci_results.append(ci_results)
        print(f"    Credible interval coverage:")
        for p_name, res in ci_results.items():
            print(f"      {p_name}: 68%={'yes' if res['in68'] else 'NO'}, 95%={'yes' if res['in95'] else 'NO'}")

    # Summary table
    print("\n=== CREDIBLE INTERVAL SUMMARY ===")
    print(f"{'Parameter':<45} {'Mock1 68%':>9} {'Mock1 95%':>9} {'Mock2 68%':>9} {'Mock2 95%':>9} {'Mock3 68%':>9} {'Mock3 95%':>9}")
    for p_name in param_names:
        row = f"{p_name:<45}"
        for ci_res in all_ci_results:
            row += f"  {'yes' if ci_res[p_name]['in68'] else 'NO':>7}"
            row += f"  {'yes' if ci_res[p_name]['in95'] else 'NO':>7}"
        print(row)

    # Save CI results
    with open(DATA_DIR / "mcmc_ci_results.json", "w") as f:
        json.dump({
            "mock_indices": mock_indices,
            "mock_labels": labels,
            "ci_results": all_ci_results,
        }, f, indent=2)
    print("\nStep 5 complete.")


if __name__ == "__main__":
    main()

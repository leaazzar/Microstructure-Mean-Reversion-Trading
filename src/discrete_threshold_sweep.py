import matplotlib.pyplot as plt
import numpy as np

from simulate_discrete import simulate_discrete_gap
from monte_carlo_sweep import (paired_diff, s_G, summarize_for_phi, theta_dawson,
                                theta_star, theta_trade_stats)


def run_seeds_discrete(thetas, alpha, mu, delta, sigma_X, dt, duration, n_seeds, base_seed=2000):
    n_steps = int(duration / dt)
    per_seed = []
    for i in range(n_seeds):
        sim = simulate_discrete_gap(n_steps, alpha, mu, delta, sigma_X, dt, seed=base_seed + i)
        per_seed.append(theta_trade_stats(sim["gap"], thetas, dt))
    return per_seed


def plot_comparison(thetas, summary, theta_D_ou, theta_star_ou, phi, n_seeds,
                     path="paper/notes/discrete_mc_sweep.png"):
    thetas_arr = np.array(sorted(thetas))
    means = np.array([summary[th]["mean_rate"] for th in thetas_arr])
    sems = np.array([summary[th]["sem_rate"] for th in thetas_arr])

    plt.figure()
    plt.errorbar(thetas_arr, means, yerr=sems, marker="o", markersize=3, capsize=3,
                 linewidth=1, label="discrete model: mean profit rate ± SE")
    plt.axvline(theta_D_ou, color="darkorange", linestyle="--",
                label=f"theta_D (OU surrogate, exact) = {theta_D_ou:.4f}")
    plt.axvline(theta_star_ou, color="red", linestyle="--",
                label=f"theta* (OU leading-order) = {theta_star_ou:.4f}")
    plt.xlabel("theta")
    plt.ylabel("profit rate (per second)")
    plt.title(f"Discrete tick-jump model threshold sweep (phi={phi}, n_seeds={n_seeds})")
    plt.legend(fontsize=8)
    plt.savefig(path, dpi=150)
    plt.show()


if __name__ == "__main__":
    alpha, mu, delta, sigma_X, dt, duration = 0.5, 1.0, 0.005, 0.02, 0.001, 100
    phi = 0.005

    # Same alpha, phi as the OU section; the OU section's s_G=0.02 by construction
    # (sigma=0.02, alpha=0.5), giving the theta_D this discrete model is compared to.
    s_g_ou = 0.02
    theta_D_ou = theta_dawson(phi, s_g_ou)
    theta_star_ou = theta_star(phi, s_g_ou)
    print(f"OU-surrogate benchmarks: theta* = {theta_star_ou:.5f}, theta_D = {theta_D_ou:.5f}")

    thetas_grid = sorted(set(np.round(np.arange(0.008, 0.0341, 0.001), 4).tolist())
                          | {theta_D_ou, theta_star_ou})

    n_seeds = 200
    per_seed = run_seeds_discrete(thetas_grid, alpha, mu, delta, sigma_X, dt, duration, n_seeds)

    summary = summarize_for_phi(per_seed, thetas_grid, phi, duration)
    best_th = max(thetas_grid, key=lambda th: summary[th]["mean_rate"])
    print(f"\ndiscrete-model Monte Carlo sweep ({n_seeds} seeds, phi={phi}): "
          f"empirical grid-argmax peak = theta={best_th:.4f}")
    for th in sorted(thetas_grid):
        s = summary[th]
        tag = ""
        if abs(th - theta_D_ou) < 1e-9:
            tag = "  <- theta_D (OU)"
        if abs(th - theta_star_ou) < 1e-9:
            tag = "  <- theta* (OU)"
        if abs(th - best_th) < 1e-9:
            tag += "  <- discrete empirical peak"
        print(f"  theta={th:.4f}: mean rate={s['mean_rate']:.5f} ± {s['sem_rate']:.5f} (SE), "
              f"reversals={s['mean_reversals']:.2f}, wait={s['mean_wait']:.2f}s{tag}")

    print("\npaired comparisons (same 200 discrete-model seeds each side):")
    for a, b, label in [
        (best_th, theta_D_ou, "discrete empirical peak - theta_D (OU)"),
        (theta_D_ou, theta_star_ou, "theta_D (OU) - theta* (OU), evaluated on discrete paths"),
    ]:
        if abs(a - b) < 1e-9:
            print(f"  {label}: identical theta, no comparison")
            continue
        d, se = paired_diff(per_seed, a, b, phi, duration)
        print(f"  {label}: {d:+.5f} ± {se:.5f}  ({abs(d)/se:.2f} SE from zero)")

    rel_gap = 100 * (theta_D_ou - best_th) / theta_D_ou
    print(f"\ndiscrete empirical peak vs theta_D: {best_th:.4f} vs {theta_D_ou:.4f} "
          f"({rel_gap:+.1f}% -- positive means the discrete peak sits inside theta_D, "
          f"the direction the paper's jump-overshoot correction predicts)")

    plot_comparison(thetas_grid, summary, theta_D_ou, theta_star_ou, phi, n_seeds)

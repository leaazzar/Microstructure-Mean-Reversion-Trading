import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import brentq
from scipy.special import dawsn

from simulate_ou import simulate_ou
from trading_strategy import run_trading_rule


def s_G(alpha, sigma):
    return sigma / np.sqrt(2 * alpha)


def theta_star(phi, s_g):
    """Large-threshold (Kramers/Arrhenius) approximation: theta*(theta*-phi) = s_G^2."""
    return (phi + np.sqrt(phi**2 + 4 * s_g**2)) / 2


def theta_dawson(phi, s_g):
    """
    Exact continuous-surrogate optimum. Solves u - gamma = sqrt(2) * D(u/sqrt(2))
    for u = theta/s_g, gamma = phi/s_g, D the Dawson function -- the exact
    first-order condition theta_star(theta*-phi)=s_G^2 is its large-u asymptote.
    """
    gamma = phi / s_g

    def f(u):
        return u - gamma - np.sqrt(2) * dawsn(u / np.sqrt(2))

    lo = gamma + 1e-9
    hi = lo + 1.0
    while f(hi) < 0:
        hi *= 2
    u_d = brentq(f, lo, hi)
    return u_d * s_g


def theta_trade_stats(gap, thetas, dt):
    """
    Run each theta once on the same gap path. Reward is affine in phi at
    fixed trades (each trade subtracts phi*quantity), and trade timing and
    quantities never depend on phi (only the threshold-crossing logic does,
    which has no phi in it). So each seed's trades are simulated ONCE per
    theta (at phi=0, giving reward0 and total_quantity), and profit_rate for
    any phi is recovered after the fact via
        rate(phi) = (reward0 - phi*total_quantity) / duration
    with no re-simulation.
    """
    stats = {}
    for th in thetas:
        trades = run_trading_rule(gap, theta=th, phi=0.0, dt=dt)
        n_trades = len(trades)
        n_reversals = sum(1 for tr in trades if tr.quantity == 2)
        total_quantity = sum(tr.quantity for tr in trades)
        reward0 = sum(tr.reward for tr in trades)

        times = [tr.time for tr in trades]
        time_between = np.diff(times) if n_trades > 1 else np.array([])
        avg_wait = time_between.mean() if len(time_between) else np.nan

        stats[th] = dict(n_trades=n_trades, n_reversals=n_reversals,
                          total_quantity=total_quantity, reward0=reward0,
                          avg_wait=avg_wait)
    return stats


def run_seeds(thetas, alpha, sigma, dt, duration, n_seeds, base_seed=1000):
    n_steps = int(duration / dt)
    per_seed = []
    for i in range(n_seeds):
        gap = simulate_ou(n_steps=n_steps, alpha=alpha, mu=0.0, sigma=sigma, dt=dt, seed=base_seed + i)
        per_seed.append(theta_trade_stats(gap, thetas, dt))
    return per_seed


def rate_series(per_seed, theta, phi, duration):
    """Per-seed profit rate at a given theta, phi -- the array paired comparisons are built from."""
    return np.array([(s[theta]["reward0"] - phi * s[theta]["total_quantity"]) / duration for s in per_seed])


def summarize_for_phi(per_seed, thetas, phi, duration):
    n_seeds = len(per_seed)
    summary = {}
    for th in thetas:
        rates = rate_series(per_seed, th, phi, duration)
        reversals = np.array([s[th]["n_reversals"] for s in per_seed])
        waits = np.array([s[th]["avg_wait"] for s in per_seed])
        summary[th] = dict(
            mean_rate=rates.mean(),
            sem_rate=rates.std(ddof=1) / np.sqrt(n_seeds) if n_seeds > 1 else float("nan"),
            mean_reversals=reversals.mean(),
            mean_wait=np.nanmean(waits),
        )
    return summary


def paired_diff(per_seed, theta_a, theta_b, phi, duration):
    """
    Paired comparison of R(theta_a) - R(theta_b): since every theta is run on
    the SAME seeds, differencing within each seed before averaging removes
    the shared path-to-path noise, giving a tighter SE than comparing two
    independent-looking error bars would.
    """
    diffs = rate_series(per_seed, theta_a, phi, duration) - rate_series(per_seed, theta_b, phi, duration)
    n = len(diffs)
    return diffs.mean(), diffs.std(ddof=1) / np.sqrt(n)


def plot_mc_sweep(thetas, summary, theta_star_val, theta_dawson_val, phi, n_seeds,
                   path="paper/notes/ou_mc_sweep.png"):
    thetas_arr = np.array(sorted(thetas))
    means = np.array([summary[th]["mean_rate"] for th in thetas_arr])
    sems = np.array([summary[th]["sem_rate"] for th in thetas_arr])

    plt.figure()
    plt.errorbar(thetas_arr, means, yerr=sems, marker="o", markersize=3, capsize=3,
                 linewidth=1, label="mean profit rate ± SE")
    plt.axvline(theta_star_val, color="red", linestyle="--",
                label=f"theta* leading-order = {theta_star_val:.4f}")
    plt.axvline(theta_dawson_val, color="darkorange", linestyle="--",
                label=f"theta_D exact (Dawson) = {theta_dawson_val:.4f}")
    plt.xlabel("theta")
    plt.ylabel("profit rate (per second)")
    plt.title(f"Monte Carlo threshold sweep (phi={phi}, n_seeds={n_seeds})")
    plt.legend(fontsize=8)
    plt.savefig(path, dpi=150)
    plt.show()


def plot_phi_sweep(per_seed, thetas_grid, phis, alpha, sigma, duration,
                    path="paper/notes/ou_phi_sweep.png"):
    s_g = s_G(alpha, sigma)
    empirical_peaks, theory_star_pts, theory_dawson_pts = [], [], []
    for phi in phis:
        summary = summarize_for_phi(per_seed, thetas_grid, phi, duration)
        best_th = max(thetas_grid, key=lambda th: summary[th]["mean_rate"])
        empirical_peaks.append(best_th)
        theory_star_pts.append(theta_star(phi, s_g))
        theory_dawson_pts.append(theta_dawson(phi, s_g))

    phi_fine = np.linspace(0, max(phis), 200)
    star_curve = [theta_star(p, s_g) for p in phi_fine]
    dawson_curve = [theta_dawson(p, s_g) for p in phi_fine]

    plt.figure()
    plt.plot(phi_fine, star_curve, color="red", linestyle="--", label="theta* (leading-order)")
    plt.plot(phi_fine, dawson_curve, color="darkorange", label="theta_D (exact, Dawson)")
    plt.scatter(phis, empirical_peaks, color="black", zorder=5, label="empirical peak (grid argmax)")
    plt.xlabel("phi (half-spread)")
    plt.ylabel("optimal theta")
    plt.title("Optimal threshold vs half-spread")
    plt.legend(fontsize=8)
    plt.savefig(path, dpi=150)
    plt.show()

    return empirical_peaks, theory_star_pts, theory_dawson_pts


if __name__ == "__main__":
    alpha, sigma, dt, duration = 0.5, 0.02, 0.01, 100
    phi = 0.005
    s_g = s_G(alpha, sigma)
    th_star = theta_star(phi, s_g)
    th_dawson = theta_dawson(phi, s_g)
    print(f"s_G = {s_g:.4f}, gamma = {phi/s_g:.4f}")
    print(f"theta* (leading-order)  = {th_star:.5f}")
    print(f"theta_D (exact, Dawson) = {th_dawson:.5f}  "
          f"({100*(th_star-th_dawson)/th_star:.1f}% below theta*)")

    # Broad grid: fine enough to resolve the peak, wide enough (down to 0.005)
    # that small-phi optima aren't cut off at the boundary, and includes the
    # exact theta* / theta_D values as real grid points for paired testing.
    thetas_grid = sorted(set(np.round(np.arange(0.005, 0.0381, 0.001), 4).tolist()))
    special_points = set()
    for p in [0.0025, 0.005, 0.01, 0.02]:
        special_points.add(theta_star(p, s_g))
        special_points.add(theta_dawson(p, s_g))
    thetas_grid = sorted(set(thetas_grid) | special_points)

    n_seeds = 200
    per_seed = run_seeds(thetas_grid, alpha, sigma, dt, duration, n_seeds)

    summary = summarize_for_phi(per_seed, thetas_grid, phi, duration)
    best_th = max(thetas_grid, key=lambda th: summary[th]["mean_rate"])
    print(f"\nMonte Carlo sweep ({n_seeds} seeds, phi={phi}): "
          f"empirical grid-argmax peak = theta={best_th:.4f}")
    for th in sorted(thetas_grid):
        if 0.012 <= th <= 0.030:
            s = summary[th]
            tag = ""
            if abs(th - th_dawson) < 1e-9:
                tag = "  <- theta_D"
            if abs(th - th_star) < 1e-9:
                tag = "  <- theta*"
            if abs(th - best_th) < 1e-9:
                tag += "  <- empirical peak"
            print(f"  theta={th:.4f}: mean rate={s['mean_rate']:.5f} ± {s['sem_rate']:.5f} (SE), "
                  f"reversals={s['mean_reversals']:.2f}, wait={s['mean_wait']:.2f}s{tag}")

    print("\npaired comparisons (mean diff ± paired SE, same 200 seeds each side):")
    for a, b, label in [
        (best_th, th_dawson, "empirical peak - theta_D"),
        (th_dawson, th_star, "theta_D - theta*"),
        (best_th, th_star, "empirical peak - theta*"),
    ]:
        d, se = paired_diff(per_seed, a, b, phi, duration)
        print(f"  {label}: {d:+.5f} ± {se:.5f}  ({abs(d)/se:.2f} SE from zero)" if se > 0
              else f"  {label}: {d:+.5f} (identical theta, no comparison)")

    plot_mc_sweep(thetas_grid, summary, th_star, th_dawson, phi, n_seeds)

    print("\nvarying phi -- theta* vs theta_D (exact) vs empirical peak:")
    phis = [0.0025, 0.005, 0.01, 0.02]
    for p in phis:
        p_summary = summarize_for_phi(per_seed, thetas_grid, p, duration)
        p_best = max(thetas_grid, key=lambda th: p_summary[th]["mean_rate"])
        at_lower_edge = " (AT GRID'S LOWER EDGE -- extend grid further)" if p_best <= min(thetas_grid) + 1e-9 else ""
        print(f"  phi={p:.4f}: theta*={theta_star(p, s_g):.4f}, theta_D={theta_dawson(p, s_g):.4f}, "
              f"empirical={p_best:.4f}{at_lower_edge}")

    plot_phi_sweep(per_seed, thetas_grid, phis, alpha, sigma, duration)

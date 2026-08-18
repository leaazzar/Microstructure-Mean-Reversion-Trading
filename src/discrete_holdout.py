import numpy as np

from discrete_threshold_sweep import run_seeds_discrete
from monte_carlo_sweep import rate_series

if __name__ == "__main__":
    alpha, mu, delta, sigma_X, dt, duration = 0.5, 1.0, 0.005, 0.02, 0.001, 100
    phi = 0.005

    # Frozen BEFORE generating any new data -- these are the three candidates
    # from the prior (in-sample) sweep: the sweep's own grid-argmax winner,
    # and the OU section's theta_D and theta*. No re-selection happens here.
    theta_emp = 0.0180
    theta_D = 0.0193
    theta_star = 0.0227
    thetas = [theta_emp, theta_D, theta_star]

    # Fresh seeds, disjoint from the base_seed=2000 block used to pick theta_emp.
    n_seeds = 200
    per_seed = run_seeds_discrete(thetas, alpha, mu, delta, sigma_X, dt, duration,
                                   n_seeds, base_seed=5000)

    r_emp = rate_series(per_seed, theta_emp, phi, duration)
    r_D = rate_series(per_seed, theta_D, phi, duration)
    r_star = rate_series(per_seed, theta_star, phi, duration)

    print(f"holdout sweep: {n_seeds} fresh seeds (base_seed=5000, disjoint from the "
          f"base_seed=2000 selection sample), phi={phi}\n")
    for name, r in [("theta_emp=0.0180", r_emp), ("theta_D=0.0193", r_D), ("theta*=0.0227", r_star)]:
        print(f"  {name}: mean rate = {r.mean():.5f} ± {r.std(ddof=1)/np.sqrt(n_seeds):.5f} (SE)")

    def report(diffs, label, baseline_mean):
        n = len(diffs)
        mean_d = diffs.mean()
        se = diffs.std(ddof=1) / np.sqrt(n)
        ci_lo, ci_hi = mean_d - 1.96 * se, mean_d + 1.96 * se
        pct = 100 * mean_d / baseline_mean
        print(f"\n{label}")
        print(f"  mean diff       = {mean_d:+.5f}")
        print(f"  SE              = {se:.5f}")
        print(f"  95% CI          = [{ci_lo:+.5f}, {ci_hi:+.5f}]")
        print(f"  as % of theta_D rate = {pct:+.2f}%")
        includes_zero = ci_lo <= 0 <= ci_hi
        print(f"  CI includes zero: {includes_zero}")
        return ci_lo, ci_hi

    print("\n" + "=" * 60)
    diffs1 = r_emp - r_D
    ci1 = report(diffs1, "R(0.0180) - R(theta_D):", r_D.mean())
    print("  -> " + ("the data do NOT distinguish theta=0.018 from theta_D"
                      if ci1[0] <= 0 <= ci1[1]
                      else "0.018 detectably differs from theta_D on this holdout sample"))

    print("\n" + "=" * 60)
    diffs2 = r_D - r_star
    ci2 = report(diffs2, "R(theta_D) - R(theta*):", r_D.mean())
    print("  -> " + ("the data do NOT detect a disadvantage from using theta* in the jump model"
                      if ci2[0] <= 0 <= ci2[1]
                      else "theta* detectably underperforms theta_D on this holdout sample"))

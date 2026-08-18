import numpy as np

from discrete_threshold_sweep import run_seeds_discrete
from monte_carlo_sweep import rate_series

ALPHA, MU, SIGMA_X = 0.5, 1.0, 0.02
PHI = 0.005
DURATION = 100
THETA_EMP, THETA_D, THETA_STAR = 0.0180, 0.01928636498888229, 0.022655644370746374
N_SEEDS = 150


def paired_report(r_a, r_b, label):
    diffs = r_a - r_b
    n = len(diffs)
    mean_d, se = diffs.mean(), diffs.std(ddof=1) / np.sqrt(n)
    pct = 100 * mean_d / r_b.mean()
    print(f"    {label}: mean diff={mean_d:+.3e}  SE={se:.3e}  "
          f"({abs(mean_d)/se:.2f} SE, {pct:+.3f}% of baseline rate)")


def run_one(delta, dt, base_seed):
    per_seed = run_seeds_discrete([THETA_EMP, THETA_D, THETA_STAR], ALPHA, MU, delta,
                                   SIGMA_X, dt, DURATION, N_SEEDS, base_seed=base_seed)
    r_emp = rate_series(per_seed, THETA_EMP, PHI, DURATION)
    r_D = rate_series(per_seed, THETA_D, PHI, DURATION)
    r_star = rate_series(per_seed, THETA_STAR, PHI, DURATION)
    return r_emp, r_D, r_star


if __name__ == "__main__":
    print(f"Frozen thresholds: theta_emp={THETA_EMP}, theta_D={THETA_D:.6f}, "
          f"theta*={THETA_STAR:.6f}, n_seeds={N_SEEDS}\n")

    print("=" * 70)
    print("1) TICK-SIZE SWEEP (dt=0.001 fixed, delta varied)")
    print("=" * 70)
    for delta in [0.0025, 0.005, 0.01, 0.02]:
        max_p = ALPHA / delta * (3 * SIGMA_X) * 0.001 + 2 * MU * 0.001  # rough worst-case check
        print(f"\ndelta={delta}  (tick/s_G ratio = {delta/0.02:.3f})")
        r_emp, r_D, r_star = run_one(delta, 0.001, base_seed=9000 + int(delta * 100000))
        print(f"    mean rates: emp={r_emp.mean():.5f}  theta_D={r_D.mean():.5f}  "
              f"theta*={r_star.mean():.5f}")
        paired_report(r_emp, r_D, "R(0.018) - R(theta_D)  ")
        paired_report(r_D, r_star, "R(theta_D) - R(theta*)  ")

    print("\n" + "=" * 70)
    print("2) TIME-STEP SWEEP (delta=0.005 fixed, dt varied)")
    print("=" * 70)
    for dt in [0.001, 0.0005, 0.0002]:
        print(f"\ndt={dt}")
        r_emp, r_D, r_star = run_one(0.005, dt, base_seed=9500 + int(1 / dt))
        print(f"    mean rates: emp={r_emp.mean():.5f}  theta_D={r_D.mean():.5f}  "
              f"theta*={r_star.mean():.5f}")
        paired_report(r_emp, r_D, "R(0.018) - R(theta_D)  ")
        paired_report(r_D, r_star, "R(theta_D) - R(theta*)  ")

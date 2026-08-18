import matplotlib.pyplot as plt
import numpy as np


def simulate_discrete_gap(n_steps, alpha, mu, delta, sigma_X, dt, seed=None):
    """
    X: continuous efficient price (Brownian). M: mid, moves only by +-delta
    ticks via a thinned two-event (up/down) jump process whose intensities
    lean toward X. G = M - X is saved post-jump; g_pre is the gap the
    intensities were evaluated at (pre-jump, post-X-move), kept only for
    validating the up/down asymmetry against the sign of G.
    """
    rng = np.random.default_rng(seed)
    sqrt_dt = np.sqrt(dt)

    X = 0.0
    M = 0.0
    eff = np.empty(n_steps)
    mid = np.empty(n_steps)
    gap = np.empty(n_steps)
    g_pre = np.empty(n_steps)
    jump = np.empty(n_steps, dtype=np.int8)   # +1 up, -1 down, 0 none
    max_p_total = 0.0

    for t in range(n_steps):
        X += sigma_X * sqrt_dt * rng.standard_normal()

        g = M - X
        lam_up = mu + (alpha / delta) * max(-g, 0.0)
        lam_down = mu + (alpha / delta) * max(g, 0.0)
        p_up = lam_up * dt
        p_down = lam_down * dt
        max_p_total = max(max_p_total, p_up + p_down)

        u = rng.random()
        if u < p_up:
            M += delta
            jump[t] = 1
        elif u < p_up + p_down:
            M -= delta
            jump[t] = -1
        else:
            jump[t] = 0

        eff[t] = X
        mid[t] = M
        g_pre[t] = g
        gap[t] = M - X

    return dict(gap=gap, g_pre=g_pre, mid=mid, efficient=eff, jump=jump,
                dt=dt, max_p_total=max_p_total)


def autocovariance(x, max_lag):
    x = x - x.mean()
    n = len(x)
    return np.array([np.mean(x[:n - k] * x[k:]) if k > 0 else np.mean(x * x)
                      for k in range(max_lag)])


def validate_structure(sim, alpha, delta, dt, sigma_X):
    """Fast checks: tick grid, continuity, up/down asymmetry, mean -- cheap on a short path."""
    gap, g_pre, mid, jump = sim["gap"], sim["g_pre"], sim["mid"], sim["jump"]

    print(f"max (lambda_up+lambda_down)*dt observed = {sim['max_p_total']:.4f}  "
          f"(should be << 1)")

    mid_diffs = np.diff(mid)
    steps_in_ticks = np.round(mid_diffs / delta)
    on_grid = np.allclose(mid_diffs, steps_in_ticks * delta, atol=1e-12)
    only_unit_steps = np.all(np.isin(steps_in_ticks, [-1, 0, 1]))
    print(f"mid moves only in +-1 tick steps: {on_grid and only_unit_steps}")

    eff_diffs = np.diff(sim["efficient"])
    frac_on_tick_grid = np.mean(np.isclose(np.mod(eff_diffs, delta), 0, atol=1e-9)
                                 | np.isclose(np.mod(eff_diffs, delta), delta, atol=1e-9))
    print(f"efficient price looks continuous: "
          f"std(dX)={eff_diffs.std():.6f} (theory sigma_X*sqrt(dt)={sigma_X*np.sqrt(dt):.6f}), "
          f"fraction of dX landing on the tick grid = {frac_on_tick_grid:.4f} (should be ~0)")

    up = jump == 1
    down = jump == -1
    neg_g = g_pre < 0
    pos_g = g_pre > 0
    print(f"P(up-jump | G<0)   = {up[neg_g].mean():.5f}")
    print(f"P(up-jump | G>=0)  = {up[~neg_g].mean():.5f}  (should be smaller)")
    print(f"P(down-jump | G>0) = {down[pos_g].mean():.5f}")
    print(f"P(down-jump | G<=0)= {down[~pos_g].mean():.5f}  (should be smaller)")

    mean_gap = gap.mean()
    s_g_hat = gap.std()
    print(f"mean(G) = {mean_gap:+.5f}  (should be near 0)")
    print(f"std(G)  = {s_g_hat:.5f}")

    return dict(mean_gap=mean_gap, s_g_hat=s_g_hat)


def validate_decay_rate(sim, alpha, dt, burn_in_seconds=50.0, max_lag_seconds=6.0):
    """
    Decay-rate check: needs far more data than the structural checks above.
    A short path (~100-2000s) gives a systematically-too-fast fitted rate --
    not a model bug, just an unreliable single-exponential fit to a noisy
    long-lag autocovariance estimate. See the research log for the
    diagnosis; ~5000+ seconds (2500+ decorrelation times at alpha=0.5) is
    needed for the fit to settle near the true alpha.
    """
    gap = sim["gap"][int(burn_in_seconds / dt):]
    max_lag = int(max_lag_seconds / dt)
    acov = autocovariance(gap, max_lag)
    lags_t = np.arange(max_lag) * dt
    positive = acov > 0
    slope, intercept = np.polyfit(lags_t[positive], np.log(acov[positive] / acov[0]), 1)
    print(f"autocovariance decay (duration={len(sim['gap'])*dt:.0f}s, "
          f"{burn_in_seconds:.0f}s burn-in discarded): "
          f"fitted rate = {-slope:.4f}  (input alpha = {alpha})")

    return dict(alpha_hat=-slope, lags_t=lags_t, acov=acov)


def plot_path(sim, dt, window_seconds=10, path="paper/notes/discrete_path.png"):
    n = int(window_seconds / dt)
    t = np.arange(n) * dt

    _, axes = plt.subplots(2, 1, sharex=True, figsize=(8, 6))
    axes[0].step(t, sim["mid"][:n], where="post", label="mid M (ticks)", color="black")
    axes[0].plot(t, sim["efficient"][:n], label="efficient price X", color="tab:blue", linewidth=1)
    axes[0].set_ylabel("price")
    axes[0].legend(fontsize=8)
    axes[0].set_title("Discrete tick-jump model: mid vs efficient price")

    axes[1].plot(t, sim["gap"][:n], color="tab:green", linewidth=1)
    axes[1].axhline(0, color="black", linewidth=0.5)
    axes[1].set_ylabel("gap G")
    axes[1].set_xlabel("time (s)")

    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.show()


def plot_autocovariance(validation, alpha, s_g_hat, path="paper/notes/discrete_autocov.png"):
    lags_t, acov = validation["lags_t"], validation["acov"]

    plt.figure()
    plt.plot(lags_t, acov / acov[0], label="empirical autocorrelation")
    plt.plot(lags_t, np.exp(-alpha * lags_t), "--", color="red", label=f"theory e^(-alpha h), alpha={alpha}")
    plt.xlabel("lag h (s)")
    plt.ylabel("autocorrelation of G")
    plt.title("Gap autocovariance: discrete model vs OU theory")
    plt.legend()
    plt.savefig(path, dpi=150)
    plt.show()


if __name__ == "__main__":
    alpha = 0.5
    mu = 1.0
    delta = 0.005
    sigma_X = 0.02
    dt = 0.001

    duration = 100
    n_steps = int(duration / dt)
    sim = simulate_discrete_gap(n_steps, alpha, mu, delta, sigma_X, dt, seed=0)
    print(f"structural checks: n_steps={n_steps}, dt={dt}, duration={duration}s\n")
    struct = validate_structure(sim, alpha, delta, dt, sigma_X)
    plot_path(sim, dt)

    print()
    duration_long = 6000
    n_steps_long = int(duration_long / dt)
    sim_long = simulate_discrete_gap(n_steps_long, alpha, mu, delta, sigma_X, dt, seed=1)
    decay = validate_decay_rate(sim_long, alpha, dt)
    plot_autocovariance(decay, alpha, sim_long["gap"].std())

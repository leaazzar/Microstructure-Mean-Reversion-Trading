import matplotlib.pyplot as plt
import numpy as np


def simulate_ou(n_steps, alpha=1.0, mu=0.0, sigma=1.0, dt=1.0, seed=None):
    rng = np.random.default_rng(seed)

    gap = 0.0
    gaps = np.empty(n_steps)

    for t in range(n_steps):
        z = rng.standard_normal()
        pull = alpha * (mu - gap) * dt
        shock = sigma * np.sqrt(dt) * z
        gap += pull + shock
        gaps[t] = gap

    return gaps


if __name__ == "__main__":
    alpha = 0.5
    sigma = 0.02
    dt = 0.01
    duration = 100
    n_steps = int(duration / dt)

    path = simulate_ou(n_steps=n_steps, alpha=alpha, mu=0.0, sigma=sigma, dt=dt, seed=0)
    time = np.arange(1, n_steps + 1) * dt

    plt.plot(time, path)
    plt.xlabel("time (s)")
    plt.ylabel("gap")
    plt.title("Simulated OU gap")
    plt.savefig("paper/notes/ou_gap.png", dpi=150)
    plt.show()

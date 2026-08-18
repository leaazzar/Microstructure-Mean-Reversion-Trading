from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np


@dataclass
class Trade:
    time: float
    gap: float
    action: str            # "buy" or "sell"
    position_before: int   # q_before: +1, 0, or -1
    position_after: int    # q_after:  +1, 0, or -1
    quantity: int          # |q_after - q_before|: 1 for the first entry, 2 for a reversal
    reward: float          # -gap * delta_q - phi * quantity


def run_trading_rule(gap, theta, phi=0.0, initial_position=0, dt=1.0):
    """
    q in {-1, 0, +1}: short one lot, flat, long one lot.

    Starts flat (q=0). The first threshold crossing opens a single-lot
    position. Every crossing after that reverses the position directly
    (long -> short or short -> long) rather than passing through flat, so
    it trades two lots at once: one to close the existing side, one to
    open the opposite side.

    Each trade's reward, marked relative to the efficient price, is
    -gap * delta_q - phi * |delta_q|, where phi is the half-spread cost
    per lot traded.
    """
    position = initial_position
    trades = []

    for t, g in enumerate(gap):
        new_position = None
        if position == 0:
            if g <= -theta:
                new_position = 1
            elif g >= theta:
                new_position = -1
        elif position == -1 and g <= -theta:
            new_position = 1
        elif position == 1 and g >= theta:
            new_position = -1

        if new_position is not None:
            delta_q = new_position - position
            quantity = abs(delta_q)
            reward = -g * delta_q - phi * quantity
            action = "buy" if delta_q > 0 else "sell"
            trades.append(Trade(
                time=t * dt, gap=g, action=action,
                position_before=position, position_after=new_position,
                quantity=quantity, reward=reward,
            ))
            position = new_position

    return trades


def check_alternation(trades):
    for prev, curr in zip(trades, trades[1:]):
        assert prev.action != curr.action, f"consecutive {curr.action} at t={curr.time}"


def timing_stats(trades):
    n_trades = len(trades)
    n_buys = sum(1 for tr in trades if tr.action == "buy")
    n_sells = n_trades - n_buys

    times = [tr.time for tr in trades]
    time_between = np.diff(times) if n_trades > 1 else np.array([])
    avg_time_between = time_between.mean() if len(time_between) else float("nan")

    return {
        "n_trades": n_trades,
        "n_buys": n_buys,
        "n_sells": n_sells,
        "time_between": time_between,
        "avg_time_between": avg_time_between,
    }


def profit_rate(trades, duration):
    total_reward = sum(tr.reward for tr in trades)
    return total_reward, total_reward / duration


def plot_trades(time, gap, theta, trades, path="paper/notes/ou_trades.png"):
    buys = [tr for tr in trades if tr.action == "buy"]
    sells = [tr for tr in trades if tr.action == "sell"]

    plt.figure()
    plt.plot(time, gap, linewidth=0.8, label="gap")
    plt.axhline(theta, color="gray", linestyle="--", linewidth=1, label=f"+theta={theta}")
    plt.axhline(-theta, color="gray", linestyle="--", linewidth=1, label=f"-theta={theta}")
    plt.scatter([tr.time for tr in buys], [tr.gap for tr in buys],
                marker="^", color="green", zorder=5, label="buy")
    plt.scatter([tr.time for tr in sells], [tr.gap for tr in sells],
                marker="v", color="red", zorder=5, label="sell")

    plt.xlabel("time (s)")
    plt.ylabel("gap")
    plt.title(f"OU gap with trades (theta={theta})")
    plt.legend()
    plt.savefig(path, dpi=150)
    plt.show()


def plot_threshold_sweep(gap, thetas, phi, dt, duration, path="paper/notes/ou_threshold_sweep.png"):
    n_trades_list = []
    rates = []
    for th in thetas:
        trades = run_trading_rule(gap, theta=th, phi=phi, dt=dt)
        n_trades_list.append(len(trades))
        _, rate = profit_rate(trades, duration)
        rates.append(rate)

    n_trades_arr = np.array(n_trades_list)
    rates_arr = np.array(rates)
    best_idx = int(np.argmax(rates_arr))

    _, axes = plt.subplots(2, 1, sharex=True, figsize=(7, 7))
    axes[0].plot(thetas, n_trades_arr, marker="o")
    axes[0].set_ylabel("number of trades")
    axes[0].set_title(f"Threshold sweep (phi={phi})")

    axes[1].plot(thetas, rates_arr, marker="o")
    axes[1].scatter([thetas[best_idx]], [rates_arr[best_idx]], color="red", zorder=5,
                     label=f"peak: theta={thetas[best_idx]:.4f}")
    axes[1].axhline(0, color="black", linewidth=0.5)
    axes[1].set_xlabel("theta")
    axes[1].set_ylabel("profit rate (per second)")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.show()

    return n_trades_arr, rates_arr


if __name__ == "__main__":
    from simulate_ou import simulate_ou

    alpha, sigma, dt, duration = 0.5, 0.02, 0.01, 100
    phi = 0.005  # half-spread; placeholder until a real spread value is available
    n_steps = int(duration / dt)
    gap = simulate_ou(n_steps=n_steps, alpha=alpha, mu=0.0, sigma=sigma, dt=dt, seed=0)
    time = np.arange(1, n_steps + 1) * dt

    theta = 0.03
    trades = run_trading_rule(gap, theta=theta, phi=phi, dt=dt)
    check_alternation(trades)
    plot_trades(time, gap, theta, trades)

    stats = timing_stats(trades)
    total_reward, rate = profit_rate(trades, duration)
    print(f"theta={theta}: {stats['n_trades']} trades "
          f"({stats['n_buys']} buys, {stats['n_sells']} sells), "
          f"avg time between trades = {stats['avg_time_between']:.2f}s")
    print("time between consecutive trades:", np.round(stats["time_between"], 2).tolist())
    for tr in trades:
        print(f"  t={tr.time:6.2f}  {tr.action:4s}  q:{tr.position_before:+d}->{tr.position_after:+d}"
              f"  qty={tr.quantity}  reward={tr.reward:+.4f}")
    print(f"total reward = {total_reward:.4f}, profit rate = {rate:.5f} per second")
    print(f"theoretical reward per full reversal = 2*(theta-phi) = {2*(theta-phi):.4f}")

    print("\nthreshold sweep:")
    for th in [0.01, 0.02, 0.03, 0.04]:
        sweep_trades = run_trading_rule(gap, theta=th, phi=phi, dt=dt)
        check_alternation(sweep_trades)
        sweep_stats = timing_stats(sweep_trades)
        sweep_total, sweep_rate = profit_rate(sweep_trades, duration)
        avg = sweep_stats["avg_time_between"]
        avg_str = f"{avg:.2f}s" if not np.isnan(avg) else "n/a"
        print(f"  theta={th:.2f}: {sweep_stats['n_trades']:3d} trades, "
              f"avg wait = {avg_str}, total reward = {sweep_total:.4f}, "
              f"profit rate = {sweep_rate:.5f}/s "
              f"(theory per reversal = {2*(th-phi):.4f})")

    fine_thetas = np.arange(0.005, 0.0525, 0.0025)
    n_trades_arr, rates_arr = plot_threshold_sweep(gap, fine_thetas, phi, dt, duration)
    peak_idx = int(np.argmax(rates_arr))
    print(f"\nfine sweep peak: theta={fine_thetas[peak_idx]:.4f}, "
          f"profit rate={rates_arr[peak_idx]:.5f}/s, trades={n_trades_arr[peak_idx]}")

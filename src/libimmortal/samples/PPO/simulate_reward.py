#!/usr/bin/env python3
# simulate_reward_vectorized.py
#
# Vectorized reward curve visualizer (no reward.py import).
# This emulates the distance-based part of RewardShaper.__call__ using scalar BFS tile distances.

import argparse
import os
from dataclasses import dataclass
from typing import Optional

import numpy as np
import matplotlib.pyplot as plt


# -------------------------
# Config (match your train.py defaults)
# -------------------------
@dataclass
class RewardConfig:
    w_progress: float = 50.0
    w_time: float = 0.01
    w_damage: float = 0.05
    w_not_actionable: float = 0.01
    terminal_failure_penalty: float = 3.0
    terminal_bonus: float = 3.0
    reward_clip: Optional[float] = 20.0

    success_if_raw_reward_ge: Optional[float] = 1.0
    time_limit: Optional[float] = 300.0
    success_speed_bonus: float = 10.0

    use_bfs_progress: bool = True
    w_acceleration: float = 0.2

    # Success threshold based on BFS distance (normalized)
    # RewardShaper logic you pasted earlier used: d_bfs <= 1.5/max_dist
    success_threshold_norm: float = 1.5 / 250.0  # default assumes max_dist=250


# -------------------------
# Vectorized reward calculation
# -------------------------
def compute_rewards_vectorized(
    d_start_tile: np.ndarray,
    d_end_tile: np.ndarray,
    cfg: RewardConfig,
    *,
    max_dist: float,
    raw_reward: float = 0.0,
    damage_delta: float = 0.0,
    is_actionable: float = 1.0,
) -> np.ndarray:
    """
    Compute per-step reward for transitions d_start -> d_end (tile distances).
    - No env, no BFS. We treat BFS distance as a scalar input.
    - This matches the algebraic structure of your pasted RewardShaper.__call__.

    Notes:
    - We assume raw_goal_dist is NOT triggering success here (set to large), so success is BFS-threshold or raw_reward.
    - terminal_failure_penalty depends on `done` in the real code; here we don't simulate failure endings.
    """
    d_start_tile = d_start_tile.astype(np.float32)
    d_end_tile = d_end_tile.astype(np.float32)

    # Normalize
    norm_start = d_start_tile / float(max_dist)
    norm_end = d_end_tile / float(max_dist)

    rewards = np.zeros_like(norm_start, dtype=np.float32)

    # -------------------------
    # 1) Success condition
    # -------------------------
    is_success = (norm_end <= float(cfg.success_threshold_norm))
    if cfg.success_if_raw_reward_ge is not None:
        is_success = is_success | (float(raw_reward) >= float(cfg.success_if_raw_reward_ge))

    # -------------------------
    # 2) Progress & magnet
    # -------------------------
    if cfg.use_bfs_progress:
        # A) Linear progress
        rewards += float(cfg.w_progress) * (norm_start - norm_end)

        # B) Magnet shaping: max(0, exp(-a*d_end) - exp(-a*d_start))
        alpha = 15.0
        phi_prev = np.exp(-alpha * norm_start)
        phi_curr = np.exp(-alpha * norm_end)
        rewards += float(cfg.w_acceleration) * np.maximum(0.0, phi_curr - phi_prev)

    # -------------------------
    # 3) Penalties (one-step approximation)
    # -------------------------
    rewards += -float(cfg.w_time)

    # damage penalty uses delta of cumulative damage in your code:
    # r += w_damage * -max(0, dmg - prev_dmg)
    # If we model per-step delta directly:
    rewards += -float(cfg.w_damage) * max(0.0, float(damage_delta))

    # not-actionable penalty
    if float(is_actionable) < 0.5:
        rewards += -float(cfg.w_not_actionable)

    # -------------------------
    # 4) Clipping + terminal override
    # -------------------------
    if cfg.reward_clip is not None:
        clip = float(cfg.reward_clip)

        # In your RewardShaper:
        # terminal_r = clip * terminal_bonus
        terminal_r = float(cfg.terminal_bonus)

        # Optional speed bonus (implementation-dependent in your codebase).
        # Here we model it as a constant add-on when success triggers.
        # If your reward.py uses remaining-time fraction, change this line accordingly.
        terminal_r = terminal_r + float(cfg.success_speed_bonus)

        # Non-success: clip to [-clip, hi]
        hi = clip if terminal_r > clip else (terminal_r - 1e-3)
        clipped = np.clip(rewards, -clip, hi)

        # Success: override to terminal_r
        final_rewards = np.where(is_success, terminal_r, clipped).astype(np.float32)
    else:
        # If no clipping, your pasted RewardShaper does NOT force terminal_r.
        # So we keep the shaped reward as-is.
        final_rewards = rewards.astype(np.float32)

    return final_rewards


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max_tile", type=int, default=150)

    # Match your train.py flags / defaults
    ap.add_argument("--w_progress", type=float, default=100.0)
    ap.add_argument("--w_time", type=float, default=0.01)
    ap.add_argument("--w_damage", type=float, default=0.1)
    ap.add_argument("--w_not_actionable", type=float, default=0.01)
    ap.add_argument("--terminal_bonus", type=float, default=4.0)
    ap.add_argument("--terminal_failure_penalty", type=float, default=5.0)
    ap.add_argument("--reward_clip", type=float, default=10.0)

    ap.add_argument("--success_if_raw_reward_ge", type=float, default=1.0)
    ap.add_argument("--time_limit", type=float, default=300.0)
    ap.add_argument("--success_speed_bonus", type=float, default=0.5)

    ap.add_argument("--w_acceleration", type=float, default=1.0)

    # Distance model params
    ap.add_argument("--map_h", type=int, default=90)
    ap.add_argument("--map_w", type=int, default=160)
    ap.add_argument("--success_threshold_tiles", type=float, default=1.5)

    # Optional penalties for visualization
    ap.add_argument("--damage_delta", type=float, default=0.0)
    ap.add_argument("--is_actionable", type=float, default=1.0)

    ap.add_argument("--save_dir", type=str, default=None)
    ap.add_argument("--show", action="store_true")
    args = ap.parse_args()

    max_dist = float(args.map_h + args.map_w)  # 250 by default

    cfg = RewardConfig(
        w_progress=float(args.w_progress),
        w_time=float(args.w_time),
        w_damage=float(args.w_damage),
        w_not_actionable=float(args.w_not_actionable),
        terminal_failure_penalty=float(args.terminal_failure_penalty),
        terminal_bonus=float(args.terminal_bonus),
        reward_clip=float(args.reward_clip) if args.reward_clip is not None else None,
        success_if_raw_reward_ge=float(args.success_if_raw_reward_ge) if args.success_if_raw_reward_ge is not None else None,
        time_limit=float(args.time_limit) if args.time_limit is not None else None,
        success_speed_bonus=float(args.success_speed_bonus),
        use_bfs_progress=True,
        w_acceleration=float(args.w_acceleration),
        success_threshold_norm=float(args.success_threshold_tiles) / max_dist,
    )

    tiles = np.arange(0, int(args.max_tile) + 1, dtype=np.float32)

    # Closer / Stay / Farther
    d_end_closer = np.maximum(tiles - 1, 0)
    d_end_stay = tiles
    d_end_farther = np.minimum(tiles + 1, float(args.max_tile))

    r_closer = compute_rewards_vectorized(
        tiles, d_end_closer, cfg, max_dist=max_dist,
        raw_reward=0.0, damage_delta=args.damage_delta, is_actionable=args.is_actionable
    )
    r_stay = compute_rewards_vectorized(
        tiles, d_end_stay, cfg, max_dist=max_dist,
        raw_reward=0.0, damage_delta=args.damage_delta, is_actionable=args.is_actionable
    )
    r_farther = compute_rewards_vectorized(
        tiles, d_end_farther, cfg, max_dist=max_dist,
        raw_reward=0.0, damage_delta=args.damage_delta, is_actionable=args.is_actionable
    )

    # Save dir
    if args.save_dir:
        os.makedirs(args.save_dir, exist_ok=True)

    # Plot per-step
    plt.figure(figsize=(10, 5))
    plt.plot(tiles, r_closer, label="Closer (d -> d-1)", linewidth=2)
    plt.plot(tiles, r_stay, label="Stay (d -> d)", linestyle="--", alpha=0.7)
    plt.plot(tiles, r_farther, label="Farther (d -> d+1)", linestyle=":", alpha=0.7)
    plt.axhline(0, linewidth=0.5)
    plt.axvline(x=float(args.success_threshold_tiles), linestyle="--", alpha=0.35, label=f"Success threshold (<= {args.success_threshold_tiles:.2f} tiles)")

    plt.xlabel("Current BFS Tile Distance")
    plt.ylabel("Per-step Reward")
    plt.title(
        "Per-step Reward (Vectorized)\n"
        f"w_progress={cfg.w_progress}, w_acc={cfg.w_acceleration}, w_time={cfg.w_time}, clip={cfg.reward_clip}, term_bonus={cfg.terminal_bonus}, speed_bonus={cfg.success_speed_bonus}"
    )
    plt.legend()
    plt.grid(True, alpha=0.3)
    if args.save_dir:
        plt.savefig(os.path.join(args.save_dir, "vectorized_step_reward.png"), dpi=150, bbox_inches="tight")

    # Cumulative return (always closer) with early stop on success threshold
    max_t = int(args.max_tile)

    # transitions k -> k-1 for k=1..max_t
    path_start = np.arange(1, max_t + 1, dtype=np.float32)
    path_end = path_start - 1.0
    step_rewards = compute_rewards_vectorized(
        path_start, path_end, cfg, max_dist=max_dist,
        raw_reward=0.0, damage_delta=args.damage_delta, is_actionable=args.is_actionable
    )

    # terminal step if end <= threshold
    is_terminal_step = (path_end / max_dist) <= float(cfg.success_threshold_norm)

    cum_returns = np.zeros(max_t + 1, dtype=np.float32)
    for d in range(1, max_t + 1):
        r_step = step_rewards[d - 1]
        term = bool(is_terminal_step[d - 1])
        if term:
            cum_returns[d] = r_step
        else:
            cum_returns[d] = r_step + cum_returns[d - 1]

    plt.figure(figsize=(10, 5))
    plt.plot(np.arange(max_t + 1), cum_returns, label="Cumulative Return (always closer)")
    plt.xlabel("Start BFS Tile Distance")
    plt.ylabel("Total Return")
    plt.title("Cumulative Return vs Start Distance (Vectorized DP)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    if args.save_dir:
        plt.savefig(os.path.join(args.save_dir, "vectorized_cumulative_return.png"), dpi=150, bbox_inches="tight")

    print("=== simulate_reward_vectorized.py ===")
    print(f"max_dist={max_dist}")
    print(f"success_threshold_tiles={args.success_threshold_tiles} (norm={cfg.success_threshold_norm:.6f})")
    print(f"w_progress={cfg.w_progress}, w_acceleration={cfg.w_acceleration}")
    print(f"w_time={cfg.w_time}, w_damage={cfg.w_damage}, w_not_actionable={cfg.w_not_actionable}")
    print(f"clip={cfg.reward_clip}, terminal_bonus={cfg.terminal_bonus}, success_speed_bonus={cfg.success_speed_bonus}")
    print(f"sample step (50->49): {step_rewards[49]:.4f}")
    print(f"sample terminal step (2->1): {step_rewards[1]:.4f}")
    print(f"max cumulative (d={max_t}): {cum_returns[-1]:.4f}")

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()

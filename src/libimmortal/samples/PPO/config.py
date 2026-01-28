import argparse

def build_argparser():
    p = argparse.ArgumentParser()

    # Immortal env args
    p.add_argument("--game_path", type=str, default=r"/root/immortal_suffering/immortal_suffering_linux_build.x86_64")
    p.add_argument("--port", type=int, default=5005)
    p.add_argument("--port_stride", type=int, default=50)
    p.add_argument("--time_scale", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--width", type=int, default=720)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--verbose", action="store_true")

    # Graphics flags: default is headless (no_graphics=True)
    g = p.add_mutually_exclusive_group()
    g.add_argument("--no_graphics", dest="no_graphics", action="store_true", default=False)
    g.add_argument("--graphics", dest="no_graphics", action="store_false")

    p.add_argument("--unity_timeout_wait", type=int, default=120)
    p.add_argument("--max_env_restarts", type=int, default=20)

    # Debug
    p.add_argument("--player_log_tail", type=int, default=200)

    # Runner steps (PER RANK)
    p.add_argument("--max_steps", type=int, default=2500000)

    # PPO hyperparams
    p.add_argument("--max_ep_len", type=int, default=1500)
    p.add_argument("--update_timestep", type=int, default=6000)
    p.add_argument("--k_epochs", type=int, default=10)
    p.add_argument("--eps_clip", type=float, default=0.2)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--lr_actor", type=float, default=1e-4)
    p.add_argument("--lr_critic", type=float, default=2e-4)
    p.add_argument("--mini_batch_size", type=int, default=64)

    # Action std
    p.add_argument("--action_std", type=float, default=0.6)
    p.add_argument("--action_std_decay_rate", type=float, default=0.05)
    p.add_argument("--min_action_std", type=float, default=0.1)
    p.add_argument("--action_std_decay_freq", type=int, default=250000)

    # Saving (rank0 only)
    p.add_argument("--save_model_freq", type=int, default=40000)

    # Resume
    p.add_argument("--resume", action="store_true")
    p.add_argument("--checkpoint", type=str, default=None)

    # Reward shaping knobs
    p.add_argument("--w_progress", type=float, default=100.0)
    p.add_argument("--w_time", type=float, default=0.02)
    p.add_argument("--w_damage", type=float, default=0.05)
    p.add_argument("--w_not_actionable", type=float, default=0.05)
    p.add_argument("--terminal_bonus", type=float, default=4.0)
    p.add_argument("--reward_clip", type=float, default=15.0)
    p.add_argument("--success_if_raw_reward_ge", type=float, default=1.0)
    p.add_argument("--time_limit", type=float, default=500.0)
    p.add_argument("--success_speed_bonus", type=float, default=3)

    # Debug: dump id_map path (rank0 only, on update steps)
    p.add_argument("--dump_id_map", type=str, default=None)

    # Reward scaling toggle (default OFF)
    p.add_argument("--reward_scaling", action="store_true", default=False)

    # wandb (rank0 only)
    p.add_argument("--wandb", action="store_true")
    p.add_argument("--wandb_project", type=str, default="ppo-immortal")
    p.add_argument("--wandb_name", type=str, default=None)
    p.add_argument("--wandb_log_freq", type=int, default=200)

    # Seed mixing / domain randomization schedule
    p.add_argument("--env_seed_mode", type=str, default="mix",
                choices=["fixed", "random", "mix"],
                help="fixed: always base seed+rank, random: always random, mix: anneal fixed->random per episode/step")

    p.add_argument("--env_seed_base", type=int, default=1234567,
                help="Base for generating random-looking env seeds (deterministic hash).")

    p.add_argument("--env_seed_mix_start", type=float, default=0.0,
                help="Initial random probability p0 in mix mode (0~1).")

    p.add_argument("--env_seed_mix_end", type=float, default=1.0,
                help="Final random probability p1 in mix mode (0~1).")

    p.add_argument("--env_seed_mix_warmup_episodes", type=int, default=50000,
                help="Warmup length for p_random schedule in mix mode, measured in episodes.")

    p.add_argument("--env_seed_mix_schedule", type=str, default="linear",
                choices=["linear", "exp"],
                help="Schedule type for random probability in mix mode.")

    p.add_argument("--env_seed_mix_tau", type=float, default=20000.0,
                help="Time constant for exp schedule (episodes). Only used when --env_seed_mix_schedule=exp.")
    
    p.add_argument("--freeze_backbone", action="store_true", default=False,
                help="Freeze backbone network weights and only train heads.")

    return p
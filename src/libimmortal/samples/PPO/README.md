**How to run**
0. Prerequisites

Use following commands to prepare training.

```
rm -f "/root/.config/unity3d/DefaultCompany/Immortal Suffering/Player.log"
rm -f "/root/.config/unity3d/DefaultCompany/Immortal Suffering/Player-prev.log"

ln -s /dev/null "/root/.config/unity3d/DefaultCompany/Immortal Suffering/Player.log"
ln -s /dev/null "/root/.config/unity3d/DefaultCompany/Immortal Suffering/Player-prev.log"

export CUDA_VISIBLE_DEVICES=1,2,3
export DEVICE=:0
```


1. Without loading checkpoint

Fix seed

```
xvfb-run -a -s "-screen 0 1024x768x24" \
torchrun --standalone --nproc_per_node=3 ./src/libimmortal/samples/PPO/train.py \
  --port 17005 --port_stride 200 \
  --max_ep_len 1500 --update_timestep 6000 --max_steps 1000000\
  --save_model_freq 50000 --wandb \
  --seed 42
  --wandb \
```

Seed mixing

```
xvfb-run -a -s "-screen 0 1024x768x24" \
torchrun --standalone --nproc_per_node=3 ./src/libimmortal/samples/PPO/train.py \
  --port 17005 --port_stride 200 \
  --max_ep_len 1500 --update_timestep 6000 --max_steps 3000000\
  --save_model_freq 50000 --wandb \
  --env_seed_mode mix \
  --env_seed_mix_start 0.50 \
  --env_seed_mix_end 0.50 \
  --env_seed_mix_warmup_episodes 600 \
  --env_seed_mix_schedule linear \
  --env_seed_base 1234567 \
  --wandb \
```

2. Resume checkpoint

Fix seed

```
xvfb-run -a -s "-screen 0 1024x768x24" \
torchrun --standalone --nproc_per_node=3 ./src/libimmortal/samples/PPO/train.py \
  --port 17005 --port_stride 200 \
  --max_ep_len 1500 --update_timestep 6000 --max_steps 1000000\
  --save_model_freq 50000 --wandb \
  --resume --checkpoint /root/libimmortal/src/libimmortal/samples/PPO/checkpoints/PPO_ImmortalSufferingEnv_seed42_550000.pth \
  --seed 42
  --wandb \
```

```
xvfb-run -a -s "-screen 0 1024x768x24" \
torchrun --standalone --nproc_per_node=4 ./src/libimmortal/samples/PPO/train.py \
  --port 17005 --port_stride 200 \
  --max_ep_len 1500 --update_timestep 6000 --max_steps 1000000\
  --save_model_freq 50000 --wandb \
  --resume --checkpoint /root/libimmortal/src/libimmortal/samples/PPO/checkpoints/Necto2_ImmortalSufferingEnv_randseed_2400000.pth \
  --env_seed_mode mix \
  --env_seed_mix_start 1.00 \
  --env_seed_mix_end 1.00 \
  --env_seed_mix_warmup_episodes 500 \
  --env_seed_mix_schedule linear \
  --env_seed_base 1234567 \
  --wandb
```
#!/bin/bash

CUDA_VISIBLE_DEVICES=2 python train.py \
  --policy pretrain_flow \
  --env halfcheetah-gravity \
  --shift_level 0.5 \
  --srctype medium \
  --seed 1 \
  --mode 1 \
  --dir runs \
  --load_model ./calql_hc_medium_seed2_offline_result.pt \
  --dynamics_gap_reward_scale 0.0 &

CUDA_VISIBLE_DEVICES=1 python train.py \
  --policy pretrain_flow \
  --env halfcheetah-gravity \
  --shift_level 0.5 \
  --srctype medium \
  --seed 2 \
  --mode 1 \
  --dir runs \
  --load_model ./calql_hc_medium_seed2_offline_result.pt \
  --dynamics_gap_reward_scale 0.0 &

CUDA_VISIBLE_DEVICES=3 python train.py \
  --policy pretrain_flow \
  --env halfcheetah-gravity \
  --shift_level 0.5 \
  --srctype medium \
  --seed 3 \
  --mode 1 \
  --dir runs \
  --load_model ./calql_hc_medium_seed2_offline_result.pt \
  --dynamics_gap_reward_scale 0.0 &

wait

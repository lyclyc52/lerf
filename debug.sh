#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=2
export TORCH_USE_CUDA_DSA=1
export CUDA_LAUNCH_BLOCKING=1

ns-train samnerf-v2 \
--data /ssddata/yliugu/data/Datasets/teatime \
--save_only_latest_checkpoint False \
--pipeline.use-contrastive False \
--project-name origin_nerf \
--pipeline.feature-type sam \
--pipeline.datamanager.feature_starting_epoch 100000 \
--pipeline.datamanager.supersampling False \
--base-dir origin_nerf \
--pipeline.feature-type sam \
--use-wandb False
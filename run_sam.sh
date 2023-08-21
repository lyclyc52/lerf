#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=1

ns-train samnerf-v2 \
--data /ssddata/yliugu/data/Datasets/teatime \
--save_only_latest_checkpoint False \
--pipeline.use-contrastive False \
--project-name contrastive \
--pipeline.feature-type sam \
--pipeline.datamanager.feature_starting_epoch 1000000 \
--pipeline.datamanager.supersampling False \
--base-dir sam \
--pipeline.feature-type sam \
--use-wandb False
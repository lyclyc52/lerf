#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=1

ns-train samnerf-v2 \
--data /ssddata/yliugu/lerf/data/Datasets/teatime \
--save_only_latest_checkpoint False \
--pipeline.use-contrastive True \
--project-name contrastive \
--pipeline.feature-type sam \
--pipeline.datamanager.contrastive_starting_epoch 5000 \
--use-wandb True
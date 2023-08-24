#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=1

ns-train samnerf-v2 \
--data /ssddata/yliugu/data/Datasets/teatime \
--save_only_latest_checkpoint False \
--pipeline.use-contrastive False \
--project-name contrastive \
--pipeline.feature-type sam \
--pipeline.datamanager.feature_starting_epoch 10000 \
--pipeline.datamanager.supersampling False \
--base-dir sam_0 \
--pipeline.feature-type sam \
--use-wandb True 
# --load_origin_nerf_checkpoint /ssddata/yliugu/lerf/outputs/teatime/samnerf-v2/origin_nerf/nerfstudio_models/step-000029999.ckpt
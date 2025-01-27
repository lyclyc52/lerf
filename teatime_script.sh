#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=2
ns-train samnerf-v2 \
--data /ssddata/yliugu/lerf/data/Datasets/teatime \
--save_only_latest_checkpoint False \
--pipeline.datamanager.disable-contrastive True \
--project-name origin_global_sample \
--use-wandb False \
--logging.local_writer.max_log_size 1 \

# export CUDA_VISIBLE_DEVICES=6
# ns-train lerf --data /disk1/yliugu/lerf/data/nerfstudio/poster --load-dir /disk1/yliugu/lerf/outputs/poster/lerf/2023-06-21_041344/nerfstudio_models

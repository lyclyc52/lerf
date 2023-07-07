#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=7
ns-train samnerf-v2 \
--data /disk1/yliugu/lerf/data/Datasets/waldo_kitchen \
--save_only_latest_checkpoint False \
--pipeline.datamanager.disable-contrastive False \
--project-name contrastive_global_sample
# export CUDA_VISIBLE_DEVICES=6
# ns-train lerf --data /disk1/yliugu/lerf/data/nerfstudio/poster --load-dir /disk1/yliugu/lerf/outputs/poster/lerf/2023-06-21_041344/nerfstudio_models

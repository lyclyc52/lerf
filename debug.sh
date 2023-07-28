#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=0

ns-train samnerf-v2 \
--data /ssddata/yliugu/lerf/data/Datasets/teatime \
--save_only_latest_checkpoint False \
--pipeline.datamanager.disable-contrastive True \
--project-name origin_global_sample \
--pipeline.feature-type sam 
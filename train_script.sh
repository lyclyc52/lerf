#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=1
ns-train samnerf --data data/nerfstudio/poster --load-dir /disk1/yliugu/lerf/outputs/poster/samnerf/pretrain/nerfstudio_models
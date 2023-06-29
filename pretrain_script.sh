#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=1
ns-train samnerf --data data/nerfstudio/poster

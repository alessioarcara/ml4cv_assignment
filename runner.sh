#!/bin/bash

EXPERIMENTS=(
    "configs/base.yaml configs/transforms/full.yaml configs/decoders/fuse_stem.yaml"
    "configs/base.yaml configs/transforms/full.yaml configs/decoders/fpn.yaml"
    "configs/base.yaml configs/transforms/full.yaml configs/decoders/no_aspp.yaml"
    "configs/base.yaml configs/transforms/full.yaml configs/decoders/default_atrous_rates.yaml"
    "configs/base.yaml configs/transforms/full.yaml configs/decoders/fapn_no_dilation.yaml"
)

echo "Found ${#EXPERIMENTS[@]} experiments to run."

for EXP_ARGS in "${EXPERIMENTS[@]}"; do
    uv run python scripts/train.py --configs $EXP_ARGS || true
done
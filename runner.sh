#!/bin/bash

LOCK_FILE="/tmp/gpu_training.lock"

echo "Acquiring lock for GPU training..."

touch "$LOCK_FILE" 2>/dev/null
exec 200<"$LOCK_FILE"

flock 200 || { echo "Failed to acquire lock. Exiting."; exit 1; }

echo "Lock acquired. Starting experiments..."

EXPERIMENTS=(
  # Baseline model 
  #"configs/base.yaml configs/transforms/full.yaml configs/proto_segnet.yaml"
  "configs/base.yaml configs/transforms/full.yaml configs/proto_segnet.yaml configs/ood_proto_segnet.yaml"

  # Encoder ablation
  #"configs/base.yaml configs/transforms/full.yaml configs/encoders/resnet_18d.yaml"
  #"configs/base.yaml configs/transforms/full.yaml configs/encoders/efficientnet_b2.yaml"
  #"configs/base.yaml configs/transforms/full.yaml configs/encoders/convnext_tiny.yaml"
  #"configs/base.yaml configs/transforms/full.yaml configs/encoders/pvtv2_b1.yaml"

  # Decoder ablation
  #"configs/base.yaml configs/transforms/full.yaml configs/decoders/fuse_stem.yaml"
  #"configs/base.yaml configs/transforms/full.yaml configs/decoders/fpn.yaml"
  #"configs/base.yaml configs/transforms/full.yaml configs/decoders/no_aspp.yaml"
  #"configs/base.yaml configs/transforms/full.yaml configs/decoders/default_atrous_rates.yaml"
  #"configs/base.yaml configs/transforms/full.yaml configs/decoders/fapn_no_dilation.yaml"

  # Augmentations ablation
  #"configs/base.yaml configs/decoders/fpn_no_aspp.yaml configs/transforms/full.yaml"
  #"configs/base.yaml configs/decoders/fpn_no_aspp.yaml configs/transforms/no_random_resized_crop.yaml"
  #"configs/base.yaml configs/decoders/fpn_no_aspp.yaml configs/transforms/no_random_resized_crop_with_aspp.yaml"
  #"configs/base.yaml configs/decoders/fpn_no_aspp.yaml configs/transforms/no_occlusion.yaml"
  #"configs/base.yaml configs/decoders/fpn_no_aspp.yaml configs/transforms/occlusion_semantic_inpainting.yaml"
  #"configs/base.yaml configs/decoders/fpn_no_aspp.yaml configs/transforms/no_photometric.yaml"
  #"configs/base.yaml configs/decoders/fpn_no_aspp.yaml configs/transforms/no_weather.yaml"
  #"configs/base.yaml configs/decoders/fpn_no_aspp.yaml configs/transforms/imagenet_normalize.yaml"
  #"configs/base.yaml configs/decoders/fpn_no_aspp.yaml configs/transforms/per_image_normalize.yaml"

  # Contrastive Losses 
  #"configs/base.yaml configs/transforms/ablation_result.yaml configs/decoders/fpn_no_aspp.yaml configs/losses/closed/ptl.yaml"
  #"configs/base.yaml configs/transforms/ablation_result.yaml configs/decoders/fpn_no_aspp.yaml configs/losses/closed/dml.yaml"
  #"configs/base.yaml configs/transforms/ablation_result.yaml configs/decoders/fpn_no_aspp.yaml configs/losses/closed/ow.yaml"
  #"configs/base.yaml configs/transforms/ablation_result.yaml configs/decoders/fpn_no_aspp.yaml configs/losses/open/ptl_objectosphere.yaml"
  #"configs/base.yaml configs/transforms/ablation_result.yaml configs/decoders/fpn_no_aspp.yaml configs/losses/open/dml_objectosphere.yaml"

  #"configs/base.yaml configs/transforms/ablation_result.yaml configs/decoders/fpn_no_aspp.yaml configs/losses/open/ood_bce.yaml"

  # Combo Losses
  #"configs/base.yaml configs/transforms/ablation_result.yaml configs/decoders/fpn_no_aspp.yaml configs/losses/closed/ptl_dice.yaml"
  #"configs/base.yaml configs/transforms/ablation_result.yaml configs/decoders/fpn_no_aspp.yaml configs/losses/closed/ptl_focal.yaml"
  #"configs/base.yaml configs/transforms/ablation_result.yaml configs/decoders/fpn_no_aspp.yaml configs/losses/closed/ptl_focal_dice.yaml"
  #"configs/base.yaml configs/transforms/ablation_result.yaml configs/decoders/fpn_no_aspp.yaml configs/losses/open/ptl_dice_objectosphere.yaml"
  #"configs/base.yaml configs/transforms/ablation_result.yaml configs/decoders/fpn_no_aspp.yaml configs/losses/open/ptl_focal_objectosphere.yaml"
  #"configs/base.yaml configs/transforms/ablation_result.yaml configs/decoders/fpn_no_aspp.yaml configs/losses/open/ptl_focal_dice_objectosphere.yaml"
)

echo "Found ${#EXPERIMENTS[@]} experiments to run."

for EXP_ARGS in "${EXPERIMENTS[@]}"; do
  uv run python scripts/train.py --configs $EXP_ARGS || true
done

echo "Releasing lock for GPU training..."
import numpy as np
import torch
import torch.nn as nn
from transformers.models.mask2former.modeling_mask2former import (
    Mask2FormerConfig,
    Mask2FormerPixelDecoderEncoderOnly,
    Mask2FormerPixelDecoderOutput,
    Mask2FormerSinePositionEmbedding,
)


# Modified from from transformers.models.detr.modeling_deformable_detr.DeformableDetrModel with DeformableDetrModel->Mask2FormerPixelDecoder
class Mask2FormerPixelDecoder(nn.Module):
    def __init__(self, config: Mask2FormerConfig, feature_channels):
        super().__init__()

        self.config = config

        feature_dim = config.feature_size
        mask_dim = config.mask_feature_size
        num_pos_features = feature_dim // 2

        self.position_embedding = Mask2FormerSinePositionEmbedding(
            num_pos_feats=num_pos_features, normalize=True
        )
        self.num_feature_levels = 3
        transformer_in_channels = feature_channels[-self.num_feature_levels :]

        self.transformer_feature_strides = config.feature_strides[
            -self.num_feature_levels :
        ]
        self.feature_channels = feature_channels
        self.level_embed = nn.Parameter(
            torch.Tensor(self.num_feature_levels, feature_dim)
        )

        # Create input projection layers
        if self.num_feature_levels > 1:
            input_projections_list = []
            for in_channels in transformer_in_channels[::-1]:
                input_projections_list.append(
                    nn.Sequential(
                        nn.Conv2d(in_channels, feature_dim, kernel_size=1),
                        nn.GroupNorm(32, feature_dim),
                    )
                )
            self.input_projections = nn.ModuleList(input_projections_list)
        else:
            self.input_projections = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Conv2d(
                            transformer_in_channels[-1], feature_dim, kernel_size=1
                        ),
                        nn.GroupNorm(32, feature_dim),
                    )
                ]
            )

        self.encoder = Mask2FormerPixelDecoderEncoderOnly(config)

        # --- MY CHANGE ---
        # 1. "The feature map x4 is processed with a 1x1 filter-size convolutional layer"
        self.conv_x4 = nn.Conv2d(feature_channels[0], feature_dim, kernel_size=1)
        # 2. "Finally, the output is passed to a 3x3 convolutional layer to produce per-pixel features F"
        self.final_conv = nn.Conv2d(feature_dim, mask_dim, kernel_size=3, padding=1)
        # --- MY CHANGE ---

        self.mask_projection = nn.Conv2d(
            feature_dim, mask_dim, kernel_size=1, stride=1, padding=0
        )

        # Extra FPN levels
        stride = min(self.transformer_feature_strides)
        self.common_stride = config.common_stride
        self.num_fpn_levels = int(np.log2(stride) - np.log2(self.common_stride))

        lateral_convs = []
        output_convs = []

        for idx, in_channels in enumerate(self.feature_channels[: self.num_fpn_levels]):
            lateral_conv = nn.Sequential(
                nn.Conv2d(in_channels, feature_dim, kernel_size=1, bias=False),
                nn.GroupNorm(32, feature_dim),
            )

            output_conv = nn.Sequential(
                nn.Conv2d(
                    feature_dim,
                    feature_dim,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=False,
                ),
                nn.GroupNorm(32, feature_dim),
                nn.ReLU(),
            )
            self.add_module(f"adapter_{idx + 1}", lateral_conv)
            self.add_module(f"layer_{idx + 1}", output_conv)

            lateral_convs.append(lateral_conv)
            output_convs.append(output_conv)

        # Order convolutional layers from low to high resolution
        self.lateral_convolutions = lateral_convs[::-1]
        self.output_convolutions = output_convs[::-1]

    def get_valid_ratio(self, mask, dtype=torch.float32):
        """Get the valid ratio of all feature maps."""

        _, height, width = mask.shape
        valid_height = torch.sum(~mask[:, :, 0], 1)
        valid_width = torch.sum(~mask[:, 0, :], 1)
        valid_ratio_height = valid_height.to(dtype) / height
        valid_ratio_width = valid_width.to(dtype) / width
        valid_ratio = torch.stack([valid_ratio_width, valid_ratio_height], -1)
        return valid_ratio

    def forward(
        self,
        features,
        encoder_outputs=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )

        # Apply 1x1 convolution to reduce the channel dimension to d_model (256 by default)
        input_embeds = []
        position_embeddings = []
        for level, x in enumerate(features[::-1][: self.num_feature_levels]):
            input_embeds.append(self.input_projections[level](x))
            position_embeddings.append(
                self.position_embedding(x.shape, x.device, x.dtype)
            )

        masks = [
            torch.zeros(
                (x.size(0), x.size(2), x.size(3)), device=x.device, dtype=torch.bool
            )
            for x in input_embeds
        ]

        # Prepare encoder inputs (by flattening)
        spatial_shapes_list = [
            (embed.shape[2], embed.shape[3]) for embed in input_embeds
        ]
        input_embeds_flat = torch.cat(
            [embed.flatten(2).transpose(1, 2) for embed in input_embeds], 1
        )
        spatial_shapes = torch.as_tensor(
            spatial_shapes_list, dtype=torch.long, device=input_embeds_flat.device
        )
        masks_flat = torch.cat([mask.flatten(1) for mask in masks], 1)

        position_embeddings = [
            embed.flatten(2).transpose(1, 2) for embed in position_embeddings
        ]
        level_pos_embed_flat = [
            x + self.level_embed[i].view(1, 1, -1)
            for i, x in enumerate(position_embeddings)
        ]
        level_pos_embed_flat = torch.cat(level_pos_embed_flat, 1)

        level_start_index = torch.cat(
            (spatial_shapes.new_zeros((1,)), spatial_shapes.prod(1).cumsum(0)[:-1])
        )
        valid_ratios = torch.stack(
            [
                self.get_valid_ratio(mask, dtype=input_embeds_flat.dtype)
                for mask in masks
            ],
            1,
        )

        # Send input_embeds_flat + masks_flat + level_pos_embed_flat (backbone + proj layer output) through encoder
        if encoder_outputs is None:
            encoder_outputs = self.encoder(
                inputs_embeds=input_embeds_flat,
                attention_mask=masks_flat,
                position_embeddings=level_pos_embed_flat,
                spatial_shapes_list=spatial_shapes_list,
                level_start_index=level_start_index,
                valid_ratios=valid_ratios,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

        last_hidden_state = encoder_outputs.last_hidden_state
        batch_size = last_hidden_state.shape[0]

        # We compute level_start_index_list separately from the tensor version level_start_index
        # to avoid iterating over a tensor which breaks torch.compile/export.
        level_start_index_list = [0]
        for height, width in spatial_shapes_list[:-1]:
            level_start_index_list.append(level_start_index_list[-1] + height * width)
        split_sizes = [None] * self.num_feature_levels
        for i in range(self.num_feature_levels):
            if i < self.num_feature_levels - 1:
                split_sizes[i] = (
                    level_start_index_list[i + 1] - level_start_index_list[i]
                )
            else:
                split_sizes[i] = last_hidden_state.shape[1] - level_start_index_list[i]

        encoder_output = torch.split(last_hidden_state, split_sizes, dim=1)

        # Compute final features
        outputs = [
            x.transpose(1, 2).view(
                batch_size, -1, spatial_shapes_list[i][0], spatial_shapes_list[i][1]
            )
            for i, x in enumerate(encoder_output)
        ]

        # --- MY CHANGE ---
        # Outputs contains processed features f3, f2, f1
        f3 = outputs[0]  # Stride 32
        f1 = outputs[2]  # Stride 8
        x4 = features[0]  # Stride 4

        # 1. "we only pass the last layer f3 to the transformer decoder"
        multi_scale_features = [f3]
        # 2. "feature map x4 is processed with a 1x1 filter"
        x4_processed = self.conv_x4(x4)

        f1_upsampled = nn.functional.interpolate(
            f1,
            size=x4.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

        combined = x4_processed + f1_upsampled
        # "Finally, the output is passed to a 3x3 convolutional layer to produce per-pixel features F"
        mask_features = self.final_conv(combined)

        return Mask2FormerPixelDecoderOutput(
            mask_features=mask_features,
            multi_scale_features=tuple(multi_scale_features),
            attentions=encoder_outputs.attentions,
        )

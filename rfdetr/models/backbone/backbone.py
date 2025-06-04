# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from LW-DETR (https://github.com/Atten4Vis/LW-DETR)
# Copyright (c) 2024 Baidu. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from Conditional DETR (https://github.com/Atten4Vis/ConditionalDETR)
# Copyright (c) 2021 Microsoft. All Rights Reserved.
# ------------------------------------------------------------------------
# Copied from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# ------------------------------------------------------------------------

"""
Backbone modules.
"""
from functools import partial
import torch
import torch.nn.functional as F
from torch import nn
import math
import types

from transformers import AutoModel, AutoProcessor, AutoModelForCausalLM, AutoConfig, AutoBackbone
from peft import LoraConfig, get_peft_model, PeftModel

from rfdetr.util.misc import NestedTensor, is_main_process

from rfdetr.models.backbone.base import BackboneBase
from rfdetr.models.backbone.projector import MultiScaleProjector
from rfdetr.models.backbone.dinov2 import DinoV2

__all__ = ["Backbone"]


class Backbone(BackboneBase):
    """backbone."""
    def __init__(self,
                 name: str,
                 pretrained_encoder: str=None,
                 window_block_indexes: list=None,
                 drop_path=0.0,
                 out_channels=256,
                 out_feature_indexes: list=None,
                 projector_scale: list=None,
                 use_cls_token: bool = False,
                 freeze_encoder: bool = False,
                 layer_norm: bool = False,
                 target_shape: tuple[int, int] = (640, 640),
                 rms_norm: bool = False,
                 backbone_lora: bool = False,
                 gradient_checkpointing: bool = False,
                 load_dinov2_weights: bool = True,
                 ):
        super().__init__()

        self.name = name
        self.target_shape = target_shape
        self.out_feature_indexes = out_feature_indexes if out_feature_indexes is not None else []
        self.encoder = AutoBackbone.from_pretrained(
            "google/vit-large-patch16-224",
            output_hidden_states=True,
            # output_attentions=False,
        )
        vit_hidden_size = self.encoder.config.hidden_size
        if not self.out_feature_indexes:
            self.encoder._out_feature_channels = [vit_hidden_size]
        else:
            self.encoder._out_feature_channels = [vit_hidden_size] * len(self.out_feature_indexes)

        # build encoder + projector as backbone module
        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False

        self.projector_scale = projector_scale
        assert len(self.projector_scale) > 0
        # x[0]
        assert (
            sorted(self.projector_scale) == self.projector_scale
        ), "only support projector scale P3/P4/P5/P6 in ascending order."
        level2scalefactor = dict(P3=2.0, P4=1.0, P5=0.5, P6=0.25)
        scale_factors = [level2scalefactor[lvl] for lvl in self.projector_scale]

        self.projector = MultiScaleProjector(
            in_channels=self.encoder._out_feature_channels,
            out_channels=out_channels,
            scale_factors=scale_factors,
            layer_norm=layer_norm,
            rms_norm=rms_norm,
        )

        self._export = False

    def export(self):
        self._export = True
        self._forward_origin = self.forward
        self.forward = self.forward_export

        if isinstance(self.encoder, PeftModel):
            print("Merging and unloading LoRA weights")
            self.encoder.merge_and_unload()
        
        def _interpolate_pos_encoding_vit(position_embeddings,
                                  patch_size,
                                  height,
                                  width):
            num_original_positions = position_embeddings.shape[1]
            embedding_dim = position_embeddings.shape[-1]

            target_h_patch = height // patch_size
            target_w_patch = width // patch_size
            num_target_patch_positions = target_h_patch * target_w_patch

            cls_pos_embed = None
            patch_pos_embed = position_embeddings

            cls_pos_embed = position_embeddings[:, 0:1, :]
            patch_pos_embed = position_embeddings[:, 1:, :]
            num_original_patch_positions = num_original_positions - 1

            if num_original_patch_positions == num_target_patch_positions and \
            int(math.sqrt(num_original_patch_positions)) == target_h_patch and \
            int(math.sqrt(num_original_patch_positions)) == target_w_patch:
                return position_embeddings

            original_grid_size = int(math.sqrt(num_original_patch_positions))
            if original_grid_size * original_grid_size != num_original_patch_positions:
                raise ValueError(
                    f"Original patch position embeddings ({num_original_patch_positions}) are not a perfect square. "
                    "Cannot reshape to 2D grid for interpolation."
                )

            patch_pos_embed_2d = patch_pos_embed.reshape(
                1, original_grid_size, original_grid_size, embedding_dim
            ).permute(0, 3, 1, 2)

            interpolated_patch_pos_embed_2d = F.interpolate(
                patch_pos_embed_2d,
                size=(target_h_patch, target_w_patch),
                mode="bicubic",
                align_corners=False,
                antialias=True,
            )

            interpolated_patch_pos_embed = interpolated_patch_pos_embed_2d.permute(0, 2, 3, 1).reshape(
                1, -1, embedding_dim
            )

            if cls_pos_embed is not None:
                new_position_embeddings = torch.cat((cls_pos_embed, interpolated_patch_pos_embed), dim=1)
            else:
                new_position_embeddings = interpolated_patch_pos_embed

            return new_position_embeddings
        
        if hasattr(self.encoder, 'embeddings') and hasattr(self.encoder.embeddings, 'position_embeddings'):
            with torch.no_grad():
                new_pos_embed = _interpolate_pos_encoding_vit(
                    self.encoder.embeddings.position_embeddings,
                    self.encoder.config.patch_size,
                    self.target_shape[0],
                    self.target_shape[1],
                )
                self.encoder.embeddings.position_embeddings = nn.Parameter(new_pos_embed, requires_grad=False)

            if hasattr(self.encoder.embeddings, 'interpolate_pos_encoding'):
                original_interpolate_method = self.encoder.embeddings.interpolate_pos_encoding

                def new_interpolate_method(self_mod, embeddings, height, width):
                    num_patches = embeddings.shape[1]
                    num_positions_with_cls = self_mod.position_embeddings.shape[1]

                    target_h_patch = self.target_shape[0] // self_mod.patch_size
                    target_w_patch = self.target_shape[1] // self_mod.patch_size
                    expected_num_patches = target_h_patch * target_w_patch

                    if num_patches == expected_num_patches:
                        return self_mod.position_embeddings[:, 1:, :]

                    return original_interpolate_method(embeddings, height, width)

                self.encoder.embeddings.interpolate_pos_encoding = types.MethodType(
                    new_interpolate_method,
                    self.encoder.embeddings
                )

    def forward(self, tensor_list: NestedTensor):
        """ """
        # (H, W, B, C)
        feats = self.encoder(tensor_list.tensors)
        feats = self.projector(feats)
        # x: [(B, C, H, W)]
        out = []
        for feat in feats:
            m = tensor_list.mask
            assert m is not None
            mask = F.interpolate(m[None].float(), size=feat.shape[-2:]).to(torch.bool)[
                0
            ]
            out.append(NestedTensor(feat, mask))
        return out

    def forward_export(self, tensors: torch.Tensor):
        outputs = self.encoder(pixel_values=tensors, output_hidden_states=True)
        all_hidden_states = outputs.hidden_states
        feats = []

        if not self.out_feature_indexes:
            last_hidden_state = all_hidden_states[-1]
            patch_tokens = last_hidden_state[:, 1:, :]
            B, N, C = patch_tokens.shape
            H_img, W_img = tensors.shape[-2:]
            patch_size = self.encoder.config.patch_size
            H_patch, W_patch = H_img // patch_size, W_img // patch_size
            feat = patch_tokens.reshape(B, H_patch, W_patch, C).permute(0, 3, 1, 2)
            feats.append(feat)
        else:
            for layer_idx in self.out_feature_indexes:
                raw_feat = all_hidden_states[layer_idx]
                patch_tokens = raw_feat[:, 1:, :]
                B, N, C = patch_tokens.shape
                H_img, W_img = tensors.shape[-2:]
                patch_size = self.encoder.config.patch_size
                H_patch, W_patch = H_img // patch_size, W_img // patch_size
                feat = patch_tokens.reshape(B, H_patch, W_patch, C).permute(0, 3, 1, 2)
                feats.append(feat)
        feats = self.projector(feats)
        out_feats = []
        out_masks = []
        for feat in feats:
            # x: [(B, C, H, W)]
            b, _, h, w = feat.shape
            out_masks.append(
                torch.zeros((b, h, w), dtype=torch.bool, device=feat.device)
            )
            out_feats.append(feat)
        return out_feats, out_masks

    def get_named_param_lr_pairs(self, args, prefix: str = ""):
        def _get_named_param_lr_pairs(self, args, prefix=""):
            named_param_lr_pairs = {}
            lr_decay_rate_arg = getattr(args, 'lr_decay_rate', 1.0)
            weight_decay_arg = getattr(args, 'weight_decay', 0.05)

            from rfdetr.util.get_param_dicts import get_vit_lr_decay_rate, get_vit_weight_decay_rate

            for name, param in self.encoder.named_parameters():
                if not param.requires_grad:
                    continue

                lr_decay_factor = get_vit_lr_decay_rate(name, lr_decay_rate_arg, self.num_layers)
                lr = args.lr * lr_decay_factor
                current_weight_decay = get_vit_weight_decay_rate(name, weight_decay_arg)

                named_param_lr_pairs[f"{prefix}.{name}"] = {
                    "params": param,
                    "lr": lr,
                    "weight_decay": current_weight_decay,
                }
            return named_param_lr_pairs

        named_param_lr_pairs = {}
        encoder_prefix = f"{prefix}.encoder" if prefix else "encoder"
        encoder_params = _get_named_param_lr_pairs(args, prefix=encoder_prefix)
        named_param_lr_pairs.update(encoder_params)

        projector_lr = getattr(args, 'lr_projector', args.lr)
        projector_weight_decay = getattr(args, 'weight_decay_projector', args.weight_decay)

        for name, param in self.projector.named_parameters():
            if param.requires_grad:
                full_name = f"{prefix}.projector.{name}" if prefix else f"projector.{name}"
                named_param_lr_pairs[full_name] = {
                    "params": param,
                    "lr": projector_lr,
                    "weight_decay": projector_weight_decay,
                }

        return named_param_lr_pairs

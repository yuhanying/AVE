#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""Video models."""

import torch
import torch.nn as nn
import sys

from .omnivore_model import omnivore_swinB_epic, omnivore_swinB,omnivore_swinB_imagenet21k,omnivore_swinS
from .build import MODEL_REGISTRY

@MODEL_REGISTRY.register()
class Omnivore(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        #self.omni = torch.hub.load("facebookresearch/omnivore", model=cfg.MODEL.ARCH, force_reload=True)
        if cfg.MODEL.ARCH == 'omnivore_swinB_epic':
            self.omni = omnivore_swinB_epic()
        elif cfg.MODEL.ARCH == 'omnivore_swinB':
            self.omni = omnivore_swinB(pretrained=True, load_heads=True)
        elif cfg.MODEL.ARCH == 'omnivore_swinS':
            self.omni = omnivore_swinS(pretrained=True, load_heads=True)
        elif cfg.MODEL.ARCH == "omnivore_swinB_imagenet21k":
            self.omni= omnivore_swinB_imagenet21k()
        else:
            print("no such architecture : ", cfg.MODEL.ARCH)
            sys.exit(1)

        # replace the last head to identity
        self.omni.heads = nn.Identity()
        # print(self.omni)
        #self.register_buffer('verb_matrix',self._get_output_transform_matrix('verb',cfg))
        #self.register_buffer('noun_matrix',self._get_output_transform_matrix('noun',cfg))

    # def _get_output_transform_matrix(self, which_one,cfg):

    #     with open('slowfast/models/omnivore_epic_action_classes.csv') as f:
    #         data = f.read().splitlines()
    #         action2index = {d:i for i,d in enumerate(data)}


    #     if which_one == 'verb':
    #         verb_classes = pd.read_csv(f'{cfg.EPICKITCHENS.ANNOTATIONS_DIR}/EPIC_100_verb_classes.csv',usecols=['key'])
    #         verb2index = {}
    #         for verb in verb_classes['key']:
    #             verb2index[verb] = [v for k,v in action2index.items() if k.split(',')[0]==verb]
    #         matrix = torch.zeros(len(action2index),len(verb2index))
    #         for i, (k,v) in enumerate(verb2index.items()):
    #             for j in v:
    #                 matrix[j,i] = 1.
    #     elif which_one == 'noun':
    #         noun_classes = pd.read_csv(f'{cfg.EPICKITCHENS.ANNOTATIONS_DIR}/EPIC_100_noun_classes.csv',usecols=['key'])
    #         noun2index = {}
    #         for noun in noun_classes['key']:
    #             noun2index[noun] = [v for k,v in action2index.items() if k.split(',')[1]==noun]
    #         matrix = torch.zeros(len(action2index),len(noun2index))
    #         for i, (k,v) in enumerate(noun2index.items()):
    #             for j in v:
    #                 matrix[j,i] = 1.
    #     return matrix


    def forward(self, x):
        y = self.omni(x, input_type="video")
        return y




# @MODEL_REGISTRY.register()
# class Omnivore(nn.Module):
#     def __init__(self, cfg):
#         super().__init__()
#         arch = cfg.MODEL.ARCH
#         # 1. Load base model without heads
#         load_heads = False
#         if arch == 'omnivore_swinB':
#             base = omnivore_swinB(pretrained=True, load_heads=load_heads)
#         elif arch == 'omnivore_swinB_epic':
#             base = omnivore_swinB_epic()
#         elif arch == 'omnivore_swinS':
#             base = omnivore_swinS(pretrained=True, load_heads=load_heads)
#         elif arch == 'omnivore_swinB_imagenet21k':
#             base = omnivore_swinB_imagenet21k()
#         else:
#             raise ValueError(f"Unsupported ARCH: {arch}")
        
#         # 2. Identify trunk module (could be base or base.trunk)
#         trunk = base.trunk if hasattr(base, 'trunk') else base
#                 # 3. Only disable the final layer's PatchMerging downsampling
#         #    depths=[2,2,18,2] so layers[3] is last stage
#         if len(trunk.layers) >= 4 and hasattr(trunk.layers[3], 'downsample'):
#             trunk.layers[3].downsample = nn.Identity()

#         # 4. Replace heads with Identity to avoid pooling
#         heads = base.heads if hasattr(base, 'heads') else nn.Identity()
#         self.trunk = trunk
#         self.heads = nn.Identity()  # ensure no further pooling/classification

#     def forward(self, x: torch.Tensor):
#         # x: (B, 3, T, H, W)
#         # Manually run SwinTransformer3D backbone to get raw feature map
#         # 1. video -> frame tokens
#         x = self.trunk.im2vid(x)
#         # 2. patch embed (RGB)
#         x = self.trunk.patch_embed(x)
#         # 3. patch embed (depth) if exists
#         if hasattr(self.trunk, 'depth_patch_embed'):
#             x = self.trunk.depth_patch_embed(x)
#         # 4. positional dropout
#         x = self.trunk.pos_drop(x)
#         # 5. sequential layers
#         for layer in self.trunk.layers:
#             x = layer(x)
#         # 6. final norm
#         x = self.trunk.norm(x)
#         # x: (B, C, T, H, W)
#         B, C, T, H, W = x.shape
#         # Reshape to (B, T, N_token, C)
#         x = x.permute(0, 2, 3, 4, 1).contiguous()  # (B, T, H, W, C)
#         x = x.view(B, T, H*W, C)                    # (B, T, N_token, C)
#         return x




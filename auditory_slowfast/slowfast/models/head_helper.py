# #!/usr/bin/env python3
# # Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

# """ResNe(X)t Head helper."""

import torch
import torch.nn as nn
import pdb

# class ResNetBasicHead(nn.Module):
#     """
#     ResNe(X)t 2D head.
#     This layer performs a fully-connected projection during training, when the
#     input size is 1x1. It performs a convolutional projection during testing
#     when the input size is larger than 1x1. If the inputs are from multiple
#     different pathways, the inputs will be concatenated after pooling.
#     """

#     def __init__(
#         self,
#         dim_in,
#         num_classes,
#         pool_size,
#         dropout_rate=0.0,
#         act_func="softmax",
#     ):
#         """
#         The `__init__` method of any subclass should also contain these
#             arguments.
#         ResNetBasicHead takes p pathways as input where p in [1, infty].

#         Args:
#             dim_in (list): the list of channel dimensions of the p inputs to the
#                 ResNetHead.
#             num_classes (int): the channel dimensions of the p outputs to the
#                 ResNetHead.
#             pool_size (list): the list of kernel sizes of p frequency temporal
#                 poolings, temporal pool kernel size, frequency pool kernel size in order.
#             dropout_rate (float): dropout rate. If equal to 0.0, perform no
#                 dropout.
#             act_func (string): activation function to use. 'softmax': applies
#                 softmax on the output. 'sigmoid': applies sigmoid on the output.
#         """
#         super(ResNetBasicHead, self).__init__()
#         assert (
#             len({len(pool_size), len(dim_in)}) == 1
#         ), "pathway dimensions are not consistent."
#         self.num_pathways = len(pool_size)

#         for pathway in range(self.num_pathways):
#             #avg_pool = nn.AvgPool2d(pool_size[pathway], stride=1)
#             avg_pool = nn.AdaptiveAvgPool2d((1,1))
#             self.add_module("pathway{}_avgpool".format(pathway), avg_pool)

#         if dropout_rate > 0.0:
#             self.dropout = nn.Dropout(dropout_rate)
#         # Perform FC in a fully convolutional manner. The FC layer will be
#         # initialized with a different std comparing to convolutional layers.
#         if isinstance(num_classes, (list, tuple)):
#             self.projection_verb = nn.Linear(sum(dim_in), num_classes[0], bias=True)
#             self.projection_noun = nn.Linear(sum(dim_in), num_classes[1], bias=True)
#         else:
#             self.projection = nn.Linear(sum(dim_in), num_classes, bias=True)
#         self.num_classes = num_classes
#         # Softmax for evaluation and testing.
#         if act_func == "softmax":
#             self.act = nn.Softmax(dim=3)
#         elif act_func == "sigmoid":
#             self.act = nn.Sigmoid()
#         else:
#             raise NotImplementedError(
#                 "{} is not supported as an activation"
#                 "function.".format(act_func)
#             )

#     def forward(self, inputs):
#         assert (
#             len(inputs) == self.num_pathways
#         ), "Input tensor does not contain {} pathway".format(self.num_pathways)
#         pool_out = []
#         for pathway in range(self.num_pathways):
#             m = getattr(self, "pathway{}_avgpool".format(pathway))
#             pool_out.append(m(inputs[pathway]))
#         x = torch.cat(pool_out, 1)
#         # (N, C, T, H) -> (N, T, H, C).
#         x = x.permute((0, 2, 3, 1))
#         # Perform dropout.
#         if hasattr(self, "dropout"):
#             x = self.dropout(x)
#         if isinstance(self.num_classes, (list, tuple)):
#             x_v = self.projection_verb(x)
#             x_n = self.projection_noun(x)

#             # Performs fully convlutional inference.
#             if not self.training:
#                 x_v = self.act(x_v)
#                 x_v = x_v.mean([1, 2])

#             x_v = x_v.view(x_v.shape[0], -1)

#             # Performs fully convlutional inference.
#             if not self.training:
#                 x_n = self.act(x_n)
#                 x_n = x_n.mean([1, 2])

#             x_n = x_n.view(x_n.shape[0], -1)
#             return (x_v, x_n), x.mean([1,2])
#         else:
#             x_feature = torch.clone(x)
#             x = self.projection(x)
#             # Performs fully convlutional inference.
#             if not self.training:
#                 x = self.act(x)
#                 x = x.mean([1, 2])

#             x = x.view(x.shape[0], -1)
#             return x, x_feature.mean([1,2])

class ResNetBasicHead(nn.Module):
    def __init__(self, dim_in, num_classes, pool_size, dropout_rate=0.0, act_func="softmax"):
        super().__init__()
        # 1) 选一个「公共」的时–频分辨率，比如 (16,16)
        #    注意：慢路 (T_s=13,F=4) 跟快路 (T_f=50,F=4) 都会被拉到同一个 (16,16)
        self.T_out, self.F_out = 4,4

        # 2) 把所有 pathway 的 pool 换成这个固定分辨率
        for p in range(len(dim_in)):
            pool = nn.AdaptiveAvgPool2d((self.T_out, self.F_out))
            self.add_module(f"pathway{p}_pool", pool)

        # 3) 分类头相关（不变）
        total_c = sum(dim_in)
        if isinstance(num_classes, (list,tuple)) and len(num_classes)==2:
            self.proj_v = nn.Linear(total_c, num_classes[0])
            self.proj_n = nn.Linear(total_c, num_classes[1])
            self.dual   = True
        else:
            nc = num_classes if isinstance(num_classes,int) else num_classes[0]
            self.proj   = nn.Linear(total_c, nc)
            self.dual   = False

        if dropout_rate>0:
            self.dropout = nn.Dropout(dropout_rate)
        self.act = nn.Softmax(dim=-1) if act_func=="softmax" else nn.Sigmoid()

    def forward(self, inputs):
        B = inputs[0].shape[0]
        # —— 1. 池化到同一个 (T_out,F_out)，再沿通道拼接 —— 
        pooled = []
        for p, x in enumerate(inputs):
            m = getattr(self, f"pathway{p}_pool")
            pooled.append(m(x))          # all become (B, C_p, T_out, F_out)
        x_cat = torch.cat(pooled, dim=1) # (B, C_s+C_f, T_out, F_out)
        C, T, F = x_cat.shape[1], x_cat.shape[2], x_cat.shape[3]

        # —— 2. 生成 patch tokens —— 
        # (B, C, T_out, F_out) -> (B, T_out*F_out, C)
        patch_tokens = x_cat.permute(0,2,3,1).reshape(B, T*F, C)

        # —— 3. 分类分支（可选） —— 
        # global pool + linear
        pooled_vec = x_cat.mean(dim=[2,3])  # (B, C_s+C_f)
        if hasattr(self, "dropout"):
            pooled_vec = self.dropout(pooled_vec)

        if self.dual:
            v = self.proj_v(pooled_vec); n = self.proj_n(pooled_vec)
            if not self.training:
                v = self.act(v); n = self.act(n)
            logits = (v, n)
        else:
            logits = self.proj(pooled_vec)
            if not self.training:
                logits = self.act(logits)

        return logits, patch_tokens

# import torch
# import torch.nn as nn

# class ResNetBasicHead(nn.Module):
#     def __init__(self, dim_in, num_classes, dropout_rate=0.0, act_func="softmax"):
#         super().__init__()
#         self.num_pathways   = len(dim_in)
#         self.total_channels = sum(dim_in)
#         # 1) 平均池化用於分類
#         for p in range(self.num_pathways):
#             self.add_module(f"pathway{p}_avgpool", nn.AdaptiveAvgPool2d((1,1)))
#         # 2) Dropout（分類專用）
#         if dropout_rate>0:
#             self.dropout = nn.Dropout(dropout_rate)
#         # 3) 分類 head
#         if isinstance(num_classes,(list,tuple)) and len(num_classes)==2:
#             self.is_dual = True
#             self.proj_v   = nn.Linear(self.total_channels, num_classes[0])
#             self.proj_n   = nn.Linear(self.total_channels, num_classes[1])
#         else:
#             self.is_dual = False
#             nc = num_classes[0] if isinstance(num_classes,(list,tuple)) else num_classes
#             self.proj     = nn.Linear(self.total_channels, nc)
#         # 4) 激活
#         self.act = nn.Softmax(dim=-1) if act_func=="softmax" else nn.Sigmoid()

#     def forward(self, inputs):
#         """
#         inputs: list of pathway tensors
#           pathway0: (B, C_s, T_s, F)
#           pathway1: (B, C_f, T_f, F)
#         """
#         B = inputs[0].shape[0]

#         ####─ 分支一：分類用 ─────────────────────────────────────
#         pool_out = []
#         for p in range(self.num_pathways):
#             m = getattr(self, f"pathway{p}_avgpool")
#             pool_out.append(m(inputs[p]))           # (B, C_p, 1, 1)
#         x_cls = torch.cat(pool_out, dim=1)         # (B, C_s+C_f, 1, 1)
#         x_cls = x_cls.view(B, -1)                  # (B, C_s+C_f)
#         if hasattr(self, "dropout"):
#             x_cls = self.dropout(x_cls)

#         if self.is_dual:
#             v = self.proj_v(x_cls)                 # (B, num_v)
#             n = self.proj_n(x_cls)                 # (B, num_n)
#             if not self.training:
#                 v = self.act(v); n = self.act(n)
#             logits = (v,n)
#         else:
#             logits = self.proj(x_cls)              # (B, num_classes)
#             if not self.training:
#                 logits = self.act(logits)

#         ####─ 分支二：patch‐token 用 ───────────────────────────────
#         token_list = []
#         for p in range(self.num_pathways):
#             feat = inputs[p]                        # shape = (B, C_p, T_p, F)
#             B, C, T, F = feat.shape
#             # (B, C, T, F) → (B, T*F, C)
#             tokens = feat.permute(0,2,3,1).reshape(B, T*F, C)
#             token_list.append(tokens)

#         # slow+fast tokens 串在一起： (B, N_s+N_f, C_s/C_f 可不同)
#         # 如果要同一維度可再加個線性投影，但這裡先原封不動 concat
#         patch_tokens = torch.cat(token_list, dim=1)  # (B, T_s*F + T_f*F, ?)

#         return logits, patch_tokens

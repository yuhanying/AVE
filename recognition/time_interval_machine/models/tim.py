import torch
import random

from torch import nn

import time_interval_machine.models.helpers.head as head
import time_interval_machine.utils.logging as logging


from time_interval_machine.models.helpers.encodings import AudioVisualFeatureEncoding, VisualFeatureEncoding, AudioFeatureEncoding
from time_interval_machine.models.helpers.transformers import TransformerEncoder, TransformerEncoderLayer
from time_interval_machine.models.helpers.pool import AVGA


logger = logging.get_logger(__name__)

class TIM(nn.Module):
    def __init__(self,
                num_class,
                visual_input_dim=1024,
                audio_input_dim=2304,
                feat_drop=0.5,
                seq_drop=0.5,
                d_model=512,
                feedforward_scale=4,
                nhead=8,
                num_layers=6,
                enc_dropout=0.1,
                input_modality="audio_visual",
                data_modality="audio_visual",
                num_feats=50,
                include_verb_noun=True,
                pool_features=False
            ):
        super(TIM, self).__init__()

        self.input_modality = input_modality
        self.data_modality = data_modality

        self.visual_input_dim = visual_input_dim
        self.audio_input_dim = audio_input_dim
        self.feat_drop=feat_drop
        self.seq_drop = seq_drop

        self.d_model = d_model
        self.dim_feedforward = d_model*feedforward_scale
        self.nhead = nhead
        self.num_layers = num_layers
        self.enc_dropout = enc_dropout

        self.num_feats = num_feats
        self.num_class = num_class
        self.include_verb_noun = include_verb_noun
        self.pool_features = pool_features

        logger.info("Building {} Transformer with {}-D, {} heads, and {} layers.".format(
                                                            self.input_modality,
                                                            2*self.d_model,
                                                            self.nhead,
                                                            self.num_layers
                                                        )
                                                    )
        self._create_model()

    def _create_model(self):
        self.time_mlp = nn.Sequential(
                            nn.Linear(2, self.d_model),
                            nn.ReLU(),
                            nn.Linear(self.d_model, self.d_model),
                            nn.ReLU(),
                            nn.Linear(self.d_model, self.d_model),
                            nn.ReLU(),
                            nn.LayerNorm(self.d_model)
                        )
        
        if self.input_modality == "audio_visual":
            self.feature_encoding = AudioVisualFeatureEncoding(
                                        self.visual_input_dim,
                                        self.audio_input_dim,
                                        self.d_model,
                                        self.feat_drop,
                                        self.seq_drop,
                                        self.num_feats,
                                        self.data_modality,
                                        self.include_verb_noun
                                    )
            self.num_feats *= 2
        elif self.input_modality == "visual":
            self.feature_encoding = VisualFeatureEncoding(
                                        self.visual_input_dim,
                                        self.d_model,
                                        self.feat_drop,
                                        self.seq_drop,
                                        self.num_feats,
                                        self.include_verb_noun
                                    )

        else:
            self.feature_encoding = AudioFeatureEncoding(
                                        self.audio_input_dim,
                                        self.d_model,
                                        self.feat_drop,
                                        self.seq_drop,
                                        self.num_feats
                                    )


        if self.data_modality == "audio_visual":
            self.cls_head = head.AudioVisualCLSHead(self.num_class, 2*self.d_model)
        elif self.data_modality == "visual":
            self.cls_head = head.VisualCLSHead(self.num_class[0], 2*self.d_model)
        else:
            self.cls_head = head.AudioCLSHead(self.num_class[1], 2*self.d_model)

        encoder_layer = TransformerEncoderLayer(
                            d_model=2*self.d_model,
                            nhead=self.nhead,
                            dim_feedforward=self.dim_feedforward,
                            dropout=self.enc_dropout,
                            activation='gelu'
                        )

        self.transformer_encoder = TransformerEncoder(
                                        encoder_layer,
                                        num_layers=self.num_layers
                                    )

        # For MLP
        self.drloc_mlp = nn.Sequential(
                                nn.Linear(4*self.d_model, self.d_model),
                                nn.ReLU(),
                                nn.Linear(self.d_model, self.d_model),
                                nn.ReLU(),
                                nn.Linear(self.d_model, 1)
                            )

        if self.pool_features:
            self.pool = AVGA(
                    a_dim=self.audio_input_dim,
                    v_dim=self.visual_input_dim,
                    hidden_size=self.visual_input_dim
            )
        else:
            self.pool = None
    def make_symmetric_causal_cross_modal_mask(self,Na: int, Nv: int, device=None) -> torch.Tensor:
        """
        Returns a [Na+Nv × Na+Nv] bool mask where
        - mask[i,j]=True 代表「位置 i 可以 attend 到位置 j」
        - audio 位置 i (0 <= i < Na)    : can attend to audio[0..i]  &  visual[0..i]
        - visual 位置 i (0 <= i < Nv)   : can attend to visual[0..i] &  audio[0..i]
        最終回傳 allow_mask，shape = [Na+Nv, Na+Nv]
        """
        L = Na + Nv
        allow = torch.zeros((L, L), dtype=torch.bool, device=device)
        # — audio positions —
        for i in range(Na):
            allow[i, 0 : i+1] = True
            allow[i, Na : Na + min(i+1, Nv)] = True
        # — visual positions —
        for vi in range(Nv):
            out_pos = Na + vi
            allow[out_pos, Na : Na + vi+1] = True
            allow[out_pos, 0 : min(vi+1, Na)] = True
        return allow  # True = 可以 attend


    def forward_encoder(
                self,
                inputs,
                time_encodings,
                num_v_queries,
                num_a_queries
            ):
        
        if self.pool_features:
            inputs[0] = self.pool(inputs[1], inputs[0])

        # Project features to lower dim and include time and modality encodings
        x = self.feature_encoding(inputs, time_encodings, num_v_queries, num_a_queries) # Shape: [S, B, C]
        # print(x.shape)
        # S, B, _ = x.shape

        # # 2) 建立 allow_mask（只對前面 feature token 作 causal cross-modal）
        # Na = self.num_feats//2  # or 你 audio feature 的數量
        # Nv = self.num_feats//2  # or 你 visual feature 的數量
        # device = x.device

        # allow_feat = self.make_symmetric_causal_cross_modal_mask(Na, Nv, device=device)
        # # allow_feat shape = [Na+Nv, Na+Nv]

        # # 3) 把「allow」轉成 Transformer 要的「mask」（True = 要遮住）
        # feat_mask = ~allow_feat  # shape=(num_feats, num_feats)

        # # 4) 整合成整體序列的 full_mask
        # num_feats = Na + Nv
        # # print(f"num_feats: {num_feats}")
        # num_queries = S - num_feats
        # full_mask = torch.zeros((S, S), dtype=torch.bool, device=device)

        # # 4a) 前 num_feats × num_feats，套上我們的 cross-modal causal mask
        # full_mask[:num_feats, :num_feats] = feat_mask

        # # 4b) queries vs. all tokens → 如果你希望 queries attend 全部（或自由決定），這裡填 False
        # full_mask[num_feats:, :] = False   # query 可以看到所有
        # full_mask[:, num_feats:] = False   # 所有可以看到 query（或改成你自己的策略）

        # # 5) expand 到 (B*nhead, S, S)
        # #    這邊沿用你原本的 repeat_interleave 方式
        # mask_batch = full_mask.unsqueeze(0) \
        #                     .repeat_interleave(self.nhead * B, dim=0)  # (B*nhead, S, S)

        # 6) 最後丟進 TransformerEncoder
        #    如果你用的是 PyTorch 原生 nn.TransformerEncoder, 遞參可能叫 src_mask 或者 attn_mask
        # x, _ = self.transformer_encoder(x, src_mask=mask_batch)

        # 7) 後續處理
        cls_scores = self.cls_head(x, num_v_queries, num_a_queries)
        return cls_scores,x      
        # x = self.feature_encoding(inputs, time_encodings, num_v_queries, num_a_queries) # Shape: [S, B, C]

        # masks = torch.ones(size=(x.size(0), x.size(0)), device=x.device)
        # masks[:, :self.num_feats] = 0.
        # masks = masks.fill_diagonal_(0.)

        # masks = masks.unsqueeze(0)
        # masks = masks.repeat_interleave(self.nhead*x.size(1), dim=0).bool()  # Masks Shape: [B*n, S, S]

        # x, _ = self.transformer_encoder(x, src_mask=masks)          # Shape: [B, S, C]

        # cls_scores = self.cls_head(x, num_v_queries, num_a_queries) # Shape: [B, Nq, C]

        # return (cls_scores, x[:, :self.num_feats]) 

    def forward(self,
                inputs,
                forward_type,
                time_encodings=None,
                num_v_queries=None,
                num_a_queries=None
            ):
        if forward_type == "time_mlp":
            # inputs: [B, T, 2]，假設 T == self.num_feats
            # print(f"inputs size:{inputs.shape}")
            raw = inputs[:, :self.num_feats, :]  # 不經過 MLP 的前 25 個
            # print(f"raw size:{raw.shape}")
            proc = inputs[:, self.num_feats:, :] 
            # if inputs.size(1) >self.num_feats:
            #     proc = self.time_mlp(inputs[:, self.num_feats:, :])  # 只處理後面的
            return raw, proc


        elif forward_type == "encoder":
            return self.forward_encoder(
                                    inputs,
                                    time_encodings,
                                    num_v_queries,
                                    num_a_queries
                                )
        elif forward_type == "drloc_mlp":
            return self.drloc_mlp(inputs).squeeze(2)


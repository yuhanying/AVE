import torch


from torch.nn.init import normal_
from torch import nn
from .timechat import AV_Qformer
class CrossAttentionFusion(nn.Module):
    def __init__(self, embed_dim, nhead=2):
        """
        embed_dim: 輸入特徵通道數，這邊預期為 2*d_model (例如 1024)
        """
        super(CrossAttentionFusion, self).__init__()
        self.cross_attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=nhead)
    def forward(self, query, key, value):
        # query, key, value: [T, B, embed_dim]
        attn_output, _ = self.cross_attn(query, key, value)
        # 殘差連接
        fused = query + attn_output
        return fused
    
class VisualFeatureEncoding(nn.Module):
    def __init__(self,
                visual_input_dim,
                d_model,
                feat_drop=0.5,
                seq_drop=0.5,
                num_feats=50,
                include_verb_noun=True
            ):
        super(VisualFeatureEncoding, self).__init__()
        self.include_verb_noun = include_verb_noun
        # Visual encoding modules
        self.visual_embedder = nn.Sequential(
                                        nn.Dropout(p=feat_drop),
                                        nn.Linear(visual_input_dim, d_model),
                                        nn.GELU(),
                                        nn.LayerNorm(d_model)
                                    )
        # CLS tokens for video
        self.action_cls = nn.Parameter(torch.empty((1, 1, d_model), requires_grad=True))
        normal_(self.action_cls, std=0.01)
        if self.include_verb_noun:
            self.verb_cls = nn.Parameter(torch.empty((1, 1, d_model), requires_grad=True))
            self.noun_cls = nn.Parameter(torch.empty((1, 1, d_model), requires_grad=True))
            normal_(self.verb_cls, std=0.01)
            normal_(self.noun_cls, std=0.01)
        self.dropout = nn.Dropout(p=seq_drop)
        self.num_feats = num_feats
        self.timechat=AV_Qformer
    def forward(self, inputs, time_encodings, num_v_queries, num_a_queries):
        batch_size = inputs[0].shape[0]
        # Project audio and visual features to a lower dim
        feat_embed = self.visual_embedder(inputs[0])
        fused_vis = self.timechat(feat_embed,time_encodings[0])
        # print(f"fused_vis:{fused_vis.shape}")
        ##fused_vis:torch.Size([8, 25, 768])

        feat_embed = fused_vis 
        # seq = torch.cat([feat_embed, time_encodings[0]], dim=-1)
        # print(f"feat_embed:{feat_embed.shape}")
        # Query-Related Tokens
        query_time_encoding = time_encodings[1]
        if self.include_verb_noun:
            verb_cls = self.verb_cls.expand(batch_size, num_v_queries, -1)
            noun_cls = self.noun_cls.expand(batch_size, num_v_queries, -1)
            verb_cls = torch.cat(
                    [verb_cls, query_time_encoding],
                    dim=-1
                )
            noun_cls = torch.cat(
                    [noun_cls, query_time_encoding],
                    dim=-1
                )
            seq = torch.cat([seq, verb_cls, noun_cls], dim=1)
        action_cls = self.action_cls.expand(batch_size, num_v_queries, -1)
        action_cls = torch.cat(
                [action_cls, query_time_encoding],
                dim=-1
            )
        # print(f"action_cls:{action_cls.shape}")
        seq = torch.cat([feat_embed, action_cls], dim=1)
        seq = self.dropout(seq)
        seq = seq.transpose(0, 1).contiguous() # [S, B ,C]
        return seq

# class VisualFeatureEncoding(nn.Module):
#     def __init__(self,
#                 visual_input_dim,
#                 d_model,
#                 feat_drop=0.5,
#                 seq_drop=0.5,
#                 num_feats=50,
#                 include_verb_noun=True
#             ):
#         super(VisualFeatureEncoding, self).__init__()

#         self.include_verb_noun = include_verb_noun

#         # Visual encoding modules
#         self.visual_embedder = nn.Sequential(
#                                         nn.Dropout(p=feat_drop),
#                                         nn.Linear(visual_input_dim, d_model),
#                                         nn.GELU(),
#                                         nn.LayerNorm(d_model)
#                                     )

#         # CLS tokens for video
#         self.action_cls = nn.Parameter(torch.empty((1, 1, d_model), requires_grad=True))
#         normal_(self.action_cls, std=0.01)
#         if self.include_verb_noun:
#             self.verb_cls = nn.Parameter(torch.empty((1, 1, d_model), requires_grad=True))
#             self.noun_cls = nn.Parameter(torch.empty((1, 1, d_model), requires_grad=True))
#             normal_(self.verb_cls, std=0.01)
#             normal_(self.noun_cls, std=0.01)

#         self.dropout = nn.Dropout(p=seq_drop)

#         self.num_feats = num_feats

#     def forward(self, inputs, time_encodings, num_v_queries, num_a_queries):
#         batch_size = inputs[0].shape[0]

#         # Project audio and visual features to a lower dim
#         feat_embed = self.visual_embedder(inputs[0])
#         seq = torch.cat([feat_embed, time_encodings[:, :self.num_feats, :]], dim=-1)

#         # Query-Related Tokens
#         query_time_encoding = time_encodings[:, self.num_feats:, :]

#         if self.include_verb_noun:
#             verb_cls = self.verb_cls.expand(batch_size, num_v_queries, -1)
#             noun_cls = self.noun_cls.expand(batch_size, num_v_queries, -1)
#             verb_cls = torch.cat(
#                     [verb_cls, query_time_encoding],
#                     dim=-1
#                 )
#             noun_cls = torch.cat(
#                     [noun_cls, query_time_encoding],
#                     dim=-1
#                 )

#             seq = torch.cat([seq, verb_cls, noun_cls], dim=1)

#         action_cls = self.action_cls.expand(batch_size, num_v_queries, -1)
#         action_cls = torch.cat(
#                 [action_cls, query_time_encoding],
#                 dim=-1
#             )

#         seq = torch.cat([seq, action_cls], dim=1)

#         seq = self.dropout(seq)
#         seq = seq.transpose(0, 1).contiguous() # [S, B ,C]
#         return seq

class AudioFeatureEncoding(nn.Module):
    def __init__(self,
                audio_input_dim,
                d_model,
                feat_drop=0.5,
                seq_drop=0.5,
                num_feats=50
            ):
        super(AudioFeatureEncoding, self).__init__()
        # Audio encoding modules
        self.audio_embedder = nn.Sequential(
                                        nn.Dropout(p=feat_drop),
                                        nn.Linear(audio_input_dim, d_model),
                                        nn.GELU(),
                                        nn.LayerNorm(d_model)
                                    )

        # CLS tokens for audio
        self.action_cls = nn.Parameter(torch.empty((1, 1, d_model), requires_grad=True))
        normal_(self.action_cls, std=0.01)

        self.dropout = nn.Dropout(p=seq_drop)

        self.num_feats = num_feats


    def forward(self, inputs, time_encodings, num_v_queries, num_a_queries):
        batch_size = inputs[1].shape[0]

        # Project audio and visual features to a lower dim
        feat_embed = self.audio_embedder(inputs[1])
        seq = torch.cat([feat_embed, time_encodings[:, :self.num_feats, :]], dim=-1)

        # Query-Releated Tokens
        query_time_encoding = time_encodings[:, self.num_feats:, :]
        action_cls = self.action_cls.expand(batch_size, num_a_queries, -1)
        action_cls = torch.cat(
            [action_cls, query_time_encoding],
            dim=-1
        )

        seq = torch.cat([seq, action_cls], dim=1)

        seq = self.dropout(seq)
        seq = seq.transpose(0, 1).contiguous() # [S, B ,C]
        return seq

class AudioVisualFeatureEncoding(nn.Module):
    def __init__(self,
                visual_input_dim,
                audio_input_dim,
                d_model,
                feat_drop=0.5,
                seq_drop=0.5,
                num_feats=50,
                data_modality="audio_visual",
                include_verb_noun=True
            ):
        super(AudioVisualFeatureEncoding, self).__init__()

        self.data_modality = data_modality
        self.include_verb_noun = include_verb_noun

        # Visual encoding modules
        self.visual_embedder = nn.Sequential(
                                        nn.Dropout(p=feat_drop),
                                        nn.Linear(visual_input_dim, d_model),
                                        nn.GELU(),
                                        nn.LayerNorm(d_model)
                                    )

        # Audio encoding modules
        self.audio_embedder = nn.Sequential(
                                        nn.Dropout(p=feat_drop),
                                        nn.Linear(audio_input_dim, d_model),
                                        nn.GELU(),
                                        nn.LayerNorm(d_model)
                                    )

        # Modality encodings
        self.visual_modality_encoding = nn.Parameter(torch.empty((1, 1, 2*d_model), requires_grad=True))
        self.audio_modality_encoding = nn.Parameter(torch.empty((1, 1, 2*d_model), requires_grad=True))
        normal_(self.visual_modality_encoding, std=0.01)
        normal_(self.audio_modality_encoding, std=0.01)

        # CLS tokens for video
        if "visual" in self.data_modality:
            self.visual_action_cls = nn.Parameter(torch.empty((1, 1, d_model), requires_grad=True))
            normal_(self.visual_action_cls, std=0.01)

            if self.include_verb_noun:
                self.visual_verb_cls = nn.Parameter(torch.empty((1, 1, d_model), requires_grad=True))
                self.visual_noun_cls = nn.Parameter(torch.empty((1, 1, d_model), requires_grad=True))
                normal_(self.visual_verb_cls, std=0.01)
                normal_(self.visual_noun_cls, std=0.01)

        # CLS tokens for audio
        if "audio" in self.data_modality:
            self.audio_action_cls = nn.Parameter(torch.empty((1, 1, d_model), requires_grad=True))
            normal_(self.audio_action_cls, std=0.01)

        self.dropout = nn.Dropout(p=seq_drop)
        self.num_feats = num_feats

        self.timechat=AV_Qformer()
        # self.timechat_query=AV_Qformer_query()
        # self.vis_time_proj = nn.Linear(768, 384)
        # self.aud_time_proj = nn.Linear(768, 384)
        # self.feats_out_proj = nn.Linear(1536, 384)
        self.cross_attn_fusion_feats_v = CrossAttentionFusion(embed_dim=768)
        self.cross_attn_fusion_feats_a = CrossAttentionFusion(embed_dim=768)


    def forward(self, inputs, time_encodings, num_v_queries, num_a_queries):
        batch_size = inputs[0].shape[0]
        # print(time_encodings.shape)
        query_time_encoding = time_encodings[1]
        # visual_action_cls = self.visual_action_cls.expand(batch_size, num_v_queries, -1)
        # visual_action_cls = (
        #     torch.cat(
        #         [visual_action_cls, query_time_encoding[:, :num_v_queries]],
        #         dim=-1
        #     )
        #     + self.visual_modality_encoding)
        
        # # print(f"query_time_encoding:{query_time_encoding.shape}")
        # # print(f"visual_action_cls:{visual_action_cls.shape}")
        # ##query_time_encoding:torch.Size([2, 40, 384])

        # audio_action_cls = self.audio_action_cls.expand(batch_size, num_a_queries, -1)
        # audio_action_cls = (
        #     torch.cat(
        #         [audio_action_cls, query_time_encoding[:, -num_a_queries:]],
        #         dim=-1
        #     )
        #     + self.audio_modality_encoding)
        


        # Project audio and visual features to a lower dim
        vis_embed = self.visual_embedder(inputs[0])
        # print(vis_embed.shape)
        #torch.Size([8, 25, 256])
        aud_embed = self.audio_embedder(inputs[1])

        # Tag inputs with positional and modality encodings (concatenation)
        # vis_embed = (torch.cat(
        #                     [vis_embed, time_encodings[:, :self.num_feats, :]],
        #                     dim=-1
        #                 )
        #             + self.visual_modality_encoding)
        vis_embed = self.timechat.encode_videoQformer_visual(vis_embed,time_encodings[0][:, :self.num_feats, :])
        # print(f"vis_out:{vis_embed.shape}")
        # print(f'time_encodings[0].shape:{time_encodings[0].shape}') #time_encodings[0].shape:torch.Size([4, 40, 2])
        # print(f'time_encodings[0].shape:{time_encodings[1].shape}') #time_encodings[0].shape:torch.Size([4, 40, 384])
        # print(f"vis_out:{vis_out.shape}")
        # vis_out=vis_out+self.visual_modality_encoding
        aud_embed = self.timechat.encode_audioQformer(aud_embed,time_encodings[0][:, self.num_feats:2*self.num_feats, :])
        # print(f"aud_out:{aud_embed.shape}")
        # print(f"aud_out:{aud_out.shape}")
        # aud_out=aud_out+self.audio_modality_encoding
        # vis_embed = self.timechat(vis_embed,time_encodings[0][:, :self.num_feats, :])
        # print(f"fused_vis:{fused_vis.shape}")
        ##fused_vis:torch.Size([8, 25, 768])
        B ,Nf, Tq, D= vis_embed.shape
        vis_embed = vis_embed.reshape(B * Nf, Tq, D).transpose(0, 1)
        aud_embed = aud_embed.reshape(B * Nf, Tq, D).transpose(0, 1)
        # print(f"vis_embed:{vis_embed.shape}")
        vis_embed = self.cross_attn_fusion_feats_v(vis_embed, aud_embed, aud_embed)  # [10, B, 2*d_model]
        aud_embed = self.cross_attn_fusion_feats_a(aud_embed, vis_embed, vis_embed)
        vis_embed = vis_embed.transpose(0, 1).reshape(B, Nf, Tq, D) 
        aud_embed = aud_embed.transpose(0, 1).reshape(B, Nf, Tq, D)
        vis_out=self.timechat.encode_slide_videoQformer_visual(vis_embed,query_time_encoding[:, :num_v_queries])
        aud_out=self.timechat.encode_slide_audioQformer(aud_embed,query_time_encoding[:, -num_a_queries:])

        seq = torch.cat([vis_out, aud_out], dim=1) ##(16,40,768)

        # feats_out = torch.cat([vis_out, aud_out], dim=-1) ##(16,40,768)
        # feats_out=feats_out.unsqueeze(2)
        # feats_out=self.feats_out_proj(feats_out)

        # vis_out=vis_out.unsqueeze(2)
        # vis_out=self.vis_time_proj(vis_out)
        # aud_out=aud_out.unsqueeze(2)
        # aud_out=self.aud_time_proj(aud_out)

        # Query-Related Tokens
     

        # vis_query_out = self.timechat_query.encode_videoQformer_visual(feats_out,query_time_encoding[:, :num_v_queries])+ self.visual_modality_encoding
        # # print(f"vis_query_out:{vis_query_out.shape}")
        # seq= torch.cat([seq, vis_query_out], dim=1)
        # aud_query_out = self.timechat_query.encode_audioQformer(feats_out,query_time_encoding[:, -num_a_queries:])+ self.audio_modality_encoding

        # seq= torch.cat([seq, aud_query_out], dim=1)
    
        # if "visual" in self.data_modality and num_v_queries > 0:
        #     if self.include_verb_noun:
        #         visual_verb_cls = self.visual_verb_cls.expand(batch_size, num_v_queries, -1)
        #         visual_noun_cls = self.visual_noun_cls.expand(batch_size, num_v_queries, -1)

        #         visual_verb_cls = (
        #             torch.cat(
        #                 [visual_verb_cls, query_time_encoding[:, :num_v_queries]],
        #                 dim=-1
        #             )
        #             + self.visual_modality_encoding)
        #         visual_noun_cls = (
        #             torch.cat(
        #                 [visual_noun_cls, query_time_encoding[:, :num_v_queries]],
        #                 dim=-1
        #             )
        #             + self.visual_modality_encoding)

        #         seq = torch.cat([seq, visual_verb_cls, visual_noun_cls], dim=1)


        #     visual_action_cls = self.visual_action_cls.expand(batch_size, num_v_queries, -1)
        #     visual_action_cls = (
        #         torch.cat(
        #             [visual_action_cls, query_time_encoding[:, :num_v_queries]],
        #             dim=-1
        #         )
        #         + self.visual_modality_encoding)

        #     seq = torch.cat([seq, visual_action_cls], dim=1)

        # if "audio" in self.data_modality and num_a_queries > 0:
        #     audio_action_cls = self.audio_action_cls.expand(batch_size, num_a_queries, -1)
        #     audio_action_cls = (
        #         torch.cat(
        #             [audio_action_cls, query_time_encoding[:, -num_a_queries:]],
        #             dim=-1
        #         )
        #         + self.audio_modality_encoding)

        #     seq = torch.cat([seq, audio_action_cls], dim=1)

        # seq = self.dropout(seq)
        # seq = seq.transpose(0, 1).contiguous() # [S, B ,C]
        return seq
# class AudioVisualFeatureEncoding(nn.Module):
#     def __init__(self,
#                 visual_input_dim,
#                 audio_input_dim,
#                 d_model,
#                 feat_drop=0.5,
#                 seq_drop=0.5,
#                 num_feats=50,
#                 data_modality="audio_visual",
#                 include_verb_noun=True
#             ):
#         super(AudioVisualFeatureEncoding, self).__init__()

#         self.data_modality = data_modality
#         self.include_verb_noun = include_verb_noun

#         # Visual encoding modules
#         self.visual_embedder = nn.Sequential(
#                                         nn.Dropout(p=feat_drop),
#                                         nn.Linear(visual_input_dim, d_model),
#                                         nn.GELU(),
#                                         nn.LayerNorm(d_model)
#                                     )

#         # Audio encoding modules
#         self.audio_embedder = nn.Sequential(
#                                         nn.Dropout(p=feat_drop),
#                                         nn.Linear(audio_input_dim, d_model),
#                                         nn.GELU(),
#                                         nn.LayerNorm(d_model)
#                                     )

#         # Modality encodings
#         self.visual_modality_encoding = nn.Parameter(torch.empty((1, 1, 2*d_model), requires_grad=True))
#         self.audio_modality_encoding = nn.Parameter(torch.empty((1, 1, 2*d_model), requires_grad=True))
#         normal_(self.visual_modality_encoding, std=0.01)
#         normal_(self.audio_modality_encoding, std=0.01)

#         # CLS tokens for video
#         if "visual" in self.data_modality:
#             self.visual_action_cls = nn.Parameter(torch.empty((1, 1, d_model), requires_grad=True))
#             normal_(self.visual_action_cls, std=0.01)

#             if self.include_verb_noun:
#                 self.visual_verb_cls = nn.Parameter(torch.empty((1, 1, d_model), requires_grad=True))
#                 self.visual_noun_cls = nn.Parameter(torch.empty((1, 1, d_model), requires_grad=True))
#                 normal_(self.visual_verb_cls, std=0.01)
#                 normal_(self.visual_noun_cls, std=0.01)

#         # CLS tokens for audio
#         if "audio" in self.data_modality:
#             self.audio_action_cls = nn.Parameter(torch.empty((1, 1, d_model), requires_grad=True))
#             normal_(self.audio_action_cls, std=0.01)

#         self.dropout = nn.Dropout(p=seq_drop)
#         self.num_feats = num_feats


#     def forward(self, inputs, time_encodings, num_v_queries, num_a_queries):
#         batch_size = inputs[0].shape[0]


#         # Project audio and visual features to a lower dim
#         vis_embed = self.visual_embedder(inputs[0])
#         aud_embed = self.audio_embedder(inputs[1])

#         # Tag inputs with positional and modality encodings (concatenation)
#         vis_embed = (torch.cat(
#                             [vis_embed, time_encodings[:, :self.num_feats, :]],
#                             dim=-1
#                         )
#                     + self.visual_modality_encoding)

#         aud_embed = (torch.cat(
#                             [aud_embed, time_encodings[:, self.num_feats:2*self.num_feats, :]],
#                             dim=-1
#                         )
#                     + self.audio_modality_encoding)

#         seq = torch.cat([vis_embed, aud_embed], dim=1)

#         # Query-Related Tokens
#         query_time_encoding = time_encodings[:, 2*self.num_feats:]

#         if "visual" in self.data_modality and num_v_queries > 0:
#             if self.include_verb_noun:
#                 visual_verb_cls = self.visual_verb_cls.expand(batch_size, num_v_queries, -1)
#                 visual_noun_cls = self.visual_noun_cls.expand(batch_size, num_v_queries, -1)

#                 visual_verb_cls = (
#                     torch.cat(
#                         [visual_verb_cls, query_time_encoding[:, :num_v_queries]],
#                         dim=-1
#                     )
#                     + self.visual_modality_encoding)
#                 visual_noun_cls = (
#                     torch.cat(
#                         [visual_noun_cls, query_time_encoding[:, :num_v_queries]],
#                         dim=-1
#                     )
#                     + self.visual_modality_encoding)

#                 seq = torch.cat([seq, visual_verb_cls, visual_noun_cls], dim=1)


#             visual_action_cls = self.visual_action_cls.expand(batch_size, num_v_queries, -1)
#             visual_action_cls = (
#                 torch.cat(
#                     [visual_action_cls, query_time_encoding[:, :num_v_queries]],
#                     dim=-1
#                 )
#                 + self.visual_modality_encoding)

#             seq = torch.cat([seq, visual_action_cls], dim=1)

#         if "audio" in self.data_modality and num_a_queries > 0:
#             audio_action_cls = self.audio_action_cls.expand(batch_size, num_a_queries, -1)
#             audio_action_cls = (
#                 torch.cat(
#                     [audio_action_cls, query_time_encoding[:, -num_a_queries:]],
#                     dim=-1
#                 )
#                 + self.audio_modality_encoding)

#             seq = torch.cat([seq, audio_action_cls], dim=1)

#         seq = self.dropout(seq)
#         seq = seq.transpose(0, 1).contiguous() # [S, B ,C]
#         return seq
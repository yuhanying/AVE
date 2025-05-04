

import logging
import math
import random

import torch
from torch.cuda.amp import autocast as autocast
import torch.nn as nn


# from .blip2 import Blip2Base, disabled_train
# from timechat.models.Qformer import BertEncoder
# from transformers.models.bert.modeling_bert import BertEncoder
from transformers import BertTokenizer
import einops
import copy
from .Qformer_casual import BertConfig, BertLMHeadModel

def disabled_train(self, mode=True):
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    return self


class AV_Qformer(nn.Module):

    @classmethod
    def init_tokenizer(cls):
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        tokenizer.add_special_tokens({"bos_token": "[DEC]"})
        return tokenizer

    @classmethod
    def init_video_Qformer(cls, num_query_token, vision_width, 
                           num_hidden_layers=2, causal_encoder=False):
        encoder_config = BertConfig.from_pretrained("bert-base-uncased")
        if num_hidden_layers > 0:
            encoder_config.num_hidden_layers = num_hidden_layers
            encoder_config.cross_attention_freq = 2
            encoder_config.causal_encoder = causal_encoder
        else:
            encoder_config.cross_attention_freq = 2
        encoder_config.encoder_width = vision_width
        # insert cross-attention layer every other block
        encoder_config.add_cross_attention = True
        encoder_config.query_length = num_query_token
        Qformer = BertLMHeadModel(config=encoder_config)
        query_tokens = nn.Parameter(
            torch.zeros(1, num_query_token, encoder_config.hidden_size)
        )
        query_tokens.data.normal_(mean=0.0, std=encoder_config.initializer_range)
        return Qformer, query_tokens
    # def init_video_Qformer(cls, num_query_token, vision_width, num_hidden_layers=2):
    #     encoder_config = BertConfig.from_pretrained("bert-base-uncased")
    #     encoder_config.num_hidden_layers = num_hidden_layers
    #     encoder_config.encoder_width = vision_width
    #     # insert cross-attention layer every other block
    #     encoder_config.add_cross_attention = True
    #     encoder_config.cross_attention_freq = 1
    #     encoder_config.query_length = num_query_token
    #     Qformer = BertLMHeadModel(config=encoder_config)
    #     query_tokens = nn.Parameter(
    #         torch.zeros(1, num_query_token, encoder_config.hidden_size)
    #     )
    #     query_tokens.data.normal_(mean=0.0, std=encoder_config.initializer_range)
    #     return Qformer, query_tokens
    def __init__(
        self,
        # frame-level 與 sliding-window 的 token 數量
        num_frame_query_token=20,
        num_slide_query_token=20,
        num_query_token=20,
        shared_feature_width=384,
        # 視覺／音頻 feature 維度
        video_width=384,
        audio_width=384,
        # sliding_window 是否用 causal decoder
        causal_slide: bool = True,
        # frame_level 通常不用 causal
        causal_frame: bool = False,
        q_former_model="https://storage.googleapis.com/sfr-vision-language-research/LAVIS/models/InstructBLIP/instruct_blip_vicuna7b_trimmed.pth",
        frozen_vid_fQformer=False,
        frozen_vid_sQformer=False,
        frozen_aud_fQformer=False,
        frozen_aud_sQformer=False,
        frozen_share_fQformer=False,
        use_ind_qformer=True,

        max_frame_pos=25,

        qformer_text_input=True,

    ):
        super().__init__()

        self.tokenizer = self.init_tokenizer()

        
        print("loading frame encoder q former")
        # 1) Frame-Level Q-Former for Visual

        if use_ind_qformer:
            # self.frame_vis_Qformer, self.frame_vis_qtok = self.init_Qformer(
            #     num_frame_query_token, video_width
            # )
            self.frame_vis_Qformer, self.frame_vis_qtok = self.init_video_Qformer(
            num_query_token=num_frame_query_token,
            vision_width=video_width,
            num_hidden_layers=8,
            causal_encoder=causal_frame
            )

            if not qformer_text_input:
                self.frame_vis_Qformer.bert.embeddings.word_embeddings = None
                self.frame_vis_Qformer.bert.embeddings.position_embeddings = None
                for layer in self.frame_vis_Qformer.bert.encoder.layer:
                    layer.output = None
                    layer.intermediate = None
            else:
                print("use text input for Qformer")
                self.frame_vis_Qformer.resize_token_embeddings(len(self.tokenizer))
            self.frame_vis_Qformer.cls = None
            self.qformer_text_input = qformer_text_input
            # self.load_from_pretrained(url_or_filename=q_former_model)
            # 2) Frame-Level Q-Former for Audio
          
            if frozen_vid_fQformer:
                for name, param in self.frame_vis_Qformer.named_parameters():
                    param.requires_grad = False
                self.frame_vis_Qformer = self.frame_vis_Qformer.eval()
                self.frame_vis_Qformer.train = disabled_train
                self.frame_vis_qtok.requires_grad = False
                logging.info("freeze Qformer")


            # self.frame_aud_Qformer, self.frame_aud_qtok = self.init_Qformer(
            #     num_frame_query_token, audio_width
            # )
            self.frame_aud_Qformer, self.frame_aud_qtok = self.init_video_Qformer(
                num_query_token=num_frame_query_token,
                vision_width=audio_width,
                num_hidden_layers=8,
                causal_encoder=causal_frame
            )   
            if not qformer_text_input:
                self.frame_aud_Qformer.bert.embeddings.word_embeddings = None
                self.frame_aud_Qformer.bert.embeddings.position_embeddings = None
                for layer in self.frame_aud_Qformer.bert.encoder.layer:
                    layer.output = None
                    layer.intermediate = None
            else:
                print("use text input for Qformer") 
                self.frame_aud_Qformer.resize_token_embeddings(len(self.tokenizer))
            self.frame_aud_Qformer.cls = None
            self.qformer_text_input = qformer_text_input
            # self.load_from_pretrained(url_or_filename=q_former_model)


            if frozen_aud_fQformer:
                for name, param in self.frame_aud_Qformer.named_parameters():
                    param.requires_grad = False
                self.frame_aud_Qformer = self.frame_aud_Qformer.eval()
                self.frame_aud_Qformer.train = disabled_train
                self.frame_aud_qtok.requires_grad = False
                logging.info("freeze Qformer")
        else:
            self.shared_frame_Qformer, self.shared_frame_qtok = self.init_Qformer(
                num_query_token, shared_feature_width
            )
            if not qformer_text_input:
                    self.shared_frame_Qformer.bert.embeddings.word_embeddings = None
                    self.shared_frame_Qformer.bert.embeddings.position_embeddings = None
                    for layer in self.shared_frame_Qformer.bert.encoder.layer:
                        layer.output = None
                        layer.intermediate = None
            else:
                print("use text input for Qformer") 
                self.shared_frame_Qformer.resize_token_embeddings(len(self.tokenizer))
            self.shared_frame_Qformer.cls = None
            self.qformer_text_input = qformer_text_input
            # self.load_from_pretrained(url_or_filename=q_former_model)


            if frozen_share_fQformer:
                for name, param in self.shared_frame_Qformer.named_parameters():
                    param.requires_grad = False
                self.shared_frame_Qformer = self.shared_frame_Qformer.eval()
                self.shared_frame_Qformer.train = disabled_train
                self.shared_frame_qtok.requires_grad = False
                logging.info("freeze Qformer")
        logging.info('Loading frame Q-Former Done')

        print("loading wondow encoder q former")
        self.vis_frame_position_embedding = nn.Embedding(max_frame_pos, self.frame_vis_Qformer.config.hidden_size)
        # 3) Sliding-Window Q-Former for Visual
        self.slide_vis_Qformer, self.slide_vis_qtok = self.init_video_Qformer(
            num_query_token=num_slide_query_token,
            vision_width=self.frame_vis_Qformer.config.hidden_size,
            num_hidden_layers=4,
            causal_encoder=causal_slide
        )
        # self.slide_vis_Qformer, self.slide_vis_qtok = self.init_video_Qformer(
        #     num_query_token=num_slide_query_token,
        #     vision_width=self.frame_vis_Qformer.config.hidden_size,
        # )  
        if not qformer_text_input:
            self.slide_vis_Qformer.bert.embeddings.word_embeddings = None
            self.slide_vis_Qformer.bert.embeddings.position_embeddings = None
            for layer in self.slide_vis_Qformer.bert.encoder.layer:
                layer.output = None
                layer.intermediate = None
        else:
            print("use text input for Qformer") 
            self.slide_vis_Qformer.resize_token_embeddings(len(self.tokenizer))
 

        # self.slide_vis_Qformer.cls = None
        # self.slide_vis_Qformer.bert.embeddings.word_embeddings = None
        # self.slide_vis_Qformer.bert.embeddings.position_embeddings = None
        # for layer in self.slide_vis_Qformer.bert.encoder.layer:
        #     layer.output = None
        #     layer.intermediate = None

        if frozen_vid_sQformer:
            for name, param in self.slide_vis_Qformer.named_parameters():
                param.requires_grad = False
            for name, param in self.vis_frame_position_embedding.named_parameters():
                param.requires_grad = False
            self.slide_vis_qtok.requires_grad = False

            logging.info('video_fQformer is frozen')
        else:
            for name, param in self.slide_vis_Qformer.named_parameters():
                param.requires_grad = True
            for name, param in self.vis_frame_position_embedding.named_parameters():
                param.requires_grad = True
            self.slide_vis_qtok.requires_grad = True
            logging.info('video_fQformer is not frozen')


        self.aud_frame_position_embedding = nn.Embedding(max_frame_pos, self.frame_aud_Qformer.config.hidden_size)
        # 4) Sliding-Window Q-Former for Audio
        self.slide_aud_Qformer, self.slide_aud_qtok = self.init_video_Qformer(
            num_query_token=num_slide_query_token,
            vision_width=self.frame_aud_Qformer.config.hidden_size,
            num_hidden_layers=4,
            causal_encoder=causal_slide
        )
        # self.slide_aud_Qformer, self.slide_aud_qtok = self.init_video_Qformer(
        #     num_query_token=num_slide_query_token,
        #     vision_width=self.frame_aud_Qformer.config.hidden_size,
        # )
        if not qformer_text_input:
            self.slide_aud_Qformer.bert.embeddings.word_embeddings = None
            self.slide_aud_Qformer.bert.embeddings.position_embeddings = None
            for layer in self.slide_aud_Qformer.bert.encoder.layer:
                layer.output = None
                layer.intermediate = None
        else:
            print("use text input for Qformer") 
            self.slide_aud_Qformer.resize_token_embeddings(len(self.tokenizer))


        # self.slide_aud_Qformer.cls = None
        # self.slide_aud_Qformer.bert.embeddings.word_embeddings = None
        # self.slide_aud_Qformer.bert.embeddings.position_embeddings = None
        # for layer in self.slide_aud_Qformer.bert.encoder.layer:
        #     layer.output = None
        #     layer.intermediate = None

        if frozen_aud_sQformer:
            #  todo frozen  llama_proj
            for name, param in self.slide_aud_Qformer.named_parameters():
                param.requires_grad = False
            for name, param in self.aud_frame_position_embedding.named_parameters():
                param.requires_grad = False
            self.slide_aud_qtok.requires_grad = False

            logging.info('audio_Qformer is frozen')
        else:
            for name, param in self.slide_aud_Qformer.named_parameters():
                param.requires_grad = True
            for name, param in self.aud_frame_position_embedding.named_parameters():
                param.requires_grad = True
            self.slide_aud_qtok.requires_grad = True
            logging.info('audio_Qformer is not frozen')

        # 为 sinusoidal PE 分配一维度：这里我们把 hidden_size 平分给 s 和 e
        Dq = self.slide_aud_Qformer.config.hidden_size
        assert Dq % 2 == 0, "hidden_size 必须是偶数"
        self.time_pe_dim = Dq // 2

    def sinusoidal_time_pe(self, times: torch.Tensor):
        """
        times: [B, Nq, 2] (start, end)
        输出: [B, Nq, 2 * time_pe_dim]
        """
        B, Nq, _ = times.shape
        device = times.device

        pe_list = []
        for idx in range(2):  # 分别对 start 和 end
            t = times[..., idx].unsqueeze(-1)  # [B,Nq,1]
            dim = self.time_pe_dim
            # 计算角度： [dim] = 1 / (10000^(2i/dim))
            inv_freq = 1.0 / (
                10000 ** (torch.arange(0, dim, 2, device=device).float() / dim)
            )
            angles = t  * inv_freq  # broadcast to [B,Nq,dim/2]
            # 拼 sin,cos
            pe = torch.cat([angles.sin(), angles.cos()], dim=-1)  # [B,Nq,dim]
            pe_list.append(pe)

        return torch.cat(pe_list, dim=-1)  # [B,Nq,2*dim = Dq]


    def encode_videoQformer_visual(self, feats_vis: torch.Tensor, time_windows: list):
        """
        Args:
            feats_vis: Tensor[B, Nf, T, D]，Omnivore 输出的视觉特征
                       D=1024，T=16（temporal tokens）
            time_windows: List[B][Nf] of (start,end)
        Returns:
            fused_feats: Tensor[B, Nf, D_q]
        """
        B, Nf, T, D = feats_vis.shape
        device = feats_vis.device

        image_embeds=einops.rearrange(feats_vis, 'b t q h -> (b t) q h')
        image_mask = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(device)
        # print(f"image_embeds:{image_embeds.shape} | image_mask:{image_mask.shape}")

        # 2) 创建 prompt （跟你原本一样）
        prompts = [
            f"frame timestamp is from {s:.3f} to {e:.3f}."
            for b in range(B) for (s, e) in time_windows[b]
        ]
        # print(prompts[1])

        toks = self.tokenizer(prompts,
                            return_tensors="pt",
                            padding="longest",
                            max_length=32,
                            truncation=True).to(device)
        

        input_ids, text_mask = toks.input_ids, toks.attention_mask  # (B*Nf, L)
        # tokens = self.tokenizer.convert_ids_to_tokens(input_ids[1].tolist()) ##看一下prompt被切成怎樣
        # print(tokens)
        # print(f"input_ids:{input_ids.shape} | text_mask:{text_mask.shape}")

        # 3) 文本 embedding + [CLS]
        text_embs   = self.frame_vis_Qformer.get_input_embeddings()(input_ids)  # (B*Nf, L, D)  ##token->向量
        text_queries= text_embs[:, :1, :]                             # (B*Nf, 1, D)

        # 4) learnable Q tokens
        q_tokens = self.frame_vis_qtok.expand(B * Nf, -1, -1)           # (B*Nf, Q, D)
        q_mask   = torch.ones(q_tokens.size()[:-1], dtype=torch.long,
                              device=device)                         # (B*Nf, Q)

        # 5) 拼接 Q＋[CLS]（可选）
        if self.qformer_text_input:
            q_embeds = torch.cat([q_tokens, text_queries], dim=1)     # (B*Nf, Q+1, D)
            attn_mask= torch.cat([q_mask, text_mask], dim=1)         # (B*Nf, Q+1)
        else:
            q_embeds, attn_mask = q_tokens, q_mask

        # 6) Cross‐attention：encoder_hidden_states 现在是 (B*Nf, T, D)
        out = self.frame_vis_Qformer.bert(
            input_ids=input_ids,                     # 不再给它原本的 input_ids
            attention_mask=attn_mask,
            query_embeds=q_tokens,              # Q + optional CLS
            encoder_hidden_states=image_embeds,   # 16 时序 tokens
            encoder_attention_mask=image_mask,
            return_dict=True
        )
        # print(f"input_ids:{input_ids.shape} | q_tokens:{q_tokens.shape} )")

        q_hidden_state = out.last_hidden_state # (B*Nf, D_q) 
        # print(f"q_hidden_state:{q_hidden_state.shape}")
        # print(f"q_hidden_state:{q_hidden_state.shape}")

        # add frame_pos embedding
        position_ids = torch.arange(Nf, dtype=torch.long, device=q_tokens.device)
        position_ids = position_ids.unsqueeze(0).expand(B, -1)
        frame_position_embeddings = self.vis_frame_position_embedding(position_ids)


        


        ##sliding window qformer casual
        frame_position_embeddings = frame_position_embeddings.unsqueeze(-2)
        frame_hidden_state = einops.rearrange(q_hidden_state, '(b t) q h -> b t q h', b=B, t=Nf)
        frame_hidden_state = frame_position_embeddings + frame_hidden_state

        return frame_hidden_state
    
    def encode_videoQformer_visual_fuse(self, feats_vis: torch.Tensor, time_windows: list):
        """
        Args:
            feats_vis: Tensor[B, Nf, T, D]，Omnivore 输出的视觉特征
                       D=1024，T=16（temporal tokens）
            time_windows: List[B][Nf] of (start,end)
        Returns:
            fused_feats: Tensor[B, Nf, D_q]
        """
        B, Nf, T, D = feats_vis.shape
        device = feats_vis.device

        image_embeds=einops.rearrange(feats_vis, 'b t q h -> (b t) q h')
        image_mask = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(device)
        # print(f"image_embeds:{image_embeds.shape} | image_mask:{image_mask.shape}")

        # 2) 创建 prompt （跟你原本一样）
        prompts = [
            f"frame timestamp is from {s:.3f} to {e:.3f}."
            for b in range(B) for (s, e) in time_windows[b]
        ]
        # print(prompts[1])

        toks = self.tokenizer(prompts,
                            return_tensors="pt",
                            padding="longest",
                            max_length=32,
                            truncation=True).to(device)
        

        input_ids, text_mask = toks.input_ids, toks.attention_mask  # (B*Nf, L)
        # tokens = self.tokenizer.convert_ids_to_tokens(input_ids[1].tolist()) ##看一下prompt被切成怎樣
        # print(tokens)
        # print(f"input_ids:{input_ids.shape} | text_mask:{text_mask.shape}")

        # 3) 文本 embedding + [CLS]
        text_embs   = self.shared_frame_Qformer.get_input_embeddings()(input_ids)  # (B*Nf, L, D)
        text_queries= text_embs[:, :1, :]                             # (B*Nf, 1, D)

        # 4) learnable Q tokens
        q_tokens = self.shared_frame_qtok.expand(B * Nf, -1, -1)           # (B*Nf, Q, D)
        q_mask   = torch.ones(q_tokens.size()[:-1], dtype=torch.long,
                              device=device)                         # (B*Nf, Q)

        # 5) 拼接 Q＋[CLS]（可选）
        if self.qformer_text_input:
            q_embeds = torch.cat([q_tokens, text_queries], dim=1)     # (B*Nf, Q+1, D)
            attn_mask= torch.cat([q_mask, text_mask], dim=1)         # (B*Nf, Q+1)
        else:
            q_embeds, attn_mask = q_tokens, q_mask

        # 6) Cross‐attention：encoder_hidden_states 现在是 (B*Nf, T, D)
        out = self.shared_frame_Qformer.bert(
            input_ids=input_ids,                     # 不再给它原本的 input_ids
            attention_mask=attn_mask,
            query_embeds=q_tokens,              # Q + optional CLS
            encoder_hidden_states=image_embeds,   # 16 时序 tokens
            encoder_attention_mask=image_mask,
            return_dict=True
        )

        q_hidden_state = out.last_hidden_state # (B*Nf, D_q) 
        # print(f"q_hidden_state:{q_hidden_state.shape}")
        # print(f"q_hidden_state:{q_hidden_state.shape}")

        # add frame_pos embedding
        position_ids = torch.arange(Nf, dtype=torch.long, device=q_tokens.device)
        position_ids = position_ids.unsqueeze(0).expand(B, -1)
        frame_position_embeddings = self.vis_frame_position_embedding(position_ids)


        


        ##sliding window qformer casual
        frame_position_embeddings = frame_position_embeddings.unsqueeze(-2)
        frame_hidden_state = einops.rearrange(q_hidden_state, '(b t) q h -> b t q h', b=B, t=Nf)
        frame_hidden_state = frame_position_embeddings + frame_hidden_state

        return frame_hidden_state

    def encode_slide_videoQformer_visual(self, feats_vis: torch.Tensor, query_time_windows: list,use_text_query=True, use_sin=False):

        B, Nf, T, D = feats_vis.shape
        device = feats_vis.device

        clip_hidden_state = einops.rearrange(feats_vis, 'b t q h -> b (t q) h', b=B)
        clip_atts = torch.ones(clip_hidden_state.size()[:-1], dtype=torch.long).to(device)
        if use_text_query:
            query_prompts = [
                f"what happened between {s:.1f} and {e:.1f} seconds?"
                for b in range(B) for (s, e) in query_time_windows[b]
            ]
            query_toks = self.tokenizer(query_prompts,
                            return_tensors="pt",
                            padding="longest",
                            max_length=32,
                            truncation=True).to(device)
            
            query_input_ids = query_toks.input_ids
            query_embs = self.slide_vis_Qformer.get_input_embeddings()(query_input_ids)
            # print(f"query_input_ids:{query_input_ids.shape} | query_embs:{query_embs.shape}")
            text_query_embs = query_embs[:, 0:1, :]  # 使用 CLS 位置的向量

            vis_query_tokens = text_query_embs.view(B, 20, -1)  # 假設你有 20 個 query 時間段
            learned_query_tokens=self.slide_vis_qtok.expand(clip_hidden_state.shape[0], -1, -1)
            # print(f"vis_query_tokens:{vis_query_tokens.shape} | learned_query_tokens:{learned_query_tokens.shape}")
            vis_query_tokens = vis_query_tokens + learned_query_tokens
        elif use_sin:
            vis_query_tokens = self.slide_vis_qtok.expand(clip_hidden_state.shape[0], -1, -1)
            time_pe = self.sinusoidal_time_pe(query_time_windows)
            vis_query_tokens = vis_query_tokens + time_pe

        else:
            vis_query_tokens = self.slide_vis_qtok.expand(clip_hidden_state.shape[0], -1, -1)
            vis_query_tokens = query_time_windows + vis_query_tokens

        # print(f"vis_query_tokens:{vis_query_tokens.shape} | learned_query:{learned_query_tokens.shape} | clip_atts:{clip_atts.shape} | query_input_ids:{query_input_ids.shape}")

        vis_query_output = self.slide_vis_Qformer.bert(
            query_embeds=vis_query_tokens,
            encoder_hidden_states=clip_hidden_state,
            encoder_attention_mask=clip_atts,
            return_dict=True,
        )
        video_hidden = vis_query_output.last_hidden_state  # [bsz, t, dim]
        # print(f"video_hidden:{video_hidden.shape}")

        return video_hidden


    def encode_audioQformer(self, feats_vis: torch.Tensor, time_windows: list):
        """
        Args:
            feats_vis: Tensor[B, Nf, T, D]，Omnivore 输出的视觉特征
                       D=1024，T=16（temporal tokens）
            time_windows: List[B][Nf] of (start,end)
        Returns:
            fused_feats: Tensor[B, Nf, D_q]
        """
        B, Nf, T, D = feats_vis.shape
        device = feats_vis.device

        audio_embeds=einops.rearrange(feats_vis, 'b t q h -> (b t) q h')
        audio_mask = torch.ones(audio_embeds.size()[:-1], dtype=torch.long).to(device)

        # 2) 创建 prompt （跟你原本一样）
        prompts = [
            f"frame timestamp is from {s:.3f} to {e:.3f}."
            for b in range(B) for (s, e) in time_windows[b]
        ]
        toks = self.tokenizer(prompts, padding=True, truncation=True,
                              return_tensors="pt").to(device)
        input_ids, text_mask = toks.input_ids, toks.attention_mask  # (B*Nf, L)

        # 3) 文本 embedding + [CLS]
        text_embs   = self.frame_aud_Qformer.get_input_embeddings()(input_ids)  # (B*Nf, L, D)
        text_queries= text_embs[:, :1, :]                             # (B*Nf, 1, D)

        # 4) learnable Q tokens
        q_tokens = self.frame_aud_qtok.expand(B * Nf, -1, -1)           # (B*Nf, Q, D)
        q_mask   = torch.ones(q_tokens.size()[:-1], dtype=torch.long,
                              device=device)                         # (B*Nf, Q)

        # 5) 拼接 Q＋[CLS]（可选）
        if self.qformer_text_input:
            q_embeds = torch.cat([q_tokens, text_queries], dim=1)     # (B*Nf, Q+1, D)
            attn_mask= torch.cat([q_mask, text_mask], dim=1)         # (B*Nf, Q+1)
        else:
            q_embeds, attn_mask = q_tokens, q_mask

        # 6) Cross‐attention：encoder_hidden_states 现在是 (B*Nf, T, D)
        out = self.frame_aud_Qformer.bert(
            input_ids=input_ids,                     # 不再给它原本的 input_ids
            attention_mask=attn_mask,
            query_embeds=q_tokens,              # Q + optional CLS
            encoder_hidden_states=audio_embeds,   # 16 时序 tokens
            encoder_attention_mask=audio_mask,
            return_dict=True,
        )

        # add frame_pos embedding
        position_ids = torch.arange(Nf, dtype=torch.long, device=q_tokens.device)
        position_ids = position_ids.unsqueeze(0).expand(B, -1)
        frame_position_embeddings = self.aud_frame_position_embedding(position_ids)


        q_hidden_state = out.last_hidden_state # (B*Nf, D_q) 


        ##sliding window qformer casual
        frame_position_embeddings = frame_position_embeddings.unsqueeze(-2)
        frame_hidden_state = einops.rearrange(q_hidden_state, '(b t) q h -> b t q h', b=B, t=Nf)
        frame_hidden_state = frame_position_embeddings + frame_hidden_state

        return frame_hidden_state
    

    def encode_audioQformer_fuse(self, feats_vis: torch.Tensor, time_windows: list):
        """
        Args:
            feats_vis: Tensor[B, Nf, T, D]，Omnivore 输出的视觉特征
                       D=1024，T=16（temporal tokens）
            time_windows: List[B][Nf] of (start,end)
        Returns:
            fused_feats: Tensor[B, Nf, D_q]
        """
        B, Nf, T, D = feats_vis.shape
        device = feats_vis.device

        audio_embeds=einops.rearrange(feats_vis, 'b t q h -> (b t) q h')
        audio_mask = torch.ones(audio_embeds.size()[:-1], dtype=torch.long).to(device)

        # 2) 创建 prompt （跟你原本一样）
        prompts = [
            f"frame timestamp is from {s:.3f} to {e:.3f}."
            for b in range(B) for (s, e) in time_windows[b]
        ]
        toks = self.tokenizer(prompts, padding=True, truncation=True,
                              return_tensors="pt").to(device)
        input_ids, text_mask = toks.input_ids, toks.attention_mask  # (B*Nf, L)

        # 3) 文本 embedding + [CLS]
        text_embs   = self.shared_frame_Qformer.get_input_embeddings()(input_ids)  # (B*Nf, L, D)
        text_queries= text_embs[:, :1, :]                             # (B*Nf, 1, D)

        # 4) learnable Q tokens
        q_tokens = self.shared_frame_qtok.expand(B * Nf, -1, -1)           # (B*Nf, Q, D)
        q_mask   = torch.ones(q_tokens.size()[:-1], dtype=torch.long,
                              device=device)                         # (B*Nf, Q)

        # 5) 拼接 Q＋[CLS]（可选）
        if self.qformer_text_input:
            q_embeds = torch.cat([q_tokens, text_queries], dim=1)     # (B*Nf, Q+1, D)
            attn_mask= torch.cat([q_mask, text_mask], dim=1)         # (B*Nf, Q+1)
        else:
            q_embeds, attn_mask = q_tokens, q_mask

        # 6) Cross‐attention：encoder_hidden_states 现在是 (B*Nf, T, D)
        out = self.shared_frame_Qformer.bert(
            input_ids=input_ids,                     # 不再给它原本的 input_ids
            attention_mask=attn_mask,
            query_embeds=q_tokens,              # Q + optional CLS
            encoder_hidden_states=audio_embeds,   # 16 时序 tokens
            encoder_attention_mask=audio_mask,
            return_dict=True,
        )

        # add frame_pos embedding
        position_ids = torch.arange(Nf, dtype=torch.long, device=q_tokens.device)
        position_ids = position_ids.unsqueeze(0).expand(B, -1)
        frame_position_embeddings = self.aud_frame_position_embedding(position_ids)


        q_hidden_state = out.last_hidden_state # (B*Nf, D_q) 


        ##sliding window qformer casual
        frame_position_embeddings = frame_position_embeddings.unsqueeze(-2)
        frame_hidden_state = einops.rearrange(q_hidden_state, '(b t) q h -> b t q h', b=B, t=Nf)
        frame_hidden_state = frame_position_embeddings + frame_hidden_state

        return frame_hidden_state

    def encode_slide_audioQformer(self, feats_vis: torch.Tensor, query_time_windows: list,use_text_query=True, use_sin=False):


        B, Nf, T, D = feats_vis.shape
        device = feats_vis.device


        clip_hidden_state = einops.rearrange(feats_vis, 'b t q h -> b (t q) h', b=B)
        clip_atts = torch.ones(clip_hidden_state.size()[:-1], dtype=torch.long).to(device)
        if use_text_query:
            query_prompts = [
                f"what happend between {s:.1f} and {e:.1f} seconds?"
                for b in range(B) for (s, e) in query_time_windows[b]
            ]
            query_toks = self.tokenizer(query_prompts,
                                        return_tensors="pt",
                                        padding=True,
                                        truncation=True).to(device)
            query_input_ids = query_toks.input_ids
            query_embs = self.slide_aud_Qformer.get_input_embeddings()(query_input_ids)
            text_query_embs = query_embs[:, 0:1, :]  # 使用 CLS 位置的向量

            aud_query_tokens = text_query_embs.view(B, 20, -1)  # 假設你有 20 個 query 時間段
            learned_query_tokens=self.slide_aud_qtok.expand(clip_hidden_state.shape[0], -1, -1)
            aud_query_tokens = aud_query_tokens + learned_query_tokens

        elif use_sin:
            aud_query_tokens = self.slide_aud_qtok.expand(clip_hidden_state.shape[0], -1, -1)
            time_pe = self.sinusoidal_time_pe(query_time_windows)
            aud_query_tokens = aud_query_tokens + time_pe

        else:
            aud_query_tokens = self.slide_vis_qtok.expand(clip_hidden_state.shape[0], -1, -1)
            aud_query_tokens = query_time_windows + aud_query_tokens

        aud_query_output = self.slide_aud_Qformer.bert(
            query_embeds=aud_query_tokens,
            encoder_hidden_states=clip_hidden_state,
            encoder_attention_mask=clip_atts,
            return_dict=True,
        )
        aud_hidden = aud_query_output.last_hidden_state  # [bsz, t, dim]

        return aud_hidden





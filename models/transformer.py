# ----------------------------------------------------------------------------------------------
# GSRTR Official Code
# Copyright (c) Junhyeong Cho. All Rights Reserved
# Licensed under the Apache License 2.0 [see LICENSE for details]
# ----------------------------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved [see LICENSE for details]
# ----------------------------------------------------------------------------------------------

"""
GSR Verb-Role Transformer
"""
import copy
import torch
import torch.nn.functional as F
from typing import Optional
from torch import nn, Tensor


class VerbRoleTransformer(nn.Module):

    def __init__(self, d_model=512, nhead=8, num_encoder_layers=6, num_verb_decoder_layers=2, num_role_decoder_layers=2,
                 dim_feedforward=2048, dropout=0.15, activation="relu", num_steps=3, num_topkv=[100, 10, 1]):
        super().__init__()

        self.d_model = d_model
        self.nhead = nhead
        self.num_verb_classes = 504
        self.num_steps = num_steps
        self.num_topkv = num_topkv
        assert num_topkv[-1] == 1
        assert num_steps == len(num_topkv)

        # encoder
        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation)
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers,
                                          norm=nn.LayerNorm(d_model))

        # verb role decoder
        verb_decoder_layer = TransformerVerbDecoderLayer(d_model, nhead, dim_feedforward,
                                                         dropout, activation)
        role_decoder_layer = TransformerRoleDecoderLayer(d_model, nhead, dim_feedforward,
                                                         dropout, activation)
        verb_role_decoder = []
        for _ in range(num_steps):
            verb_role_decoder.append(TransformerVerbRoleDecoder(
                d_model, nhead, dim_feedforward, dropout, activation,
                verb_decoder_layer, num_verb_decoder_layers,
                role_decoder_layer, num_role_decoder_layers, ))
        self.verb_role_decoder = nn.Sequential(*verb_role_decoder)

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward_encoder(self, src, mask, pos_embed):
        # flatten NxCxHxW to HWxNxC
        bs, c, h, w = src.shape
        src = src.flatten(2).permute(2, 0, 1)
        assert src.shape == torch.Size((h*w, bs, c))
        pos_embed = pos_embed.flatten(2).permute(2, 0, 1)
        assert pos_embed.shape == torch.Size((h*w, bs, c))
        mask = mask.flatten(1)
        assert mask.shape == torch.Size((bs, h*w))

        # Transformer Encoder
        memory = self.encoder(src, src_key_padding_mask=mask, pos=pos_embed)
        assert memory.shape == torch.Size((h*w, bs, c))

        return memory

    def forward_verb_role_decoder(
            self, verb_token, role_tokens, topk_verb_role_mask, memory,
            mask: Optional[Tensor] = None,
            pos_embed: Optional[Tensor] = None,
            gt_verb: Optional[Tensor] = None, ):
        hw, bs, c = memory.shape
        pos_embed = pos_embed.flatten(2).permute(2, 0, 1)
        assert pos_embed.shape == torch.Size((hw, bs, c))
        mask = mask.flatten(1)
        assert mask.shape == torch.Size((bs, hw))

        num_verb_token, _, _ = verb_token.shape
        assert verb_token.shape == torch.Size((num_verb_token, bs, c))
        num_role_tokens, _, _ = role_tokens.shape
        assert role_tokens.shape == torch.Size((num_role_tokens, bs, c))

        # Transformer Decoder
        vhs = verb_token
        rhs = role_tokens
        role_mask = torch.zeros((bs, num_role_tokens), dtype=bool, device=rhs.device)
        step_r2v, step_vhs, step_v2r, step_role_mask, step_rhs = [], [], [], [], []
        for s in range(self.num_steps):
            r2v, vhs, v2r, role_mask, rhs = self.verb_role_decoder[s](
                    verb_query=vhs,
                    role_queries=rhs,
                    topk_verb_role_mask=topk_verb_role_mask,
                    num_topkv=self.num_topkv[s],
                    memory=memory,
                    mask=mask,
                    role_mask=role_mask,
                    pos_embed=pos_embed,
                    gt_verb=gt_verb)

            assert r2v.shape == torch.Size((num_verb_token, bs, c))
            assert vhs.shape == torch.Size((num_verb_token, bs, c))
            assert v2r.shape == torch.Size((num_role_tokens, bs, c))
            assert role_mask.shape == torch.Size((bs, num_role_tokens))
            assert rhs.shape == torch.Size((num_role_tokens, bs, c))

            step_r2v.append(r2v)
            step_vhs.append(vhs)
            step_v2r.append(v2r)
            step_role_mask.append(role_mask.transpose(0, 1))
            step_rhs.append(rhs)

        batch_step_r2v = torch.stack(step_r2v, dim=1)
        batch_step_vhs = torch.stack(step_vhs, dim=1)
        batch_step_v2r = torch.stack(step_v2r, dim=1)
        batch_step_role_mask = torch.stack(step_role_mask, dim=1)
        batch_step_rhs = torch.stack(step_rhs, dim=1)

        assert batch_step_r2v.shape == torch.Size((num_verb_token, self.num_steps, bs, c))
        assert batch_step_vhs.shape == torch.Size((num_verb_token, self.num_steps, bs, c))
        assert batch_step_v2r.shape == torch.Size((num_role_tokens, self.num_steps, bs, c))
        assert batch_step_role_mask.shape == torch.Size((num_role_tokens, self.num_steps, bs))
        assert batch_step_rhs.shape == torch.Size((num_role_tokens, self.num_steps, bs, c))

        return batch_step_r2v, batch_step_vhs, batch_step_v2r, batch_step_role_mask, batch_step_rhs


class TransformerEncoder(nn.Module):

    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src,
                mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        output = src

        for layer in self.layers:
            output = layer(output, src_attn_mask=mask,
                           src_key_padding_mask=src_key_padding_mask, pos=pos)

        if self.norm is not None:
            output = self.norm(output)

        return output


class TransformerVerbRoleDecoder(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward, dropout, activation,
                 verb_decoder_layer, num_verb_decoder_layers,
                 role_decoder_layer, num_role_decoder_layers, ):
        super().__init__()
        self.verb_decoder = TransformerVerbDecoder(verb_decoder_layer, num_verb_decoder_layers,
                                                   norm=nn.LayerNorm(d_model))
        self.verb_to_role = VerbToRoleFFN(d_model, dim_feedforward, dropout, activation)

        self.role_decoder = TransformerRoleDecoder(role_decoder_layer, num_role_decoder_layers,
                                                   norm=nn.LayerNorm(d_model))

        self.role_to_verb = RoleToVerbAttention(d_model, nhead, dropout)

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, verb_query, role_queries, topk_verb_role_mask, num_topkv, memory,
                mask: Optional[Tensor] = None,
                role_mask: Optional[Tensor] = None,
                pos_embed: Optional[Tensor] = None,
                gt_verb: Optional[Tensor] = None, ):
        hw, bs, c = memory.shape
        assert pos_embed.shape == torch.Size((hw, bs, c))
        assert mask.shape == torch.Size((bs, hw))

        num_verb_query, _, _ = verb_query.shape
        assert verb_query.shape == torch.Size((num_verb_query, bs, c))
        num_role_queries, _, _ = role_queries.shape
        assert role_queries.shape == torch.Size((num_role_queries, bs, c))
        assert role_mask.shape == torch.Size((bs, num_role_queries))

        # role to verb
        r2v = self.role_to_verb(verb_query, role_queries, role_key_padding_mask=role_mask)
        assert r2v.shape == torch.Size((num_verb_query, bs, c))

        # verb decoder
        vhs = self.verb_decoder(verb_query=verb_query,
                                memory=memory,
                                role2verb=r2v,
                                memory_key_padding_mask=mask,
                                pos=pos_embed,
                                query_pos=torch.zeros_like(verb_query))
        assert vhs.shape == torch.Size((num_verb_query, bs, c))

        # verb to role, top-k verb role mask
        v2r = self.verb_to_role(vhs, role_queries)
        assert v2r.shape == torch.Size((num_role_queries, bs, c))
        role_mask = topk_verb_role_mask(vhs, topk=num_topkv, gt_verb=gt_verb)
        assert role_mask.shape == torch.Size((bs, num_role_queries))

        # role decoder
        rhs = self.role_decoder(role_queries=role_queries,
                                memory=memory,
                                verb2role=v2r,
                                role_key_padding_mask=role_mask,
                                memory_key_padding_mask=mask,
                                pos=pos_embed,
                                query_pos=torch.zeros_like(role_queries))
        assert rhs.shape == torch.Size((num_role_queries, bs, c))

        return r2v, vhs, v2r, role_mask, rhs


class TransformerVerbDecoder(nn.Module):

    def __init__(self, verb_decoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = _get_clones(verb_decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, verb_query, memory,
                role2verb: Optional[Tensor] = None,
                memory_attn_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        output = verb_query if role2verb is None else verb_query + role2verb

        for layer in self.layers:
            output = layer(output, memory,
                           memory_attn_mask=memory_attn_mask,
                           memory_key_padding_mask=memory_key_padding_mask,
                           pos=pos, query_pos=query_pos)

        if self.norm is not None:
            output = self.norm(output)

        return output


class TransformerRoleDecoder(nn.Module):

    def __init__(self, role_decoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = _get_clones(role_decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, role_queries, memory,
                verb2role: Optional[Tensor] = None,
                role_attn_mask: Optional[Tensor] = None,
                memory_attn_mask: Optional[Tensor] = None,
                role_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        output = role_queries if verb2role is None else role_queries + verb2role

        for layer in self.layers:
            output = layer(output, memory,
                           role_attn_mask=role_attn_mask,
                           memory_attn_mask=memory_attn_mask,
                           role_key_padding_mask=role_key_padding_mask,
                           memory_key_padding_mask=memory_key_padding_mask,
                           pos=pos, query_pos=query_pos)

        if self.norm is not None:
            output = self.norm(output)

        return output


class TransformerEncoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.15, activation="relu"):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        # Implementation of Feedforward model
        self.ffn = FFN(d_model, dim_feedforward, dropout, activation)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward(self, src,
                src_attn_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        src2 = self.norm1(src)
        q = k = self.with_pos_embed(src2, pos)
        src2 = self.self_attn(q, k, value=src2, attn_mask=src_attn_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src2 = self.norm2(src)
        src2 = self.ffn(src2)
        src = src + self.dropout2(src2)
        return src


class TransformerVerbDecoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.15, activation="relu"):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        # Implementation of Feedforward model
        self.ffn = FFN(d_model, dim_feedforward, dropout, activation)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward(self, verb_query, memory,
                memory_attn_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        verb_query2 = self.norm1(verb_query)
        verb_query2 = self.multihead_attn(query=self.with_pos_embed(verb_query2, query_pos),
                                          key=self.with_pos_embed(memory, pos),
                                          value=memory, attn_mask=memory_attn_mask,
                                          key_padding_mask=memory_key_padding_mask)[0]
        verb_query = verb_query + self.dropout1(verb_query2)
        verb_query2 = self.norm2(verb_query)
        verb_query2 = self.ffn(verb_query2)
        verb_query = verb_query + self.dropout2(verb_query2)
        return verb_query


class TransformerRoleDecoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.15, activation="relu"):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        # Implementation of Feedforward model
        self.ffn = FFN(d_model, dim_feedforward, dropout, activation)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward(self, role_queries, memory,
                role_attn_mask: Optional[Tensor] = None,
                memory_attn_mask: Optional[Tensor] = None,
                role_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        role_queries2 = self.norm1(role_queries)
        q = k = self.with_pos_embed(role_queries2, query_pos)
        role_queries2 = self.self_attn(q, k, value=role_queries2, attn_mask=role_attn_mask,
                                       key_padding_mask=role_key_padding_mask)[0]
        role_queries = role_queries + self.dropout1(role_queries2)
        role_queries2 = self.norm2(role_queries)
        role_queries2 = self.multihead_attn(query=self.with_pos_embed(role_queries2, query_pos),
                                            key=self.with_pos_embed(memory, pos),
                                            value=memory, attn_mask=memory_attn_mask,
                                            key_padding_mask=memory_key_padding_mask)[0]
        role_queries = role_queries + self.dropout2(role_queries2)
        role_queries2 = self.norm3(role_queries)
        role_queries2 = self.ffn(role_queries2)
        role_queries = role_queries + self.dropout3(role_queries2)
        return role_queries


class FFN(nn.Module):
    def __init__(self, d_model, dim_feedforward, dropout, activation):
        super().__init__()
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.activation = _get_activation_fn(activation)

    def forward(self, x):
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return x


class VerbToRoleFFN(nn.Module):
    def __init__(self, d_model, dim_feedforward, dropout, activation):
        super().__init__()
        self.ffn = FFN(d_model, dim_feedforward, dropout, activation)

    def forward(self, vhs, role_queries):
        """
        Input:
            vhs: verb feature
        Output:
            v2r: verb to role info, added to role queries
        """
        num_verb_query, bs, c = vhs.shape
        num_role_queries, _, _ = role_queries.shape
        assert role_queries.shape == torch.Size((num_role_queries, bs, c))

        v2r = self.ffn(vhs)
        assert v2r.shape == torch.Size((num_verb_query, bs, c))
        v2r = v2r.repeat(num_role_queries, 1, 1)

        return v2r


class RoleToVerbAttention(nn.Module):
    def __init__(self, d_model, nhead, dropout):
        super().__init__()
        self.query_proj = nn.Sequential(nn.Linear(d_model, d_model),
                                        nn.ReLU(),
                                        nn.Dropout(dropout),
                                        nn.Linear(d_model, d_model),
                                        nn.ReLU(),
                                        nn.Dropout(dropout),)
        self.key_proj = nn.Sequential(nn.Linear(d_model, d_model),
                                      nn.ReLU(),
                                      nn.Dropout(dropout),
                                      nn.Linear(d_model, d_model),
                                      nn.ReLU(),
                                      nn.Dropout(dropout),)
        self.value_proj = nn.Sequential(nn.Linear(d_model, d_model),
                                        nn.ReLU(),
                                        nn.Dropout(dropout),
                                        nn.Linear(d_model, d_model),
                                        nn.ReLU(),
                                        nn.Dropout(dropout),)
        self.attn = nn.MultiheadAttention(d_model, nhead, dropout)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, vhs, rhs, role_key_padding_mask):
        vhs2 = self.norm(self.query_proj(vhs))
        k = self.key_proj(rhs)
        v = self.value_proj(rhs)
        vhs2 = self.attn(query=vhs2, key=k, value=v, key_padding_mask=role_key_padding_mask)[0]
        vhs = vhs + self.dropout(vhs2)
        return vhs


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")


def build_transformer(args):
    return VerbRoleTransformer(d_model=args.hidden_dim,
                               dropout=args.dropout,
                               nhead=args.nheads,
                               dim_feedforward=args.dim_feedforward,
                               num_encoder_layers=args.enc_layers,
                               num_verb_decoder_layers=args.verb_layers,
                               num_role_decoder_layers=args.role_layers,
                               num_steps=args.num_steps,
                               num_topkv=args.num_topkv, )

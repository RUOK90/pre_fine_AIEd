# pylint: skip-file
import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


def scaled_dot_attention(q, k, v, mask=None, dropout=None):
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(q.size(-1))

    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)

    prob = scores.softmax(-1)

    if dropout is not None:
        prob = dropout(prob)

    return torch.matmul(prob, v)


class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        "Implements Figure 2"
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = [
            l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
            for l, x in zip(self.linears, (query, key, value))
        ]

        # 2) Apply attention on all the projected vectors in batch.
        x = scaled_dot_attention(query, key, value, mask=mask, dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)


class MultiHeadedAttentionHF(nn.Module):
    """
    Added condition no_last_layer because last linear layer
    that is in old attention is not in HF version
    In HF, the last linear layer is implemented in Bert(Self)Output
    In old version, this layer is not in SublayerConnection
    but added at the end of MultiHeadedeAttentionHF and
    PositionwiseFeedForwardHF
    """

    def __init__(self, h, d_model, dropout=0.1):
        "Take in model size and number of heads."
        super(MultiHeadedAttentionHF, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        # CHANGE: self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.linears = nn.ModuleList(
            [
                nn.Linear(d_model, d_model),
                nn.Linear(d_model, d_model),
                nn.Linear(d_model, d_model),
                nn.Linear(d_model, d_model),
            ]
        )
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None, no_last_layer=True):
        "Implements Figure 2"
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = [
            l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
            for l, x in zip(self.linears, (query, key, value))
        ]

        # 2) Apply attention on all the projected vectors in batch.
        x = scaled_dot_attention(query, key, value, mask=mask, dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)
        # CHANGE: return self.linears[-1](x)
        if no_last_layer:
            return x
        else:
            return self.linears[-1](x)


class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."

    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = GELU()

    def forward(self, x):
        return self.w_2(self.dropout(self.activation(self.w_1(x))))


class PositionwiseFeedForwardHF(nn.Module):
    """
    Differences:
        Also added condition no_last_layer (check Multi
        HeadedAttentionHF docstring for details)
        Changed Gelu function from GELU() to F.gelu
    """

    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForwardHF, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.gelu

    def forward(self, x, no_last_layer=True):
        if no_last_layer:
            return self.dropout(self.activation(self.w_1(x)))
        else:
            return self.w_2(self.dropout(self.activation(self.w_1(x))))


class LayerNorm(nn.Module):
    "Construct a layernorm module (See citation for details)."

    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class LayerNormHF(nn.Module):
    "Construct a layernorm module (See citation for details)."

    def __init__(self, features, eps=1e-5):
        super(LayerNormHF, self).__init__()
        self.gamma = nn.Parameter(torch.ones(features))
        self.beta = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        var = x.var(-1, keepdim=True, unbiased=False)
        return (self.gamma * ((x - mean) / torch.sqrt(var + self.eps))) + self.beta


class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """

    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        return self.norm(x + self.dropout(sublayer(x)))


class SublayerConnectionHF(nn.Module):
    """
    Differences:
        used torch.nn.LayerNorm instead of self defined LayerNorm in
        old_layers
    """

    def __init__(self, size, dropout):
        super(SublayerConnectionHF, self).__init__()
        # CHANGE: self.norm = LayerNorm(size)
        self.norm = torch.nn.LayerNorm(size, eps=1e-12)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        return self.norm(x + self.dropout(sublayer(x)))


class GELU(nn.Module):
    """
    Paper Section 3.4, last paragraph notice that BERT used the GELU instead of RELU
    """

    # fmt: off
    def forward(self, x):
        # fmt: off
        return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
    # fmt: on


class GELUHF(nn.Module):
    """
    Paper Section 3.4, last paragraph notice that BERT used the GELU instead of RELU
    """

    # fmt: off
    def forward(self, x):
        return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0))) # 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
    # fmt: on


class EncoderLayer(nn.Module):
    "Encoder is made up of self-attn and feed forward (defined below)"

    def __init__(self, size, h, dropout):
        super().__init__()
        self.self_attn = MultiHeadedAttention(h, size, dropout)
        self.feed_forward = PositionwiseFeedForward(size, size * 4, dropout)
        self.att_sublayer = SublayerConnection(size, dropout)
        self.ff_sublayer = SublayerConnection(size, dropout)
        self.size = size

    def forward(self, x, mask):
        "Follow Figure 1 (left) for connections."
        out = self.att_sublayer(x, lambda x: self.self_attn(x, x, x, mask))

        return self.ff_sublayer(out, self.feed_forward)


class EncoderLayerHF(nn.Module):
    "Encoder is made up of self-attn and feed forward (defined below)"

    def __init__(self, size, h, dropout):
        super().__init__()
        self.self_attn = MultiHeadedAttentionHF(h, size, dropout)
        ## CHANGE : self.feed_forward = PositionwiseFeedForward(size, size * 4, dropout)
        ## Bert uses 3072 as ffn dimension
        self.feed_forward = PositionwiseFeedForwardHF(size, 3072, dropout)
        self.att_sublayer = SublayerConnectionHF(size, dropout)
        self.ff_sublayer = SublayerConnectionHF(size, dropout)
        self.size = size

    def forward(self, x, mask):
        "Follow Figure 1 (left) for connections."
        out = self.att_sublayer(
            x, lambda x: self.self_attn(x, x, x, mask, no_last_layer=False)
        )
        return self.ff_sublayer(
            out, lambda out: self.feed_forward(out, no_last_layer=False)
        )


class Generator(nn.Module):
    def __init__(self, d_model, vocab):
        super().__init__()
        self.proj = nn.Linear(d_model, vocab)

    def forward(self, x):
        return self.proj(x)


# class EmbeddingLayer(nn.Embedding):
#     def __init__(self, input_size, embed_size):
#         super().__init__(
#             num_embeddings=input_size + 1, embedding_dim=embed_size, padding_idx=0
#         )


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (
            torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)
        ).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        # register_buffer because we don't want these embeddings to be trained
        self.register_buffer("pe", pe)

    def forward(self, x):
        return self.pe[:, : x.size(1)]


# class DecoderLayer(nn.Module):
#    def __init__(self, size, h, dropout):
#        super().__init__()
#        self.self_attn = MultiHeadedAttention(h, size, dropout)
#        self.src_attn = MultiHeadedAttention(h, size, dropout)
#        self.feed_forward = PositionwiseFeedForward(size, size * 4, dropout)
#        self.self_att_sublayer = SublayerConnection(size, dropout)
#        self.ende_att_sublayer = SublayerConnection(size, dropout)
#        self.ff_sublayer = SublayerConnection(size, dropout)
#        self.size = size
#
#    def forward(self, x, m, src_mask, tgt_mask):
#        x = self.self_att_sublayer(x, lambda x: self.self_attn(x, x, x, tgt_mask))
#        x = self.ende_att_sublayer(x, lambda x: self.src_attn(x, m, m, src_mask))
#        return self.ff_sublayer(x, self.feed_forward)


class Encoder(nn.Module):
    def __init__(self, layer, N):
        super().__init__()
        self.layers = clones(layer, N)

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return x


# class Decoder(nn.Module):
#    def __init__(self, layer, N):
#        super().__init__()
#        self.layers = clones(layer, N)

#    def forward(self, x, memory, src_mask, tgt_mask):
#        for layer in self.layers:
#            x = layer(x, memory, src_mask, tgt_mask)
#        return x


class EncoderEmbedding(nn.Module):
    def __init__(
        self,
        embed_question,
        embed_assessment,
        embed_finished_time,
        embed_postion,
        embed_part,
        dropout,
        is_feature_based,
        device,
        freeze_embedding=False,
    ):
        super().__init__()
        self.embed_question = embed_question
        self.embed_assessment = embed_assessment
        self.embed_position = embed_postion
        self.embed_finished_time = embed_finished_time
        # self.embed_start_time = embed_start_time
        # self.embed_elapsed_time = embed_elapsed_time
        self.embed_part = embed_part
        self.dropout = nn.Dropout(dropout)
        self.is_feature_based = is_feature_based
        self.device = device
        self.freeze_embedding = freeze_embedding

    def update_embed_question(self, embed_question):
        self.embed_question = embed_question
        self.linear = nn.Linear(
            embed_question.weight.size(1), self.embed_assessment.weight.size(1)
        ).to(self.device)

    def forward(
        self,
        qid_list,
        input_processed_assessment_list,
        input_finished_time_list,
        part_list,
    ):
        qid_list = qid_list.long()
        input_processed_assessment_list = input_processed_assessment_list.long()
        input_finished_time_list = input_finished_time_list.long()
        part_list = part_list.long()

        embedded_questions = self.embed_question(qid_list)
        if self.is_feature_based is True:
            embedded_questions = self.linear(embedded_questions)

        embed_output = (
            embedded_questions
            + self.embed_position(qid_list)
            + self.embed_assessment(input_processed_assessment_list)
            + self.embed_finished_time(input_finished_time_list)
            + self.embed_part(part_list)
        )  # + self.embed_start_time(s) + self.embed_elapsed_time(e)

        if self.freeze_embedding is True:
            embed_output = embed_output.detach()

        return self.dropout(embed_output)


class ScoreGenerator(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_model // 2)
        self.fc2 = nn.Linear(d_model // 2, d_model // 4)
        self.fc3 = nn.Linear(d_model // 4, 1)
        self.activation = GELU()

    def forward(self, x, masks):
        x = torch.where(masks.squeeze(1).unsqueeze(-1) == 1, x, torch.zeros_like(x))
        outputs = x.sum(1)
        outputs /= masks.squeeze(1).sum(1).unsqueeze(-1).float()
        logits = self._fully_connected(outputs)

        return logits

    def _fully_connected(self, x):
        hidden1 = self.activation(self.fc1(x))
        hidden2 = self.activation(self.fc2(hidden1))
        outputs = self.fc3(hidden2)

        return torch.sigmoid(outputs)


class ReviewGenerator(ScoreGenerator):
    def __init__(self, d_model):
        super().__init__(d_model * 2)

    def forward(self, x, masks, target_question):
        x = torch.where(masks.squeeze(1).unsqueeze(-1) == 1, x, torch.zeros_like(x))
        outputs = x.sum(1)
        outputs /= masks.squeeze(1).sum(1).unsqueeze(-1).float()
        outputs = torch.cat([outputs, target_question.squeeze(1)], -1)
        logits = self._fully_connected(outputs)

        return logits


# class DecoderEmbedding(nn.Module):
#     def __init__(self, embed_response, embed_postion, embed_start_time, embed_elapsed_time, dropout):
#         super().__init__()
#         self.embed_response = embed_response
#         self.embed_position = embed_postion
#         self.embed_start_time = embed_start_time
#         self.embed_elapsed_time = embed_elapsed_time
#         self.dropout = nn.Dropout(dropout)

#     def forward(self, r, s, e):
#         r = r.long()
#         s = s.long()
#         e = e.long()

#         embed = self.embed_response(r) + self.embed_position(r) + self.embed_start_time(s) + self.embed_elapsed_time(e)
#         return self.dropout(embed)

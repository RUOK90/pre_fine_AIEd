"""
Definition of Layers from 'The Annotated Transformers'
"""

import math

import torch

from am_v2 import config, util


def scaled_dot_attention(query, key, value, mask=None, dropout=None):
    """
    Scaled dot product attention following the Annotated Transformer
    """
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(query.size(-1))

    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)

    prob = scores.softmax(-1)

    if dropout is not None:
        prob = dropout(prob)

    return torch.matmul(prob, value)


class MultiHeadedAttention(torch.nn.Module):
    """
    Multi-head Attention module
    """

    def __init__(self, d_h, d_model, dropout=0.1):
        "Takes in model size and number of heads, and optionally dropout"
        super(MultiHeadedAttention, self).__init__()
        assert d_model % d_h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // d_h
        self.d_h = d_h
        self.linears = util.clones(torch.nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = torch.nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        "Implements Figure 2"

        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = [
            l(x).view(nbatches, -1, self.d_h, self.d_k).transpose(1, 2)
            for l, x in zip(self.linears, (query, key, value))
        ]

        # 2) Apply attention on all the projected vectors in batch.
        a_h = scaled_dot_attention(query, key, value, mask=mask, dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear.
        a_h = a_h.transpose(1, 2).contiguous().view(nbatches, -1, self.d_h * self.d_k)
        return self.linears[-1](a_h)


class PositionwiseFeedForward(torch.nn.Module):
    "Implements FFN equation."

    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = torch.nn.Linear(d_model, d_ff)
        self.w_2 = torch.nn.Linear(d_ff, d_model)
        self.dropout = torch.nn.Dropout(dropout)
        self.activation = GELU()

    def forward(self, x):
        return self.w_2(self.dropout(self.activation(self.w_1(x))))


class LayerNorm(torch.nn.Module):
    "Construct a layernorm module (See citation for details)."

    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = torch.nn.Parameter(torch.ones(features))
        self.b_2 = torch.nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class SublayerConnection(torch.nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """

    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        return self.norm(x + self.dropout(sublayer(x)))


class GELU(torch.nn.Module):
    """
    Paper Section 3.4, last paragraph notice that BERT used the GELU instead of RELU
    """

    def forward(self, x):
        return (
            0.5
            * x
            * (
                1
                + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3)))
            )
        )


class EncoderLayer(torch.nn.Module):
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


class Generator(torch.nn.Module):
    """
    Base class for answer generator (response_correctness, timeliness)
    """

    def __init__(self, d_model, vocab):
        super().__init__()
        self.proj = torch.nn.Linear(d_model, vocab)

    def forward(self, x):
        return self.proj(x)


class EmbeddingLayer(torch.nn.Embedding):
    """
    Wrapper class for nn.Embedding layer.
    """

    def __init__(self, input_size, embed_size):
        super().__init__(
            num_embeddings=input_size + 1, embedding_dim=embed_size, padding_idx=0
        )


class PositionalEncoding(torch.nn.Module):
    """
    Positional Encoding class using the sinusoidal functions
    TODO: Why don't we use relative positional encoding as written in the paper?
    """

    def __init__(self, d_model, max_len=512):
        super().__init__()

        # Compute the positional encodings once in log space.
        pos_enc = torch.zeros(max_len, d_model).float()
        pos_enc.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (
            torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)
        ).exp()

        pos_enc[:, 0::2] = torch.sin(position * div_term)
        pos_enc[:, 1::2] = torch.cos(position * div_term)

        pos_enc = pos_enc.unsqueeze(0)
        self.register_buffer("pos_enc", pos_enc)

    def forward(self, seq_size):
        return self.pos_enc[:, : seq_size]


class Encoder(torch.nn.Module):
    """Sequentially applies encoder layers"""

    def __init__(self, layer, N):
        super().__init__()
        self.layers = util.clones(layer, N)

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return x


class EncoderEmbedding(torch.nn.Module):
    """Sums embedding layers"""

    def __init__(
        self,
        embedding_by_feature,
        seq_size,
        dropout,
        is_feature_based,
        device,
        freeze_embedding=False,
    ):
        super().__init__()
        self.device = device
        self.seq_size = seq_size
        self.dropout = torch.nn.Dropout(dropout)
        self.embedding_by_feature = embedding_by_feature
        self.is_feature_based = is_feature_based
        # if is_feature_based:
        #     self.embed_qid = embed_qid
        #     self.linear = torch.nn.Linear(
        #         embed_qid.weight.size(1), self.embed_is_correct.weight.size(1)
        #     ).to(self.device)
        self.freeze_embedding = freeze_embedding

    def forward(
        self, input_features
    ):

        # qid_list = qid_list.long()
        # part_list = part_list.long()
        # input_correctness = input_correctness.long()
        # input_timeliness = input_timeliness.long()
        # embedded_questions = self.embed_question(qid_list)
        # if self.is_feature_based:
        #     embedded_questions = self.linear(embedded_questions)
        embed_output = 0
        for name, feature in input_features.items():
            embed_output += self.embedding_by_feature[name](feature)
        embed_output += self.embedding_by_feature['position'](self.seq_size)

        if self.freeze_embedding:
            embed_output = embed_output.detach()

        return self.dropout(embed_output)


class ScoreGenerator(torch.nn.Module):
    """Predicts estimated score given AM output"""

    def __init__(self, d_model):
        super().__init__()
        self.fc1 = torch.nn.Linear(d_model, d_model // 2)
        self.fc2 = torch.nn.Linear(d_model // 2, d_model // 4)
        self.fc3 = torch.nn.Linear(d_model // 4, 1)
        self.activation = GELU()

    def forward(self, am_outputs, masks):
        am_outputs = torch.where(
            masks.squeeze(1).unsqueeze(-1) == 1,
            am_outputs,
            torch.zeros_like(am_outputs),
        )
        outputs = am_outputs.sum(1)
        outputs /= masks.squeeze(1).sum(1).unsqueeze(-1).float()
        return self._fully_connected(outputs)

    def _fully_connected(self, am_outputs):
        """Runs am_outputs sequentially through MLP layers"""
        hidden1 = self.activation(self.fc1(am_outputs))
        hidden2 = self.activation(self.fc2(hidden1))
        outputs = self.fc3(hidden2)

        if config.ARGS.score_last_activation == "none":
            return outputs
        if config.ARGS.score_last_activation == "sigmoid":
            return torch.sigmoid(outputs)
        if config.ARGS.score_last_activation == "tanh":
            return torch.tanh(outputs)

        raise ValueError("No proper activation defined")


class ReviewGenerator(ScoreGenerator):
    """Predicts review correctness given AM output"""

    def __init__(self, d_model):
        super().__init__(d_model * 2)

    def forward(self, am_outputs, masks, target_question):
        am_outputs = torch.where(
            masks.squeeze(1).unsqueeze(-1) == 1,
            am_outputs,
            torch.zeros_like(am_outputs),
        )
        outputs = am_outputs.sum(1)
        count = torch.clamp(masks.squeeze(1).sum(1).unsqueeze(-1).float(), min=1e-4)
        outputs /= count
        outputs = torch.cat([outputs, target_question.squeeze(1)], -1)
        logits = self._fully_connected(outputs)

        return logits

"""Network code for fine-tuning
"""
import torch

from am_v2 import layers


class Model(torch.nn.Module):
    """Pytorch model for fine-tuning"""

    def __init__(
        self,
        q_size,
        time_size,
        r_size,
        p_size,
        n_layer=3,
        d_model=512,
        heads=8,
        dropout=0.1,
        device=None,
        max_len=512,
        is_feature_based=False,
    ):
        super().__init__()

        self.device = device
        self.encoder = layers.Encoder(
            layers.EncoderLayer(d_model, heads, dropout), n_layer
        )
        self.q_size = q_size

        self.embed_question = layers.EmbeddingLayer(
            input_size=q_size, embed_size=d_model
        )
        self.embed_response = layers.EmbeddingLayer(
            input_size=r_size, embed_size=d_model
        )
        self.embed_time = layers.EmbeddingLayer(
            input_size=time_size, embed_size=d_model
        )
        self.embed_position = layers.PositionalEncoding(d_model, max_len=max_len)
        self.embed_part = layers.EmbeddingLayer(input_size=p_size, embed_size=d_model)

        self.src_embed = layers.EncoderEmbedding(
            self.embed_question,
            self.embed_part,
            self.embed_response,
            self.embed_time,
            self.embed_position,
            dropout,
            is_feature_based,
            self.device,
        )
        self.lc_generator = layers.ScoreGenerator(d_model)
        self.rc_generator = layers.ScoreGenerator(d_model)

    def freeze_embed(self):
        """Freeze embedding layer during fine-tuning"""
        for param in self.src_embed.parameters():
            param.requires_grad = False

    def freeze_encoder(self):
        """Freeze encoder blocks during fine-tuning"""
        for param in self.encoder.parameters():
            param.requires_grad = False

    def forward(self, input_features):
        input_features = [feat.to(self.device) for feat in input_features]
        (qid_list, part_list, input_correctness, input_timeliness,) = input_features

        # items that are not padded
        src_mask = (qid_list != 0).unsqueeze(-2)

        output = self.encoder(
            self.src_embed(qid_list, part_list, input_correctness, input_timeliness,),
            src_mask,
        )

        lc_output = self.lc_generator(output, src_mask)
        rc_output = self.rc_generator(output, src_mask)

        return lc_output, rc_output

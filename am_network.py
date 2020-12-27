import torch

import am_layers
from config import *


class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        self.encoder = am_layers.Encoder(
            am_layers.EncoderLayer(ARGS.d_model, ARGS.num_heads, ARGS.dropout),
            ARGS.num_layers,
        )
        embed_position = am_layers.PositionalEncoding(
            ARGS.d_model, max_len=ARGS.max_seq_size
        )
        embed_qid = am_layers.EmbeddingLayer(
            input_size=Const.FEATURE_SIZE["qid"], embed_size=ARGS.d_model
        )
        embed_part = am_layers.EmbeddingLayer(input_size=8, embed_size=ARGS.d_model)
        embed_is_correct = am_layers.EmbeddingLayer(
            input_size=3, embed_size=ARGS.d_model
        )
        embed_is_on_time = am_layers.EmbeddingLayer(
            input_size=3, embed_size=ARGS.d_model
        )
        embed_elapsed_time = torch.nn.Linear(1, ARGS.d_model)
        embed_lag_time = torch.nn.Linear(1, ARGS.d_model)
        self.embedding_by_feature = torch.nn.ModuleDict(
            {
                "position": embed_position,
                "qid": embed_qid,
                "part": embed_part,
                "is_correct": embed_is_correct,
                "is_on_time": embed_is_on_time,
                "elapsed_time": embed_elapsed_time,
                "lag_time": embed_lag_time,
            }
        )
        self.src_embed = am_layers.EncoderEmbedding(
            self.embedding_by_feature,
            ARGS.max_seq_size,
            ARGS.dropout,
        )

    def forward(self, qid_list, part_list, input_correctness):
        """
        Forward function: to be implemented after being inherited
        """
        raise NotImplementedError("This is a base class")

    def freeze_embed(self):
        """
        Freeze embedding
        """
        for param in self.src_embed.parameters():
            param.requires_grad = False

    def freeze_encoder(self):
        """
        Freeze encoder
        """
        for param in self.encoder.parameters():
            param.requires_grad = False


class PretrainModel(Model):
    def __init__(self):
        super(PretrainModel, self).__init__()
        self.generator = torch.nn.ModuleDict(
            {target: am_layers.Generator(ARGS.d_model, 1) for target in ARGS.targets}
        )

    def forward(self, input_features, padding_masks):
        # items that are not padded
        src_mask = (input_features["qid"] != 0).unsqueeze(-2)

        encoder_output = self.encoder(
            self.src_embed(input_features),
            src_mask,
        )

        outputs = {}
        for pretrain_feature_name, generator in self.generator.items():
            output = generator(encoder_output).squeeze(-1)
            sigmoid_output = torch.sigmoid(output)
            outputs[pretrain_feature_name] = (output, sigmoid_output)

        return outputs


class ScoreModel(Model):
    def __init__(self):
        super(ScoreModel, self).__init__()
        self.lc_generator = am_layers.ScoreGenerator(ARGS.d_model)
        self.rc_generator = am_layers.ScoreGenerator(ARGS.d_model)

    def forward(self, input_features, padding_masks):
        # items that are not padded
        src_mask = (input_features["qid"] != 0).unsqueeze(-2)

        encoder_output = self.encoder(
            self.src_embed(input_features),
            src_mask,
        )

        outputs = {
            "lc": self.lc_generator(encoder_output, src_mask),
            "rc": self.rc_generator(encoder_output, src_mask),
        }

        return outputs


class ReviewModel(Model):
    """
    Generates two answers (response_correctness & timeliness)
    Inherits the above class Model
    """

    def __init__(
        self,
        q_size,
        t_size,
        r_size,
        p_size,
        n_layer=3,
        d_model=512,
        h=8,
        dropout=0.1,
        device=None,
        seq_size=512,
        is_feature_based=False,
        freeze_embedding=False,
        freeze_encoder_block=False,
    ):
        Model.__init__(
            self,
            q_size,
            t_size,
            r_size,
            p_size,
            n_layer,
            d_model,
            h,
            dropout,
            device,
            seq_size,
            is_feature_based,
            freeze_embedding,
            freeze_encoder_block,
        )
        self.review_generator = am_layers.ReviewGenerator(d_model)

    def forward(self, input_features, review_qid):
        """
        Runs the inputs through the model

        Args:
            input_features: list of features
                - qid_list: list of question sequences
                - part_list: parts of questions
                - input_correctness: whether questions are answered correctly
                - input_timeliness: whether questions are solved on time
                - input_elapsed_time: elapsed time in seconds
                - input_tags: tags corresponding to each question

        Returns:
        """

        # items that are not padded
        src_mask = (input_features["qid"] != 0).unsqueeze(-2)

        encoder_output = self.encoder(
            self.src_embed(input_features),
            src_mask,
        )
        review_embedding = self.embedding_by_feature["qid"](review_qid)

        output = self.review_generator(encoder_output, src_mask, review_embedding)

        return output


class BERT(torch.nn.Module):
    """
    Single answer generator model
    """

    def __init__(
        self,
        num_vocab,
        n_layer=2,
        d_model=256,
        h=8,
        dropout=0.1,
        device=None,
        seq_size=512,
        is_feature_based=False,
        freeze_embedding=False,
        freeze_encoder_block=False,
    ):
        super().__init__()

        self.device = device
        self.encoder = am_layers.Encoder(
            am_layers.EncoderLayer(d_model, h, dropout), n_layer
        )
        embed_position = am_layers.PositionalEncoding(d_model, max_len=seq_size)
        embed_word = am_layers.EmbeddingLayer(
            input_size=num_vocab + 1, embed_size=d_model
        )
        self.embedding_by_feature = torch.nn.ModuleDict(
            {
                "position": embed_position,
                "word_token": embed_word,
            }
        )
        self.src_embed = am_layers.EncoderEmbedding(
            self.embedding_by_feature,
            seq_size,
            dropout,
            is_feature_based,
            device,
            freeze_embedding,
        )

        self.generator = am_layers.Generator(d_model, num_vocab)

    def forward(self, input_features):
        """
        Forward function: to be implemented after being inherited
        """
        src_mask = (input_features["word_token"] != 0).unsqueeze(-2)

        encoder_output = self.encoder(
            self.src_embed(input_features),
            src_mask,
        )

        output = self.generator(encoder_output).squeeze(-1)

        return output

    def embed_words(self, tokens):
        # tokens.to(self.device)
        tokens = torch.LongTensor(tokens).to(self.device)
        word_vectors = self.embedding_by_feature["word_token"](tokens)
        vector = torch.mean(word_vectors, 0)
        return vector

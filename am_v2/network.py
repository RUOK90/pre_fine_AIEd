"""
AM Network definition

Typical usage exmaple:

foo = TwoGeneratorModel(...)
"""

import torch

from am_v2 import config, layers


class Model(torch.nn.Module):
    """
    Single answer generator model
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
        super().__init__()

        self.device = device
        self.encoder = layers.Encoder(layers.EncoderLayer(d_model, h, dropout), n_layer)
        embed_position = layers.PositionalEncoding(d_model, max_len=seq_size)
        embed_qid = layers.EmbeddingLayer(
            input_size=q_size, embed_size=d_model
        )
        embed_part = layers.EmbeddingLayer(input_size=p_size, embed_size=d_model)
        embed_is_correct = layers.EmbeddingLayer(
            input_size=r_size, embed_size=d_model
        )
        embed_is_on_time = layers.EmbeddingLayer(
            input_size=t_size, embed_size=d_model
        )
        embed_elapsed_time = torch.nn.Linear(1, d_model)
        embed_lag_time = torch.nn.Linear(1, d_model)
        self.embedding_by_feature = torch.nn.ModuleDict({
            "position": embed_position,
            "qid": embed_qid,
            "part": embed_part,
            "is_correct": embed_is_correct,
            "is_on_time": embed_is_on_time,
            "elapsed_time": embed_elapsed_time,
            "lag_time": embed_lag_time
        })
        self.src_embed = layers.EncoderEmbedding(
            self.embedding_by_feature,
            seq_size,
            dropout,
            is_feature_based,
            device,
            freeze_embedding,
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
        self.generator = torch.nn.ModuleDict({
            pretrain_feature_name: layers.Generator(d_model, 1)
            for pretrain_feature_name in config.ARGS.pretrain_task
        })

        self.freeze_encoder_block = freeze_encoder_block

    def forward(self, input_features):
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
            time_output: Prediction on whether is on time (used)
            start_time_output: Prediction on start time (not used)
            correct_output: Prediction on response correctness (used)
            add_task: Prediction on additional task liks long option (unused)
        """

        # items that are not padded
        src_mask = (input_features['qid'] != 0).unsqueeze(-2)

        encoder_output = self.encoder(
            self.src_embed(input_features),
            src_mask,
        )

        output = {
            pretrain_feature_name: generator(encoder_output).squeeze(-1)
            for pretrain_feature_name, generator in self.generator.items()
        }

        return output


class ScoreModel(Model):
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
        self.lc_generator = layers.ScoreGenerator(d_model)
        self.rc_generator = layers.ScoreGenerator(d_model)

    def forward(self, input_features):
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
        src_mask = (input_features['qid'] != 0).unsqueeze(-2)

        encoder_output = self.encoder(
            self.src_embed(input_features),
            src_mask,
        )

        output = {
            'lc': self.lc_generator(encoder_output, src_mask),
            'rc': self.rc_generator(encoder_output, src_mask)
        }

        return output


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
        self.review_generator = layers.ReviewGenerator(d_model)

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
        src_mask = (input_features['qid'] != 0).unsqueeze(-2)

        encoder_output = self.encoder(
            self.src_embed(input_features),
            src_mask,
        )
        review_embedding = self.embedding_by_feature['qid'](review_qid)

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
        self.encoder = layers.Encoder(layers.EncoderLayer(d_model, h, dropout), n_layer)
        embed_position = layers.PositionalEncoding(d_model, max_len=seq_size)
        embed_word = layers.EmbeddingLayer(
            input_size=num_vocab + 1, embed_size=d_model
        )
        self.embedding_by_feature = torch.nn.ModuleDict({
            "position": embed_position,
            "word_token": embed_word,
        })
        self.src_embed = layers.EncoderEmbedding(
            self.embedding_by_feature,
            seq_size,
            dropout,
            is_feature_based,
            device,
            freeze_embedding,
        )

        self.generator = layers.Generator(d_model, num_vocab)

    def forward(self, input_features):
        """
        Forward function: to be implemented after being inherited
        """
        src_mask = (input_features['word_token'] != 0).unsqueeze(-2)

        encoder_output = self.encoder(
            self.src_embed(input_features),
            src_mask,
        )

        output = self.generator(encoder_output).squeeze(-1)

        return output

    def embed_words(self, tokens):
        # tokens.to(self.device)
        tokens = torch.LongTensor(tokens).to(self.device)
        word_vectors = self.embedding_by_feature['word_token'](tokens)
        vector = torch.mean(word_vectors, 0)
        return vector 


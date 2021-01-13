import copy
from torch.distributions.categorical import Categorical
from torch.distributions.normal import Normal
import torch.nn.functional as F
from transformers.models.electra.modeling_electra import *
from models.modeling_reformer import *
from models.performer_pytorch import *
from config import *


class ElectraAIEdPretrainModel(nn.Module):
    def __init__(self):
        super(ElectraAIEdPretrainModel, self).__init__()
        # set config
        if ARGS.model == "electra":
            dis_config = ElectraConfig()
            dis_config.embedding_size = ARGS.embedding_size
            dis_config.hidden_size = ARGS.hidden_size
            dis_config.intermediate_size = ARGS.intermediate_size
            dis_config.num_hidden_layers = ARGS.num_hidden_layers
            dis_config.num_attention_heads = ARGS.num_attention_heads
            dis_config.hidden_act = ARGS.hidden_act
            dis_config.hidden_dropout_prob = ARGS.hidden_dropout_prob
            dis_config.attention_probs_dropout_prob = ARGS.attention_probs_dropout_prob
            dis_config.pad_token_id = Const.PAD_VAL
            dis_config.max_position_embeddings = ARGS.max_seq_size

            gen_config = copy.deepcopy(dis_config)
            gen_config.hidden_size = int(dis_config.hidden_size / 4)
            gen_config.intermediate_size = int(dis_config.intermediate_size / 4)
            gen_config.num_attention_heads = int(dis_config.num_attention_heads / 4)

        elif ARGS.model == "electra-reformer":
            dis_config = ReformerConfig()
            dis_config.embedding_size = ARGS.hidden_size
            dis_config.axial_pos_embds = ARGS.axial_pos_embds
            dis_config.axial_pos_shape = tuple(ARGS.axial_pos_shape)
            dis_config.axial_pos_embds_dim = tuple(ARGS.axial_pos_embds_dim)
            dis_config.num_hidden_layers = len(ARGS.attn_layers)
            dis_config.hidden_size = ARGS.hidden_size
            dis_config.hidden_act = ARGS.hidden_act
            dis_config.hidden_dropout_prob = ARGS.hidden_dropout_prob
            dis_config.feed_forward_size = ARGS.feed_forward_size
            dis_config.attention_head_size = ARGS.attention_head_size
            dis_config.attn_layers = ARGS.attn_layers
            dis_config.num_attention_heads = ARGS.num_attention_heads
            dis_config.local_attn_chunk_length = ARGS.local_attn_chunk_length
            dis_config.local_attention_probs_dropout_prob = (
                ARGS.local_attention_probs_dropout_prob
            )
            dis_config.local_num_chunks_before = ARGS.local_num_chunks_before
            dis_config.local_num_chunks_after = ARGS.local_num_chunks_after
            dis_config.lsh_attn_chunk_length = ARGS.lsh_attn_chunk_length
            dis_config.lsh_attention_probs_dropout_prob = (
                ARGS.lsh_attention_probs_dropout_prob
            )
            dis_config.lsh_num_chunks_before = ARGS.lsh_num_chunks_before
            dis_config.lsh_num_chunks_after = ARGS.lsh_num_chunks_after
            dis_config.num_hashes = ARGS.num_hashes
            dis_config.num_buckets = ARGS.num_buckets
            dis_config.is_decoder = ARGS.is_decoder
            dis_config.use_cache = ARGS.use_cache
            dis_config.pad_token_id = Const.PAD_VAL
            dis_config.max_position_embeddings = ARGS.max_seq_size
            dis_config.reformer_seed = ARGS.random_seed

            gen_config = copy.deepcopy(dis_config)
            gen_config.feed_forward_size = int(dis_config.feed_forward_size / 4)
            gen_config.num_attention_heads = int(dis_config.num_attention_heads / 4)

        if ARGS.model == "electra-performer":
            dis_config = ElectraConfig()
            dis_config.axial_pos_embds = ARGS.axial_pos_embds
            dis_config.axial_pos_shape = tuple(ARGS.axial_pos_shape)
            dis_config.axial_pos_embds_dim = tuple(ARGS.axial_pos_embds_dim)
            dis_config.axial_norm_std = 1.0
            dis_config.embedding_size = ARGS.embedding_size
            dis_config.hidden_size = ARGS.hidden_size
            dis_config.feedforward_mult = ARGS.feedforward_mult
            dis_config.num_hidden_layers = ARGS.num_hidden_layers
            dis_config.num_attn_heads = ARGS.num_attn_heads
            dis_config.hidden_dropout_prob = ARGS.hidden_dropout_prob
            dis_config.attn_probs_dropout_prob = ARGS.attn_probs_dropout_prob
            dis_config.num_random_features = ARGS.num_random_features
            dis_config.feature_redraw_interval = ARGS.feature_redraw_interval
            dis_config.use_generalized_attn = ARGS.use_generalized_attn
            dis_config.use_scale_norm = ARGS.use_scale_norm
            dis_config.use_rezero = ARGS.use_rezero
            dis_config.use_glu = ARGS.use_glu
            dis_config.causal = ARGS.causal
            dis_config.cross_attend = ARGS.cross_attend
            dis_config.pad_token_id = Const.PAD_VAL
            dis_config.max_position_embeddings = ARGS.max_seq_size

            gen_config = copy.deepcopy(dis_config)
            gen_config.hidden_size = int(dis_config.hidden_size / 4)
            gen_config.num_attn_heads = int(dis_config.num_attn_heads / 4)

        self.embeds = ElectraAIEdEmbeddings(dis_config)
        self.gen_model = ElectraAIEdMaskedLM(gen_config)
        self.dis_model = ElectraAIEdPreTraining(dis_config)

    def forward(self, unmasked_features, masked_features, input_masks, padding_masks):
        attention_masks = (~padding_masks).long()
        gen_embeds = self.embeds(masked_features)
        gen_outputs = self.gen_model(gen_embeds, attention_masks)
        dis_inputs, dis_labels = get_dis_inputs_labels(
            unmasked_features, input_masks, gen_outputs
        )
        dis_embeds = self.embeds(dis_inputs)
        dis_outputs = self.dis_model(dis_embeds, attention_masks)

        return gen_outputs, dis_outputs, dis_labels


class ElectraAIEdFinetuneModel(nn.Module):
    def __init__(self):
        super(ElectraAIEdFinetuneModel, self).__init__()
        # set config
        if ARGS.model == "electra":
            config = ElectraConfig()
            config.embedding_size = ARGS.embedding_size
            config.hidden_size = ARGS.hidden_size
            config.intermediate_size = ARGS.intermediate_size
            config.num_hidden_layers = ARGS.num_hidden_layers
            config.num_attention_heads = ARGS.num_attention_heads
            config.hidden_act = ARGS.hidden_act
            config.hidden_dropout_prob = ARGS.hidden_dropout_prob
            config.attention_probs_dropout_prob = ARGS.attention_probs_dropout_prob
            config.pad_token_id = Const.PAD_VAL
            config.max_position_embeddings = ARGS.max_seq_size

        elif ARGS.model == "electra-reformer":
            config = ReformerConfig()
            config.embedding_size = ARGS.hidden_size
            config.axial_pos_embds = ARGS.axial_pos_embds
            config.axial_pos_shape = tuple(ARGS.axial_pos_shape)
            config.axial_pos_embds_dim = tuple(ARGS.axial_pos_embds_dim)
            config.num_hidden_layers = len(ARGS.attn_layers)
            config.hidden_size = ARGS.hidden_size
            config.hidden_act = ARGS.hidden_act
            config.hidden_dropout_prob = ARGS.hidden_dropout_prob
            config.feed_forward_size = ARGS.feed_forward_size
            config.attention_head_size = ARGS.attention_head_size
            config.attn_layers = ARGS.attn_layers
            config.num_attention_heads = ARGS.num_attention_heads
            config.local_attn_chunk_length = ARGS.local_attn_chunk_length
            config.local_attention_probs_dropout_prob = (
                ARGS.local_attention_probs_dropout_prob
            )
            config.local_num_chunks_before = ARGS.local_num_chunks_before
            config.local_num_chunks_after = ARGS.local_num_chunks_after
            config.lsh_attn_chunk_length = ARGS.lsh_attn_chunk_length
            config.lsh_attention_probs_dropout_prob = (
                ARGS.lsh_attention_probs_dropout_prob
            )
            config.lsh_num_chunks_before = ARGS.lsh_num_chunks_before
            config.lsh_num_chunks_after = ARGS.lsh_num_chunks_after
            config.num_hashes = ARGS.num_hashes
            config.num_buckets = ARGS.num_buckets
            config.is_decoder = ARGS.is_decoder
            config.use_cache = ARGS.use_cache
            config.pad_token_id = Const.PAD_VAL
            config.max_position_embeddings = ARGS.max_seq_size
            config.reformer_seed = ARGS.random_seed

        elif ARGS.model == "electra-performer":
            config = ElectraConfig()
            config.axial_pos_embds = ARGS.axial_pos_embds
            config.axial_pos_shape = tuple(ARGS.axial_pos_shape)
            config.axial_pos_embds_dim = tuple(ARGS.axial_pos_embds_dim)
            config.axial_norm_std = 1.0
            config.embedding_size = ARGS.embedding_size
            config.hidden_size = ARGS.hidden_size
            config.feedforward_mult = ARGS.feedforward_mult
            config.num_hidden_layers = ARGS.num_hidden_layers
            config.num_attn_heads = ARGS.num_attn_heads
            config.hidden_dropout_prob = ARGS.hidden_dropout_prob
            config.attn_probs_dropout_prob = ARGS.attn_probs_dropout_prob
            config.num_random_features = ARGS.num_random_features
            config.feature_redraw_interval = ARGS.feature_redraw_interval
            config.use_generalized_attn = ARGS.use_generalized_attn
            config.use_scale_norm = ARGS.use_scale_norm
            config.use_rezero = ARGS.use_rezero
            config.use_glu = ARGS.use_glu
            config.causal = ARGS.causal
            config.cross_attend = ARGS.cross_attend
            config.pad_token_id = Const.PAD_VAL
            config.max_position_embeddings = ARGS.max_seq_size

        self.embeds = ElectraAIEdEmbeddings(config)
        self.dis_model = ElectraAIEdSequenceClassification(config)

    def forward(self, unmasked_features, padding_masks):
        attention_masks = (~padding_masks).long()
        embeds = self.embeds(unmasked_features)
        outputs = self.dis_model(embeds, attention_masks)

        return outputs


class ElectraAIEdEmbeddings(ReformerPreTrainedModel):
    def __init__(self, config):
        super(ElectraAIEdEmbeddings, self).__init__(config)
        feature_embeds_dict = {}
        for feature in ARGS.input_features:
            if feature in Const.CATE_VARS:
                feature_embeds_dict[feature] = nn.Embedding(
                    Const.FEATURE_SIZE[feature] + 3,  # +3 for pad, cls, and mask
                    config.embedding_size,
                    padding_idx=config.pad_token_id,
                )
            elif feature in Const.CONT_VARS:
                feature_embeds_dict[feature] = nn.Linear(1, config.embedding_size)
        self.position_embeds = (
            AxialPositionEmbeddings(config)
            if ARGS.axial_pos_embds
            else nn.Embedding(config.max_position_embeddings, config.embedding_size)
        )
        self.feature_embeds = torch.nn.ModuleDict(feature_embeds_dict)

        self.LayerNorm = nn.LayerNorm(config.embedding_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.register_buffer(
            "position_ids", torch.arange(config.max_position_embeddings).expand((1, -1))
        )
        self.position_embedding_type = getattr(
            config, "position_embedding_type", "absolute"
        )

        self.init_weights()

    def forward(self, inputs):
        input_shape = inputs["qid"].shape
        seq_length = input_shape[1]
        position_ids = self.position_ids[:, :seq_length]

        for name, feature in inputs.items():
            if name in Const.CONT_VARS:
                inputs[name] = feature.unsqueeze(-1)

        embeds = 0
        for name, feature in inputs.items():
            embeds += self.feature_embeds[name](feature)
        embeds += self.position_embeds(position_ids)
        embeds = self.LayerNorm(embeds)
        embeds = self.dropout(embeds)

        return embeds


class ElectraAIEdMaskedLM(ElectraPreTrainedModel):
    def __init__(self, config):
        super(ElectraAIEdMaskedLM, self).__init__(config)
        if ARGS.model == "electra":
            self.electra = ElectraAIEdModel(config)
        elif ARGS.model == "electra-reformer":
            self.electra = ReformerAIEdModel(config)
        elif ARGS.model == "electra-performer":
            self.electra = PerformerAIEdModel(config)
        self.generator_predictions = ElectraAIEdGeneratorPredictions(config)

        heads_dict = {}
        for target in ARGS.targets:
            if target in Const.CATE_VARS:
                heads_dict[f"{target}_logit_head"] = nn.Linear(
                    config.embedding_size,
                    Const.FEATURE_SIZE[target] + 1,  # +1 for cls
                )
                heads_dict[f"{target}_output_head"] = nn.Softmax(dim=-1)
            elif target in Const.CONT_VARS:
                if ARGS.gen_cont_target_sampling == "normal":
                    heads_dict[f"{target}_logit_head"] = nn.Linear(
                        config.embedding_size, 2
                    )
                    heads_dict[f"{target}_output_head"] = nn.Identity()
                elif ARGS.gen_cont_target_sampling == "none":
                    heads_dict[f"{target}_logit_head"] = nn.Linear(
                        config.embedding_size, 1
                    )
                    if ARGS.time_output_func == "identity":
                        heads_dict[f"{target}_output_head"] = nn.Identity()
                    elif ARGS.time_output_func == "sigmoid":
                        heads_dict[f"{target}_output_head"] = nn.Sigmoid()

        self.heads = torch.nn.ModuleDict(heads_dict)

        self.init_weights()

    def forward(self, inputs, attention_masks):
        generator_sequence_output = self.electra(inputs, attention_masks)[0]
        prediction_scores = self.generator_predictions(generator_sequence_output)

        outputs = {}
        for target in ARGS.targets:
            if target in Const.CATE_VARS:
                logit = self.heads[f"{target}_logit_head"](prediction_scores)
                if ARGS.gen_cate_target_sampling == "categorical":
                    output = (
                        Categorical(self.heads[f"{target}_output_head"](logit)).sample()
                        + 1
                    )  # +1 since the var starts from 1, not 0
                elif ARGS.gen_cate_target_sampling == "none":
                    output = (
                        logit.max(dim=-1)[1] + 1
                    )  # +1 since the var starts from 1, not 0
            elif target in Const.CONT_VARS:
                logit = self.heads[f"{target}_logit_head"](prediction_scores).squeeze(
                    -1
                )
                if ARGS.gen_cont_target_sampling == "normal":
                    mu = logit[:, :, 0]
                    std = F.softplus(logit[:, :, 1])
                    logit = (mu, std)
                    output = torch.clamp(Normal(mu, std).sample(), min=0, max=1)
                elif ARGS.gen_cont_target_sampling == "none":
                    output = self.heads[f"{target}_output_head"](logit).squeeze(-1)
            outputs[target] = (logit, output)

        return outputs


class ElectraAIEdPreTraining(ElectraPreTrainedModel):
    def __init__(self, config):
        super(ElectraAIEdPreTraining, self).__init__(config)
        if ARGS.model == "electra":
            self.electra = ElectraAIEdModel(config)
        elif ARGS.model == "electra-reformer":
            self.electra = ReformerAIEdModel(config)
        elif ARGS.model == "electra-performer":
            self.electra = PerformerAIEdModel(config)
        self.discriminator_predictions = ElectraAIEdDiscriminatorPredictions(config)
        self.init_weights()

    def forward(self, inputs, attention_masks):
        discriminator_sequence_output = self.electra(inputs, attention_masks)[0]
        logits = self.discriminator_predictions(discriminator_sequence_output)
        outputs = torch.sigmoid(logits)

        return (logits, outputs)


class ElectraAIEdSequenceClassification(ElectraPreTrainedModel):
    def __init__(self, config):
        super(ElectraAIEdSequenceClassification, self).__init__(config)
        if ARGS.model == "electra":
            self.electra = ElectraAIEdModel(config)
        elif ARGS.model == "electra-reformer":
            self.electra = ReformerAIEdModel(config)
        elif ARGS.model == "electra-performer":
            self.electra = PerformerAIEdModel(config)
        self.classifier = ElectraAIEdClassificationHead(config)
        self.init_weights()

    def forward(self, inputs, attention_masks):
        sequence_output = self.electra(inputs, attention_masks)[0]
        outputs = self.classifier(sequence_output, attention_masks)

        return outputs


class ElectraAIEdModel(ElectraPreTrainedModel):
    def __init__(self, config):
        super(ElectraAIEdModel, self).__init__(config)
        if config.embedding_size != config.hidden_size:
            self.embeddings_project = nn.Linear(
                config.embedding_size, config.hidden_size
            )
        self.encoder = ElectraEncoder(config)
        self.config = config
        self.init_weights()

    def forward(self, inputs, attention_masks, head_mask=None):
        input_shape = inputs.shape[:-1]
        device = inputs.device

        extended_attention_mask = self.get_extended_attention_mask(
            attention_masks, input_shape, device
        )
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        hidden_states = inputs
        if hasattr(self, "embeddings_project"):
            hidden_states = self.embeddings_project(hidden_states)

        hidden_states = self.encoder(
            hidden_states=hidden_states,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
        )

        return hidden_states


class ReformerAIEdModel(ReformerPreTrainedModel):
    def __init__(self, config):
        super(ReformerAIEdModel, self).__init__(config)
        if config.embedding_size != config.hidden_size:
            self.embeddings_project = nn.Linear(
                config.embedding_size, config.hidden_size
            )
        self.encoder = ReformerEncoder(config)
        self.config = config
        self.init_weights()

    def forward(self, inputs, attention_masks, head_mask=None):
        use_cache = self.config.use_cache
        input_shape = inputs.size()  # noqa: F841
        device = inputs.device

        head_mask = self.get_head_mask(
            head_mask, self.config.num_hidden_layers, is_attention_chunked=True
        )
        orig_sequence_length = input_shape[-2]

        hidden_states = inputs
        if hasattr(self, "embeddings_project"):
            hidden_states = self.embeddings_project(hidden_states)

        hidden_states = self.encoder(
            hidden_states=hidden_states,
            attention_mask=attention_masks,
            head_mask=head_mask,
            use_cache=use_cache,
            orig_sequence_length=orig_sequence_length,
        )

        return hidden_states


class PerformerAIEdModel(ElectraPreTrainedModel):
    def __init__(self, config):
        super(PerformerAIEdModel, self).__init__(config)
        if config.embedding_size != config.hidden_size:
            self.embeddings_project = nn.Linear(
                config.embedding_size, config.hidden_size
            )
        self.encoder = Performer(
            dim=config.hidden_size,
            depth=config.num_hidden_layers,
            heads=config.num_attn_heads,
            local_attn_heads=0,
            local_window_size=None,
            causal=config.causal,
            ff_mult=config.feedforward_mult,
            nb_features=config.num_random_features,
            feature_redraw_interval=config.feature_redraw_interval,
            reversible=True,
            ff_chunks=1,
            generalized_attention=config.use_generalized_attn,
            kernel_fn=nn.ReLU(),
            qr_uniform_q=False,
            use_scalenorm=config.use_scale_norm,
            use_rezero=config.use_rezero,
            ff_glu=config.use_glu,
            ff_dropout=config.hidden_dropout_prob,
            attn_dropout=config.attn_probs_dropout_prob,
            cross_attend=config.cross_attend,
            no_projection=False,
        )
        self.config = config
        self.init_weights()

    def forward(self, inputs, attention_masks):
        hidden_states = inputs
        if hasattr(self, "embeddings_project"):
            hidden_states = self.embeddings_project(hidden_states)
        hidden_states = self.encoder(hidden_states, **{"mask": attention_masks.bool()})

        return (hidden_states,)


class ElectraAIEdGeneratorPredictions(nn.Module):
    def __init__(self, config):
        super(ElectraAIEdGeneratorPredictions, self).__init__()
        self.config = config
        if ARGS.model == "electra" or ARGS.model == "electra-performer":
            self.dense = nn.Linear(config.hidden_size, config.embedding_size)
        elif ARGS.model == "electra-reformer":
            self.dense = nn.Linear(2 * config.hidden_size, config.embedding_size)
        self.LayerNorm = nn.LayerNorm(config.embedding_size)

    def forward(self, generator_hidden_states):
        hidden_states = self.dense(generator_hidden_states)
        hidden_states = get_activation(self.config.hidden_act)(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)

        return hidden_states


class ElectraAIEdDiscriminatorPredictions(nn.Module):
    def __init__(self, config):
        super(ElectraAIEdDiscriminatorPredictions, self).__init__()
        self.config = config
        if ARGS.model == "electra" or ARGS.model == "electra-performer":
            self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        elif ARGS.model == "electra-reformer":
            self.dense = nn.Linear(2 * config.hidden_size, config.hidden_size)
        self.dense_prediction = nn.Linear(config.hidden_size, 1)

    def forward(self, discriminator_hidden_states):
        hidden_states = self.dense(discriminator_hidden_states)
        hidden_states = get_activation(self.config.hidden_act)(hidden_states)
        logits = self.dense_prediction(hidden_states).squeeze(-1)

        return logits


class ElectraAIEdClassificationHead(nn.Module):
    def __init__(self, config):
        super(ElectraAIEdClassificationHead, self).__init__()
        self.config = config
        if ARGS.model == "electra" or ARGS.model == "electra-performer":
            self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        elif ARGS.model == "electra-reformer":
            self.dense = nn.Linear(2 * config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        if ARGS.downstream_task == "score":
            proj_dict = {}
            proj_dict["lc"] = nn.Linear(config.hidden_size, 1)
            proj_dict["rc"] = nn.Linear(config.hidden_size, 1)
            self.proj = torch.nn.ModuleDict(proj_dict)

    def forward(self, inputs, attention_masks):
        if ARGS.finetune_output_func == "mean":
            attention_masks = attention_masks.unsqueeze(-1)
            x = torch.where(attention_masks == 1, inputs, torch.zeros_like(inputs))
            x = x.sum(1) / attention_masks.sum(1)
        elif ARGS.finetune_output_func == "cls":
            x = inputs[:, 0, :]

        x = self.dropout(x)
        x = self.dense(x)
        x = get_activation(self.config.hidden_act)(x)
        x = self.dropout(x)

        outputs = {}
        if ARGS.downstream_task == "score":
            for name, proj in self.proj.items():
                logit = proj(x).squeeze(-1)
                output = torch.sigmoid(logit)
                outputs[name] = (logit, output)

        return outputs


def get_dis_inputs_labels(unmasked_features, input_masks, gen_outputs):
    input_masks = input_masks.float()
    # you should cut gradient chain
    # get dis inputs
    dis_inputs = {}
    for feature in ARGS.input_features:
        if feature in ARGS.targets:
            if feature in Const.CATE_VARS:
                input = gen_outputs[feature][1]
                dis_inputs[feature] = (
                    (
                        unmasked_features[feature] * (1 - input_masks)
                        + input * input_masks
                    )
                    .long()
                    .detach()
                )
            elif feature in Const.CONT_VARS:
                input = gen_outputs[feature][1]
                dis_inputs[feature] = (
                    (
                        unmasked_features[feature] * (1 - input_masks)
                        + input * input_masks
                    )
                    .float()
                    .detach()
                )
        else:
            dis_inputs[feature] = unmasked_features[feature].detach()

    # get dis labels
    dis_labels = torch.ones_like(unmasked_features["qid"]).bool()
    for target in ARGS.targets:
        dis_labels &= unmasked_features[target] == dis_inputs[target]
    dis_labels = dis_labels.float().detach()  # 0: replaced, 1: original

    return dis_inputs, dis_labels

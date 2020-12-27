from transformers.models.electra.modeling_electra import *
from config import *


class ElectraAIEdPretrainModel(nn.Module):
    def __init__(self):
        super(ElectraAIEdPretrainModel, self).__init__()
        # set config
        gen_config = ElectraConfig()
        gen_config.embedding_size = ARGS.embedding_size
        gen_config.hidden_size = int(ARGS.hidden_size / 4)
        gen_config.intermediate_size = int(ARGS.intermediate_size / 4)
        gen_config.num_hidden_layers = ARGS.num_hidden_layers
        gen_config.num_attention_heads = int(ARGS.num_attention_heads / 4)
        gen_config.hidden_act = ARGS.hidden_act
        gen_config.hidden_dropout_prob = ARGS.hidden_dropout_prob
        gen_config.attention_probs_dropout_prob = ARGS.attention_probs_dropout_prob
        gen_config.pad_token_id = Const.PAD_VAL
        gen_config.max_position_embeddings = ARGS.max_seq_size

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


class ElectraAIEdPreTraining(ElectraPreTrainedModel):
    def __init__(self, config):
        super(ElectraAIEdPreTraining, self).__init__(config)
        self.electra = ElectraAIEdModel(config)
        self.discriminator_predictions = ElectraDiscriminatorPredictions(config)
        self.init_weights()

    def forward(
        self,
        inputs=None,
        attention_masks=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        discriminator_hidden_states = self.electra(
            inputs,
            attention_masks,
            token_type_ids,
            position_ids,
            head_mask,
            inputs_embeds,
            output_attentions,
            output_hidden_states,
            return_dict,
        )
        discriminator_sequence_output = discriminator_hidden_states[0]

        logits = self.discriminator_predictions(discriminator_sequence_output)
        outputs = torch.sigmoid(logits)

        return (logits, outputs)


class ElectraAIEdMaskedLM(ElectraPreTrainedModel):
    def __init__(self, config):
        super(ElectraAIEdMaskedLM, self).__init__(config)
        self.electra = ElectraAIEdModel(config)
        self.generator_predictions = ElectraGeneratorPredictions(config)

        heads_dict = {}
        for target in ARGS.targets:
            if target in Const.CATE_VARS:
                heads_dict[f"{target}_logit_head"] = nn.Linear(
                    config.embedding_size,
                    Const.FEATURE_SIZE[target] + 1,  # +1 for cls
                )
                heads_dict[f"{target}_output_head"] = nn.Softmax(dim=-1)
            elif target in Const.CONT_VARS:
                heads_dict[f"{target}_logit_head"] = nn.Linear(config.embedding_size, 1)
                if ARGS.time_output_func == "identity":
                    heads_dict[f"{target}_output_head"] = nn.Identity()
                elif ARGS.time_output_func == "sigmoid":
                    heads_dict[f"{target}_output_head"] = nn.Sigmoid()
        self.heads = torch.nn.ModuleDict(heads_dict)

        self.init_weights()

    def forward(
        self,
        inputs=None,
        attention_masks=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        generator_hidden_states = self.electra(
            inputs,
            attention_masks,
            token_type_ids,
            position_ids,
            head_mask,
            inputs_embeds,
            output_attentions,
            output_hidden_states,
            return_dict,
        )
        generator_sequence_output = generator_hidden_states[0]

        prediction_scores = self.generator_predictions(generator_sequence_output)

        outputs = {}
        for target in ARGS.targets:
            if target in Const.CATE_VARS:
                logit = self.heads[f"{target}_logit_head"](prediction_scores)
                output = (
                    logit.max(dim=-1)[1] + 1
                )  # +1 since the var starts from 1, not 0 (we don't user softmax)
            elif target in Const.CONT_VARS:
                logit = self.heads[f"{target}_logit_head"](prediction_scores).squeeze(
                    -1
                )
                output = self.heads[f"{target}_output_head"](logit).squeeze(-1)
            outputs[target] = (logit, output)

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

    def forward(
        self,
        inputs=None,
        attention_masks=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )
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
            hidden_states,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        return hidden_states


class ElectraAIEdEmbeddings(ElectraPreTrainedModel):
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
        self.position_embeds = nn.Embedding(
            config.max_position_embeddings, config.embedding_size
        )
        self.feature_embeds = torch.nn.ModuleDict(feature_embeds_dict)

        self.LayerNorm = nn.LayerNorm(config.embedding_size, eps=config.layer_norm_eps)
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


class ElectraAIEdFinetuneModel(nn.Module):
    def __init__(self):
        super(ElectraAIEdFinetuneModel, self).__init__()
        # set config
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

        self.embeds = ElectraAIEdEmbeddings(config)
        self.dis_model = ElectraAIEdSequenceClassification(config)

    def forward(self, unmasked_features, padding_masks):
        attention_masks = (~padding_masks).long()
        embeds = self.embeds(unmasked_features)
        outputs = self.dis_model(embeds, attention_masks)

        return outputs


class ElectraAIEdSequenceClassification(ElectraPreTrainedModel):
    def __init__(self, config):
        super(ElectraAIEdSequenceClassification, self).__init__(config)
        self.electra = ElectraAIEdModel(config)
        self.classifier = ElectraAIEdClassificationHead(config)
        self.init_weights()

    def forward(
        self,
        inputs=None,
        attention_masks=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        discriminator_hidden_states = self.electra(
            inputs,
            attention_masks,
            token_type_ids,
            position_ids,
            head_mask,
            inputs_embeds,
            output_attentions,
            output_hidden_states,
            return_dict,
        )
        sequence_output = discriminator_hidden_states[0]
        outputs = self.classifier(sequence_output, attention_masks)

        return outputs


class ElectraAIEdClassificationHead(nn.Module):
    def __init__(self, config):
        super(ElectraAIEdClassificationHead, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
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
        x = get_activation("gelu")(x)
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

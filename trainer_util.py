import copy
import torch
import torch.optim as optim
from itertools import chain

from optim_schedule import *
from config import *


def get_chained_dataloader(dataloader, num_chaines):
    chained_dataloader = chain.from_iterable([dataloader] * num_chaines)

    return chained_dataloader


def load_pretrained_weight(model, weight_path):
    model = copy.deepcopy(model)
    if weight_path is not None:
        model_dict = model.state_dict()
        pretrained_dict = torch.load(weight_path, map_location=ARGS.device)
        filtered_pretrained_dict = {
            name: params
            for name, params in pretrained_dict.items()
            if name in model_dict
        }
        model_dict.update(filtered_pretrained_dict)
        model.load_state_dict(model_dict)

    return model


def get_optimizer(model, optimizer):
    if optimizer == "scheduled":
        adam = optim.Adam(params=model.parameters(), lr=ARGS.lr)
        return ScheduledOpt(
            optimizer=adam, d_model=ARGS.hidden_size, n_warmup_steps=ARGS.warmup_steps
        )
    elif optimizer == "noam":
        adam = optim.Adam(
            params=model.parameters(),
            lr=ARGS.lr,
            betas=(0.9, 0.98),
            eps=1e-09,
        )
        return NoamOpt(
            model_size=ARGS.hidden_size,
            factor=1,
            warmup=ARGS.warmup_steps,
            optimizer=adam,
        )


def batch_to_device(batch):
    for group_name, group in batch.items():
        if group_name == "unmasked_feature" or group_name == "masked_feature":
            for name, feature in batch[group_name].items():
                if (
                    name == "qid"
                    or name == "part"
                    or name == "is_correct"
                    or name == "is_on_time"
                ):
                    batch[group_name][name] = feature.to(ARGS.device).long()

                elif name == "elapsed_time" or name == "lag_time":
                    batch[group_name][name] = feature.to(ARGS.device).float()

        elif group_name == "label":
            for name, feature in batch[group_name].items():
                if (
                    name == "qid"
                    or name == "part"
                    or name == "is_correct"
                    or name == "is_on_time"
                ):
                    batch[group_name][name] = feature.to(ARGS.device).long()
                elif (
                    name == "elapsed_time"
                    or name == "lag_time"
                    or name == "lc"
                    or name == "rc"
                ):
                    batch[group_name][name] = feature.to(ARGS.device).float()

        elif group_name == "input_mask":
            batch[group_name] = batch[group_name].to(ARGS.device).bool()

        elif group_name == "padding_mask":
            batch[group_name] = batch[group_name].to(ARGS.device).bool()

        elif group_name == "seq_size":
            batch[group_name] = batch[group_name].to(ARGS.device).float()

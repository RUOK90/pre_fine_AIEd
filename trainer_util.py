import copy
import torch
import torch.optim as optim

from optim_schedule import *
from config import *


def load_pretrained_weight(model, weight_path):
    model = copy.deepcopy(model)
    model_dict = model.state_dict()
    pretrained_dict = torch.load(weight_path, map_location=ARGS.device)
    filtered_pretrained_dict = {
        name: params for name, params in pretrained_dict.items() if name in model_dict
    }
    model_dict.update(filtered_pretrained_dict)
    model.load_state_dict(model_dict)

    return model


def get_optimizer(model, optimizer):
    if optimizer == "scheduled":
        adam = optim.Adam(params=model.parameters(), lr=ARGS.lr)
        return ScheduledOpt(
            optimizer=adam, d_model=ARGS.d_model, n_warmup_steps=ARGS.warmup_steps
        )
    elif optimizer == "noam":
        adam = optim.Adam(
            params=model.parameters(),
            lr=ARGS.lr,
            betas=(0.9, 0.98),
            eps=1e-09,
        )
        return NoamOpt(
            model_size=ARGS.d_model, factor=1, warmup=ARGS.warmup_steps, optimizer=adam
        )


def batch_to_device(batch):
    for name, feature in batch["input"].items():
        if (
            name == "qid"
            or name == "part"
            or name == "is_correct"
            or name == "is_on_time"
        ):
            batch["input"][name] = feature.to(ARGS.device).long()

        if name == "elapsed_time" or name == "lag_time":
            batch["input"][name] = feature.to(ARGS.device).float()

    for name, feature in batch["label"].items():
        batch["label"][name] = feature.to(ARGS.device).float()

    if "input_mask" in batch:
        batch["input_mask"] = batch["input_mask"].to(ARGS.device)
    if "padding_mask" in batch:
        batch["padding_mask"] = batch["padding_mask"].to(ARGS.device)
    if "seq_size" in batch:
        batch["seq_size"] = batch["seq_size"].to(ARGS.device)

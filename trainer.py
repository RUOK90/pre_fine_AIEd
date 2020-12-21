import wandb
import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

from trainer_util import *
from config import *


class Trainer:
    def __init__(self, model, pretrain_dataloaders, finetune_trainer):
        self._model = model
        self._model.to(ARGS.device)
        self._pretrain_dataloaders = pretrain_dataloaders
        self._finetune_trainer = finetune_trainer
        self._optim = get_optimizer(model, ARGS.optim)

        self._mse_loss = nn.MSELoss(reduction="mean")
        self._bce_loss = nn.BCEWithLogitsLoss(reduction="mean")
        self._l1_loss = nn.L1Loss(reduction="none")
        self._cur_epoch = 0

    def _pretrain(self, dataloader, mode):
        total_target_cnt = 0
        batch_results = {
            target: {
                "loss": [],
                "n_corrects": 0,
                "sigmoid_output": [],
                "label": [],
                "l1": [],
            }
            for target in ARGS.gen_targets
        }

        for step, batch in enumerate(tqdm(dataloader)):
            batch_to_device(batch)
            outputs = self._model(batch["input"])
            target_cnt = batch["input_mask"].sum().item()
            total_target_cnt += target_cnt
            batch_total_loss = 0

            for target, (output, sigmoid_output) in outputs.items():
                label = batch["label"][target].masked_select(batch["input_mask"])
                output = output.masked_select(batch["input_mask"])
                sigmoid_output = sigmoid_output.masked_select(batch["input_mask"])

                if target in ["is_correct", "is_on_time"]:
                    loss = self._bce_loss(output, label)
                elif target in ["elapsed_time", "lag_time"]:
                    loss = self._mse_loss(sigmoid_output, label)
                batch_total_loss += loss
                batch_results[target]["loss"].append(loss.item())

                if target in ["is_correct", "is_on_time"]:
                    pred = (sigmoid_output >= 0.5).float()
                    batch_results[target]["n_corrects"] += (pred == label).sum().item()
                    batch_results[target]["sigmoid_output"].extend(
                        sigmoid_output.detach().cpu().numpy()
                    )
                    batch_results[target]["label"].extend(label.detach().cpu().numpy())
                elif target in ["elapsed_time", "lag_time"]:
                    l1 = self._l1_loss(sigmoid_output, label)
                    batch_results[target]["l1"].extend(l1.detach().cpu().numpy())

            if self._model.training:
                if step / len(dataloader) > ARGS.cut_point:
                    break
                self._optim.update(batch_total_loss)

            if ARGS.debug_mode and step == 4:
                break

        # print and wandb output
        for target in ARGS.gen_targets:
            loss = np.mean(batch_results[target]["loss"])
            print(f"{mode} {target} loss: {loss:.4f}")
            if target in ["is_correct", "is_on_time"]:
                acc = batch_results[target]["n_corrects"] / total_target_cnt
                auc = roc_auc_score(
                    batch_results[target]["label"],
                    batch_results[target]["sigmoid_output"],
                )
                print(f"{mode} {target} acc: {acc:.4f}, auc: {auc:.4f}")
            elif target in ["elapsed_time", "lag_time"]:
                l1 = np.mean(batch_results[target]["l1"])
                print(f"{mode} {target} l1: {l1:.4f}")

            if ARGS.use_wandb:
                wandb_log_dict = {}
                wandb_log_dict[f"{mode} {target} loss"] = loss
                if target in ["is_correct", "is_on_time"]:
                    wandb_log_dict[f"{mode} {target} acc"] = acc
                    wandb_log_dict[f"{mode} {target} auc"] = auc
                elif target in ["elapsed_time", "lag_time"]:
                    wandb_log_dict[f"{mode} {target} l1"] = l1
                wandb.log(wandb_log_dict, step=self._cur_epoch)

    def _train(self):
        if ARGS.train_mode == "finetune_only":
            self._finetune_trainer._train(None, 0)
        elif ARGS.train_mode == "both" or ARGS.train_mode == "pretrain_only":
            for epoch in range(ARGS.num_pretrain_epochs):
                print(f"\nPretraining Epoch: {epoch:03d}")
                self._cur_epoch = epoch

                # set random seed
                set_random_seed(ARGS.random_seed + epoch)

                # pretrain train
                self._model.train()
                print(f"pretraining train")
                self._pretrain(self._pretrain_dataloaders["train"], "train")

                # pretrain val
                print("pretraining validation")
                with torch.no_grad():
                    self._model.eval()
                    self._pretrain(self._pretrain_dataloaders["val"], "val")

                # save pretrained model
                pretrained_weight_path = f"{ARGS.weight_path}/{epoch}.pt"
                torch.save(self._model.state_dict(), pretrained_weight_path)

                if ARGS.train_mode == "both":
                    # fine tune
                    print(f"finetuning after the pretraining")
                    self._finetune_trainer._train(pretrained_weight_path, epoch)

import wandb
import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
from torch.distributions.normal import Normal
import numpy as np
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
from itertools import islice

from trainer_util import *
from config import *


class Trainer:
    def __init__(self, pretrain_model, pretrain_dataloaders, finetune_trainer):
        self._pretrain_model = pretrain_model
        self._pretrain_model.to(ARGS.device)
        self._pretrain_dataloaders = pretrain_dataloaders
        self._finetune_trainer = finetune_trainer
        self._optim = get_optimizer(pretrain_model, ARGS.optim)

        self._mse_loss = nn.MSELoss(reduction="mean")
        self._bce_loss = nn.BCEWithLogitsLoss(reduction="mean")
        self._ce_loss = nn.CrossEntropyLoss(reduction="mean")
        self._l1_loss = nn.L1Loss(reduction="none")
        self._cur_n_eval = 0

    def _pretrain(self, dataloader, mode):
        total_gen_target_cnt = 0
        total_dis_target_cnt = 0
        # results for generator
        batch_results = {
            target: {
                "loss": [],
                "n_corrects": 0,
                "l1": [],
            }
            for target in ARGS.targets
        }
        # results for discriminator
        batch_results["dis"] = {
            "loss": [],
            "n_corrects": 0,
            "output": [],
            "label": [],
        }

        for step, batch in enumerate(tqdm(dataloader)):
            batch_to_device(batch)
            gen_outputs, dis_outputs, dis_labels = self._pretrain_model(
                batch["unmasked_feature"],
                batch["masked_feature"],
                batch["input_mask"],
                batch["padding_mask"],
            )
            total_gen_target_cnt += batch["input_mask"].sum().item()
            gen_total_loss = 0

            # for generator outputs
            for target, (logit, output) in gen_outputs.items():
                label = batch["label"][target].masked_select(batch["input_mask"])
                output = output.masked_select(batch["input_mask"])
                if target in Const.CATE_VARS:
                    logit = logit.masked_select(batch["input_mask"].unsqueeze(-1)).view(
                        -1, logit.shape[-1]
                    )
                    loss = self._ce_loss(logit, label)
                elif target in Const.CONT_VARS:
                    logit = logit.masked_select(batch["input_mask"])
                    if ARGS.time_loss == "bce":
                        loss = self._bce_loss(logit, label)
                    elif ARGS.time_loss == "mse":
                        loss = ARGS.time_loss_lambda * self._mse_loss(output, label)
                gen_total_loss += loss
                batch_results[target]["loss"].append(loss.item())

                if target in Const.CATE_VARS:
                    batch_results[target]["n_corrects"] += (
                        (output == (label + 1)).sum().item()
                    )
                elif target in Const.CONT_VARS:
                    l1 = self._l1_loss(output, label)
                    batch_results[target]["l1"].extend(l1.detach().cpu().numpy())

            # for discriminator outputs
            if dis_outputs is not None:
                total_dis_target_cnt += (~batch["padding_mask"]).sum().item()
                label = dis_labels.masked_select(~batch["padding_mask"])
                logit = dis_outputs[0].masked_select(~batch["padding_mask"])
                output = dis_outputs[1].masked_select(~batch["padding_mask"])
                dis_loss = self._bce_loss(logit, label)
                batch_results["dis"]["loss"].append(dis_loss.item())
                batch_results["dis"]["n_corrects"] += (
                    (output.round() == label).sum().item()
                )
                batch_results["dis"]["output"].extend(output.detach().cpu().numpy())
                batch_results["dis"]["label"].extend(label.detach().cpu().numpy())

            if self._pretrain_model.training:
                self._optim.update(gen_total_loss + ARGS.dis_lambda * dis_loss)

            if ARGS.debug_mode and step == ARGS.pretrain_update_steps:
                break

        # print and wandb output
        # for generator outputs
        for target in ARGS.targets:
            loss = np.mean(batch_results[target]["loss"])
            print(f"{mode} {target} loss: {loss:.4f}")
            if target in Const.CATE_VARS:
                acc = batch_results[target]["n_corrects"] / total_gen_target_cnt
                print(f"{mode} {target} acc: {acc:.4f}")
            elif target in Const.CONT_VARS:
                l1 = np.mean(batch_results[target]["l1"])
                print(f"{mode} {target} l1: {l1:.4f}")

            if ARGS.use_wandb:
                wandb_log_dict = {}
                wandb_log_dict[f"{mode} {target} loss"] = loss
                if target in Const.CATE_VARS:
                    wandb_log_dict[f"{mode} {target} acc"] = acc
                elif target in Const.CONT_VARS:
                    wandb_log_dict[f"{mode} {target} l1"] = l1
                wandb.log(wandb_log_dict, step=self._cur_n_eval)

        # for discriminator outputs
        if dis_outputs is not None:
            loss = np.mean(batch_results["dis"]["loss"])
            print(f"{mode} dis loss: {loss:.4f}")
            acc = batch_results["dis"]["n_corrects"] / total_dis_target_cnt
            auc = roc_auc_score(
                batch_results["dis"]["label"],
                batch_results["dis"]["output"],
            )
            print(f"{mode} dis acc: {acc:.4f}, auc: {auc:.4f}")

            if ARGS.use_wandb:
                wandb_log_dict = {}
                wandb_log_dict[f"{mode} dis loss"] = loss
                wandb_log_dict[f"{mode} dis acc"] = acc
                wandb_log_dict[f"{mode} dis auc"] = auc
                wandb_log_dict[f"{mode} dis label"] = np.mean(
                    batch_results["dis"]["label"]
                )
                wandb.log(wandb_log_dict, step=self._cur_n_eval)

    def _train(self):
        if ARGS.train_mode == "finetune_only":
            self._finetune_trainer._train(None, 0, True)
        elif ARGS.train_mode == "finetune_only_from_pretrained_weight":
            if ARGS.pretrained_weight_n_eval == -1:
                # only finetune from the whole pretrained weights
                # for n_eval in range(ARGS.pretrain_max_num_evals):
                for n_eval in [1, 2, 3, 5, 6, 7, 8]:
                    print(f"\nPretraining n_eval: {n_eval:03d}")
                    pretrained_weight_path = f"{ARGS.weight_path}/{n_eval}.pt"
                    self._finetune_trainer._train(pretrained_weight_path, n_eval, False)
                # get finetune test performance
                self._get_best_val_n_eval()
            else:
                n_eval = ARGS.pretrained_weight_n_eval
                pretrained_weight_path = f"{ARGS.weight_path}/{n_eval}.pt"
                self._finetune_trainer._train(pretrained_weight_path, n_eval, True)
        elif ARGS.train_mode == "both" or ARGS.train_mode == "pretrain_only":
            chained_train_dataloader = get_chained_dataloader(
                self._pretrain_dataloaders["train"], ARGS.pretrain_max_num_evals
            )
            for n_eval in range(ARGS.pretrain_max_num_evals):
                print(f"\nPretraining n_eval: {n_eval:03d}")
                self._cur_n_eval = n_eval

                # set random seed
                set_random_seed(ARGS.random_seed + n_eval)

                # pretrain train
                self._pretrain_model.train()
                print(f"pretraining train")
                self._pretrain(
                    islice(chained_train_dataloader, ARGS.pretrain_update_steps),
                    "train",
                )

                # pretrain val
                print("pretraining validation")
                with torch.no_grad():
                    self._pretrain_model.eval()
                    self._pretrain(self._pretrain_dataloaders["val"], "val")

                # save pretrained model
                pretrained_weight_path = f"{ARGS.weight_path}/{n_eval}.pt"
                torch.save(self._pretrain_model.state_dict(), pretrained_weight_path)

                if ARGS.train_mode == "both":
                    # finetune
                    print(f"finetuning after the pretraining")
                    self._finetune_trainer._train(pretrained_weight_path, n_eval, False)

            if ARGS.train_mode == "both":
                # get finetune test performance
                self._get_best_val_n_eval()

    def _get_best_val_n_eval(self):
        if ARGS.downstream_task == "score":
            best_n_eval = np.argmin(self._finetune_trainer._pretrain_best_val_perfs)
        print(f"best val pretrain n_eval: {best_n_eval}")

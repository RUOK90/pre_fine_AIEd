import gc
import time
import wandb
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from itertools import islice
import pickle

from trainer_util import *
from config import *


class FineTuneTrainer:
    def __init__(self, model, dataloaders):
        self._dummy_model = model
        self._dummy_model.to(ARGS.device)
        self._model = None
        self._optim = None
        self._dataloaders = dataloaders

        self._bce_loss = nn.BCEWithLogitsLoss(reduction="mean")
        self._mse_loss = nn.MSELoss(reduction="mean")
        self._l1_loss = nn.L1Loss(reduction="none")
        self._best_val_perf = None
        self._test_perf = None
        self._finetuned_weight_path = None
        self._val_best_renewal = False
        self._pretrain_best_val_perfs = []
        self._patience = ARGS.finetune_patience
        self._cur_n_eval = 0

    def _score_forward(self, dataloader, cross_num, mode):
        batch_results = {"loss": [], "l1": [], "lc_l1": [], "rc_l1": []}

        for step, batch in enumerate(tqdm(dataloader)):
            batch_to_device(batch)
            outputs = self._model(batch["unmasked_feature"], batch["padding_mask"])
            lc_logit, lc_output = outputs["lc"]
            rc_logit, rc_output = outputs["rc"]
            lc_pred = lc_output * Const.SCORE_SCALING_FACTOR
            rc_pred = rc_output * Const.SCORE_SCALING_FACTOR
            total_pred = lc_pred + rc_pred
            lc_l1 = self._l1_loss(lc_pred, batch["label"]["lc"])
            rc_l1 = self._l1_loss(rc_pred, batch["label"]["rc"])
            l1 = self._l1_loss(total_pred, batch["label"]["lc"] + batch["label"]["rc"])
            batch_results["lc_l1"].extend(lc_l1.detach().cpu().numpy())
            batch_results["rc_l1"].extend(rc_l1.detach().cpu().numpy())
            batch_results["l1"].extend(l1.detach().cpu().numpy())
            if ARGS.score_loss == "bce":
                lc_loss = self._bce_loss(
                    lc_logit, batch["label"]["lc"] / Const.SCORE_SCALING_FACTOR
                )
                rc_loss = self._bce_loss(
                    rc_logit, batch["label"]["rc"] / Const.SCORE_SCALING_FACTOR
                )
            elif ARGS.score_loss == "mse":
                lc_loss = self._mse_loss(
                    lc_output, batch["label"]["lc"] / Const.SCORE_SCALING_FACTOR
                )
                rc_loss = self._mse_loss(
                    rc_output, batch["label"]["rc"] / Const.SCORE_SCALING_FACTOR
                )

            batch_total_loss = lc_loss + rc_loss
            batch_results["loss"].append(batch_total_loss.item())

            if self._model.training:
                self._optim.update(batch_total_loss)

            if ARGS.debug_mode and step == ARGS.finetune_update_steps:
                break

        # print, save model, and wandb output
        loss = np.mean(batch_results["loss"])
        lc_mae = np.mean(batch_results["lc_l1"])
        rc_mae = np.mean(batch_results["rc_l1"])
        mae = np.mean(batch_results["l1"])

        if mode == "val":
            if mae < self._best_val_perf[cross_num]:
                self._best_val_perf[cross_num] = mae
                self._val_best_renewal = True
                self._patience = ARGS.finetune_patience
            else:
                self._patience -= 1

        elif mode == "test":
            self._test_perf[cross_num] = mae

        print(
            f"{mode} {ARGS.downstream_task}_{cross_num} loss: {loss:.4f}, mae: {mae:.4f}, lc_mae: {lc_mae:.4f}, rc_mae: {rc_mae:.4f}"
        )

        if ARGS.use_finetune_wandb and cross_num == 0:
            wandb_log_dict = {}
            wandb_log_dict[f"{mode} {ARGS.downstream_task}_{cross_num} loss"] = loss
            wandb_log_dict[f"{mode} {ARGS.downstream_task}_{cross_num} mae"] = mae
            wandb_log_dict[f"{mode} {ARGS.downstream_task}_{cross_num} lc_mae"] = lc_mae
            wandb_log_dict[f"{mode} {ARGS.downstream_task}_{cross_num} rc_mae"] = rc_mae
            wandb.log(wandb_log_dict, step=self._cur_n_eval)

    def _train(self, pretrained_weight_path, pretrain_n_eval, do_test):
        if ARGS.downstream_task == "score":
            self._best_val_perf = [np.inf for i in range(ARGS.num_cross_folds)]
            self._test_perf = [np.inf for i in range(ARGS.num_cross_folds)]

        for cross_num, dataloaders in self._dataloaders.items():
            set_random_seed(ARGS.random_seed)
            chained_train_dataloader = get_chained_dataloader(
                dataloaders["train"], ARGS.finetune_max_num_evals
            )
            self._model = load_pretrained_weight(
                self._dummy_model, pretrained_weight_path, False
            )
            self._optim = get_optimizer(self._model, ARGS.optim)

            gc_n_evals = 50
            for n_eval in range(ARGS.finetune_max_num_evals):
                print(f"\nCross Num: {cross_num:03d}, Finetuning n_eval: {n_eval:03d}")
                self._cur_n_eval = n_eval

                # train
                self._model.train()
                self._score_forward(
                    islice(chained_train_dataloader, ARGS.finetune_update_steps),
                    cross_num,
                    "train",
                )

                # val
                with torch.no_grad():
                    self._model.eval()
                    self._score_forward(dataloaders["val"], cross_num, "val")

                if self._patience == 0:
                    break

                # save model
                if do_test and self._val_best_renewal:
                    self._finetuned_weight_path = f"{ARGS.weight_path}/{ARGS.downstream_task}_{pretrain_n_eval}_{cross_num}.pt"
                    torch.save(self._model.state_dict(), self._finetuned_weight_path)
                    self._val_best_renewal = False

                if (n_eval + 1) % gc_n_evals == 0:
                    gc.collect()

            # test
            if do_test:
                with torch.no_grad():
                    self._model.load_state_dict(torch.load(self._finetuned_weight_path))
                    self._model.eval()
                    self._score_forward(dataloaders["test"], cross_num, "test")

            gc.collect()

        # mean over cross validation split print and wandb output
        # print val outputs
        finetune_mean_best_val_perf = np.mean(self._best_val_perf)
        self._pretrain_best_val_perfs.append(finetune_mean_best_val_perf)
        print(
            f"mean_best_{ARGS.downstream_task}_val_perf: {finetune_mean_best_val_perf}"
        )
        if ARGS.use_wandb:
            wandb_log_dict = {}
            for i in range(ARGS.num_cross_folds):
                wandb_log_dict[
                    f"{i}_best_{ARGS.downstream_task}_val_perf"
                ] = self._best_val_perf[i]
            wandb_log_dict[
                f"mean_best_{ARGS.downstream_task}_val_perf"
            ] = finetune_mean_best_val_perf
            if (
                ARGS.train_mode == "finetune_only_from_pretrained_weight"
                and ARGS.pretrained_weight_n_eval != -1
            ):
                wandb_step = 0
            else:
                wandb_step = pretrain_n_eval
            wandb.log(wandb_log_dict, step=wandb_step)

        # print test outputs
        if do_test:
            finetune_mean_test_perf = np.mean(self._test_perf)
            print(f"mean_{ARGS.downstream_task}_test_perf: {finetune_mean_test_perf}")
            if ARGS.use_wandb:
                wandb_log_dict = {}
                for i in range(ARGS.num_cross_folds):
                    wandb_log_dict[
                        f"{i}_{ARGS.downstream_task}_test_perf"
                    ] = self._test_perf[i]
                wandb_log_dict[
                    f"mean_{ARGS.downstream_task}_test_perf"
                ] = finetune_mean_test_perf
                wandb.log(wandb_log_dict, step=0)

    def _inference(self):
        results = {}
        for cross_num, dataloaders in self._dataloaders.items():
            finetuned_weight_path = f"{ARGS.weight_path}/score_59_{cross_num}.pt"
            self._dummy_model.load_state_dict(torch.load(finetuned_weight_path))
            # val
            with torch.no_grad():
                self._dummy_model.eval()
                batch_results = {
                    "l1": [],
                    "lc_l1": [],
                    "rc_l1": [],
                    "label": [],
                    "lc_label": [],
                    "rc_label": [],
                }
                for step, batch in enumerate(tqdm(dataloaders["all"])):
                    batch_to_device(batch)
                    outputs = self._dummy_model(
                        batch["unmasked_feature"], batch["padding_mask"]
                    )
                    lc_logit, lc_output = outputs["lc"]
                    rc_logit, rc_output = outputs["rc"]
                    lc_pred = lc_output * Const.SCORE_SCALING_FACTOR
                    rc_pred = rc_output * Const.SCORE_SCALING_FACTOR
                    total_pred = lc_pred + rc_pred
                    lc_l1 = self._l1_loss(lc_pred, batch["label"]["lc"])
                    rc_l1 = self._l1_loss(rc_pred, batch["label"]["rc"])
                    l1 = self._l1_loss(
                        total_pred, batch["label"]["lc"] + batch["label"]["rc"]
                    )
                    batch_results["lc_l1"].extend(lc_l1.detach().cpu().numpy())
                    batch_results["rc_l1"].extend(rc_l1.detach().cpu().numpy())
                    batch_results["l1"].extend(l1.detach().cpu().numpy())
                    label = batch["label"]["lc"] + batch["label"]["rc"]
                    batch_results["lc_label"].extend(
                        batch["label"]["lc"].detach().cpu().numpy()
                    )
                    batch_results["rc_label"].extend(
                        batch["label"]["rc"].detach().cpu().numpy()
                    )
                    batch_results["label"].extend(label.detach().cpu().numpy())

                results[cross_num] = batch_results

        with open("/private/results/results.pkl", "wb") as f_w:
            pickle.dump(results, f_w, pickle.HIGHEST_PROTOCOL)

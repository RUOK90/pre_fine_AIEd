import wandb
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm

from trainer_util import *
from config import *


class FineTuneTrainer:
    def __init__(self, model, dataloaders):
        self._model = model
        self._model.to(ARGS.device)
        self._dataloaders = dataloaders
        self._optim = get_optimizer(model, ARGS.optim)

        self._bce_loss = nn.BCEWithLogitsLoss(reduction="mean")
        self._l1_loss = nn.L1Loss(reduction="none")
        self._cur_epoch = None
        self._best_val_perf = None
        self._test_perf = None
        self._finetuned_weight_path = None

    def _score_forward(self, dataloader, cross_num, mode, pretrain_epoch):
        batch_results = {"loss": [], "l1": [], "lc_l1": [], "rc_l1": []}

        for step, batch in enumerate(tqdm(dataloader)):
            batch_to_device(batch)
            outputs = self._model(batch["input"])
            lc_output, lc_sigmoid_output = outputs["lc"]
            rc_output, rc_sigmoid_output = outputs["rc"]
            lc_pred = lc_sigmoid_output * Const.SCORE_SCALING_FACTOR
            rc_pred = rc_sigmoid_output * Const.SCORE_SCALING_FACTOR
            total_pred = lc_pred + rc_pred
            lc_l1 = self._l1_loss(lc_pred, batch["label"]["lc"])
            rc_l1 = self._l1_loss(rc_pred, batch["label"]["rc"])
            l1 = self._l1_loss(total_pred, batch["label"]["lc"] + batch["label"]["rc"])
            batch_results["lc_l1"].extend(lc_l1.detach().cpu().numpy())
            batch_results["rc_l1"].extend(rc_l1.detach().cpu().numpy())
            batch_results["l1"].extend(l1.detach().cpu().numpy())
            lc_loss = self._bce_loss(
                lc_output, batch["label"]["lc"] / Const.SCORE_SCALING_FACTOR
            )
            rc_loss = self._bce_loss(
                rc_output, batch["label"]["rc"] / Const.SCORE_SCALING_FACTOR
            )
            batch_total_loss = lc_loss + rc_loss
            batch_results["loss"].append(batch_total_loss.item())

            if self._model.training:
                self._optim.update(batch_total_loss)

            if ARGS.debug_mode and step == 4:
                break

        # print, save model, and wandb output
        loss = np.mean(batch_results["loss"])
        lc_mae = np.mean(batch_results["lc_l1"])
        rc_mae = np.mean(batch_results["rc_l1"])
        mae = np.mean(batch_results["l1"])

        if mode == "train" or mode == "val":
            wandb_step = self._cur_epoch
        elif mode == "test":
            self._test_perf[cross_num] = mae
            wandb_step = cross_num

        if mode == "val":
            if mae < self._best_val_perf[cross_num]:
                self._best_val_perf[cross_num] = mae
                # save model
                self._finetuned_weight_path = f"{ARGS.weight_path}/{ARGS.downstream_task}_{pretrain_epoch}_{cross_num}.pt"
                torch.save(self._model.state_dict(), self._finetuned_weight_path)

            # if ARGS.use_wandb:
            #     wandb.log(
            #         {
            #             f"{mode} {ARGS.downstream_task}_{cross_num} min_mae": self._best_val_perf[
            #                 cross_num
            #             ]
            #         },
            #         step=wandb_step,
            #         commit=False,
            #     )
        print(
            f"{mode} {ARGS.downstream_task}_{cross_num} loss: {loss:.4f}, mae: {mae:.4f}, lc_mae: {lc_mae:.4f}, rc_mae: {rc_mae:.4f}"
        )

        # if ARGS.use_wandb:
        #     wandb.log(
        #         {
        #             f"{mode} {ARGS.downstream_task}_{cross_num} loss": loss,
        #             f"{mode} {ARGS.downstream_task}_{cross_num} mae": mae,
        #             f"{mode} {ARGS.downstream_task}_{cross_num} lc_mae": lc_mae,
        #             f"{mode} {ARGS.downstream_task}_{cross_num} rc_mae": rc_mae,
        #         },
        #         step=wandb_step,
        #         commit=False,
        #     )

    def _train(self, pretrained_weight_path, pretrain_epoch):
        self._best_val_perf = [np.inf for i in range(ARGS.num_cross_folds)]
        self._test_perf = [np.inf for i in range(ARGS.num_cross_folds)]
        for cross_num, dataloaders in self._dataloaders.items():
            set_random_seed(ARGS.random_seed)
            model = load_pretrained_weight(self._model, pretrained_weight_path)
            for epoch in range(ARGS.num_finetune_epochs):
                print(f"\nCross Num: {cross_num:03d}, Finetuning Epoch: {epoch:03d}")
                self._cur_epoch = epoch

                # train
                self._model.train()
                self._score_forward(
                    dataloaders["train"], cross_num, "train", pretrain_epoch
                )

                # val
                with torch.no_grad():
                    self._model.eval()
                    self._score_forward(
                        dataloaders["val"], cross_num, "val", pretrain_epoch
                    )

            # test
            with torch.no_grad():
                self._model.load_state_dict(torch.load(self._finetuned_weight_path))
                self._model.eval()
                self._score_forward(
                    dataloaders["test"], cross_num, "test", pretrain_epoch
                )

        # mean over cross validation split print and wandb output
        finetune_mean_best_val_perf = np.mean(self._best_val_perf)
        finetune_mean_test_perf = np.mean(self._test_perf)

        print(
            f"mean_best_{ARGS.downstream_task}_val_perf: {finetune_mean_best_val_perf}, mean_{ARGS.downstream_task}_test_perf: {finetune_mean_test_perf}"
        )

        if ARGS.use_wandb:
            wandb_log_dict = {}
            for i in range(ARGS.num_cross_folds):
                wandb_log_dict[
                    f"{i}_best_{ARGS.downstream_task}_val_perf"
                ] = self._best_val_perf[i]
                wandb_log_dict[
                    f"{i}_{ARGS.downstream_task}_test_perf"
                ] = self._test_perf[i]
            wandb_log_dict[
                f"mean_best_{ARGS.downstream_task}_val_perf"
            ] = finetune_mean_best_val_perf
            wandb_log_dict[
                f"mean_{ARGS.downstream_task}_test_perf"
            ] = finetune_mean_test_perf
            wandb.log(wandb_log_dict, step=pretrain_epoch)

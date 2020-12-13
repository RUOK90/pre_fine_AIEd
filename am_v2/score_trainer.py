"""
Trainer class and functions for score prediction fine-tuning
"""
import time

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
import wandb

from am_v2 import config
from am_v2.optim_schedule import ScheduledOptim

TRAIN_MODE = "train"
VALIDATION_MODE = "validation"
TEST_MODE = "test"


class Trainer:
    """Trainer class"""

    def __init__(
        self,
        model,
        debug_mode,
        num_epochs,
        weight_path,
        d_model,
        warmup_steps,
        learning_rate,
        device,
        score_gen,
        lambda1,
        lambda2,
    ):
        self._device = device
        self._model = model
        self._debug_mode = debug_mode
        self._weight_path = weight_path
        self._num_epochs = num_epochs
        self._score_gen = score_gen
        self._weighted_loss = False

        if config.ARGS.score_loss == "l1":
            self._loss = nn.L1Loss()
        elif config.ARGS.score_loss == "l2":
            self._loss = nn.MSELoss()
        elif config.ARGS.score_loss == "bce":
            self._loss = nn.BCELoss()

        self._l1loss = nn.L1Loss()
        self._l2loss = nn.MSELoss()
        self._bce_loss = nn.BCELoss(reduction="none")
        self._optimizer = torch.optim.Adam(
            self._model.parameters(), lr=learning_rate, betas=(0.9, 0.98), eps=1e-9
        )
        self._schedule_optimizer = ScheduledOptim(
            self._optimizer, d_model, n_warmup_steps=warmup_steps
        )

        self.lambda1 = lambda1
        self.lambda2 = lambda2

        self._train_mae = -1
        self._train_loss = 0.
        self._validation_mae = -1
        self._validation_min_mae = 500
        self._validation_min_mae_epoch = -1
        self._test_mae = -1

        self._patience = 10

    def _set_min_loss(self, epoch):
        """Set minimum loss"""
        if self._validation_min_mae > self._validation_mae:
            self._validation_min_mae = self._validation_mae
            self._validation_min_mae_epoch = epoch
            self._patience = 10
        else:
            self._patience -= 1

    def _update(self, loss):
        """Updates gradients and lr"""
        self._schedule_optimizer.zero_grad()
        loss.backward()
        self._schedule_optimizer.step_and_update_lr()

    def _write_wandb(self):
        """Writes train logs to WanDB"""
        wandb.log(
            {
                f"Train MAE": self._train_mae,
                f"Train loss": self._train_loss,
                f"Val MAE": self._validation_mae,
                f"Best Val MAE": self._validation_min_mae,
                f"Best Val Epoch": self._validation_min_mae_epoch,
                f"Test MAE": self._test_mae,
            }
        )

    def _inference(self, generator, mode):
        """
        Training code also used for inference when mode is dev or test
        """
        start_time = time.time()
        toeic_loss_list = []
        lc_loss_list = []
        rc_loss_list = []
        total_count = 0

        gen_iter = generator
        if not config.ARGS.use_score_val:
            gen_iter = tqdm(generator)
        for _, (input_features, lc_score, rc_score) in enumerate(gen_iter):
            input_features = {
                name: feature.to(self._device)
                for name, feature in input_features.items()
            }
            input_features = {
                'qid': input_features['qid'].long(),
                'part': input_features['part'].long(),
                'is_correct': input_features['is_correct'].long(),
                # 'is_on_time': input_features['is_on_time'].long(),
                'elapsed_time': input_features['elapsed_time'].unsqueeze(-1),
                'lag_time': input_features['lag_time'].unsqueeze(-1)
            }
            output = self._model(input_features)
            predicted_lc, predicted_rc = output['lc'], output['rc']
            predicted_toeic = (
                predicted_lc + predicted_rc
            ) * config.ARGS.score_scaling_factor

            _toeic = (lc_score + rc_score).view(-1, 1).float().to(self._device)
            _lc = (
                lc_score.view(-1, 1).float().to(self._device)
                / config.ARGS.score_scaling_factor
            )
            _rc = (
                rc_score.view(-1, 1).float().to(self._device)
                / config.ARGS.score_scaling_factor
            )

            toeic_loss = self._l1loss(predicted_toeic, _toeic)
            lc_loss = self._loss(predicted_lc, _lc)
            rc_loss = self._loss(predicted_rc, _rc)

            if mode == "train":
                # self._update(self.lambda1*(lc_loss+rc_loss)+self.lambda2*bce_loss)
                self._update(lc_loss + rc_loss)

            total_count += len(input_features['qid'])
            toeic_loss_list.append(toeic_loss.item() * len(input_features['qid']))
            lc_loss_list.append(lc_loss.item())
            rc_loss_list.append(rc_loss.item())

        mae = np.sum(toeic_loss_list) / total_count
        mean_lc_loss = np.mean(rc_loss_list)
        mean_rc_loss = np.mean(lc_loss_list)
        loss = mean_lc_loss + mean_rc_loss

        training_time = time.time() - start_time

        print_val = f"time: {training_time:.2f}, MAE: {mae:.4f},   loss: {loss:.4f}"
        if mode == TRAIN_MODE:
            self._train_loss = loss
            self._train_mae = mae
            if not config.ARGS.use_score_val:
                print(f"[Train]     {print_val}")
        elif mode == VALIDATION_MODE:
            self._validation_mae = mae
            if not config.ARGS.use_score_val:
                print(f"└───[Val]  {print_val}")
                print()
        elif mode == TEST_MODE:
            self._test_mae = mae
            print(f"└───[Test]  {print_val}")
        else:
            assert False

    def train(self):
        """
        Main training loop that calls self._inference(*)
        Evals every epoch and test at end
        """
        # print("Initial inference")
        self._model.eval()
        # self._inference(self._train_generator, mode=TEST_MODE)
        # self._inference(self._val_generator, mode=TEST_MODE)
        # self._inference(self._test_generator, mode=TEST_MODE)

        train_generator, val_generator, test_generator = self._score_gen
        epochs = range(self._num_epochs)
        if config.ARGS.use_score_val:
            epochs = tqdm(epochs)
        for epoch in epochs:
            # train
            self._model.train()
            self._inference(train_generator, TRAIN_MODE)

            # test
            with torch.no_grad():
                self._model.eval()
                self._inference(val_generator, VALIDATION_MODE)
                self._set_min_loss(epoch)

            # save_parameters and write_wandb
            cur_weight = self._model.state_dict()
            torch.save(cur_weight, f"{self._weight_path}/ft_{epoch}.pt")
            if not self._debug_mode and config.ARGS.log_score_outputs:
                self._write_wandb()

            if self._patience == 0:
                break

        path = f"{self._weight_path}/ft_{self._validation_min_mae_epoch}.pt"
        self._model.load_state_dict(torch.load(path))
        self._inference(test_generator, TEST_MODE)
        if not self._debug_mode and config.ARGS.log_score_outputs:
            self._write_wandb()

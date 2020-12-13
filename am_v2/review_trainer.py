"""
Trainer class and functions for review correctness prediction fine-tuning
"""
import time

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
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
        review_gen,
        lambda1,
        lambda2,
    ):
        self._device = device
        self._model = model
        self._debug_mode = debug_mode
        self._weight_path = weight_path
        self._num_epochs = num_epochs
        self._review_gen = review_gen

        self._bce_loss = nn.BCELoss(reduction="none")
        self._optimizer = torch.optim.Adam(
            self._model.parameters(), lr=learning_rate, betas=(0.9, 0.98), eps=1e-9
        )
        self._schedule_optimizer = ScheduledOptim(
            self._optimizer, d_model, n_warmup_steps=warmup_steps
        )

        self.lambda1 = lambda1
        self.lambda2 = lambda2

        self._train_auc = -1
        self._train_loss = 0.
        self._validation_auc = -1
        self._validation_max_auc = -1
        self._validation_max_auc_epoch = -1
        self._test_auc = -1

        self._patience = 10

    def _set_min_loss(self, epoch):
        """Set minimum loss"""
        if self._validation_max_auc < self._validation_auc:
            self._validation_max_auc = self._validation_auc
            self._validation_max_auc_epoch = epoch
            self._patience = 10
        else:
            self._patience -= 1

    def _update(self, loss):
        """Updates gradients and lr"""
        self._schedule_optimizer.zero_grad()
        loss.backward()
        self._schedule_optimizer.step_and_update_lr()


    def _inference(self, generator, mode):
        """
        Training code also used for inference when mode is dev or test
        """
        start_time = time.time()
        total_count = 0
        prediction_count = 0
        epoch_outputs = []
        epoch_labels = []


        gen_iter = generator
        for _, (input_features, review_qid, review_correct) in enumerate(gen_iter):
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
            review_qid = torch.LongTensor(review_qid).to(self._device)
            review_correct = torch.LongTensor(review_correct).float().to(self._device)

            output = self._model(input_features, review_qid).view(-1)
            review_correct = review_correct.float().to(self._device)
            
            valid_sample_idx = torch.nonzero(torch.sum((input_features['qid'] != 0), 1)).view(-1)
            output = output[valid_sample_idx]
            review_correct = review_correct[valid_sample_idx]

            batch_loss = torch.mean(
                self._bce_loss(output, review_correct)
            )
            if mode == "train":
                self._update(batch_loss)

            # prediction = (output >= 0.5)
            # prediction_count += (
            #     (prediction == review_correct)
            # ).sum().float()
            # total_count += len(valid_sample_idx)

            epoch_outputs.append(output.detach().cpu().numpy())
            epoch_labels.append(review_correct.detach().cpu().numpy())

        # acc = prediction_count / total_count

        epoch_labels = np.concatenate(epoch_labels)
        epoch_outputs = np.concatenate(epoch_outputs)
        auc = roc_auc_score(epoch_labels, epoch_outputs)

        training_time = time.time() - start_time

        print_val = f"time: {training_time:.2f}, auc: {auc:.4f}"
        if mode == TRAIN_MODE:
            self._train_auc = auc
        elif mode == VALIDATION_MODE:
            self._validation_auc = auc
        elif mode == TEST_MODE:
            self._test_auc = auc
            print(f"└───[Test]  {print_val}")
        else:
            assert False

    def train(self):
        """
        Main training loop that calls self._inference(*)
        Evals every epoch and test at end
        """
        self._model.eval()

        train_generator, val_generator, test_generator = self._review_gen
        epochs = range(self._num_epochs)
        for epoch in tqdm(epochs):
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

            if self._patience == 0:
                break

        path = f"{self._weight_path}/ft_{self._validation_max_auc_epoch}.pt"
        self._model.load_state_dict(torch.load(path))
        self._inference(test_generator, TEST_MODE)


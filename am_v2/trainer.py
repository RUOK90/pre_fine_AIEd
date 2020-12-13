"""
Trainer class for AM pre-training
"""

import time

from tqdm import tqdm

import numpy as np
import torch
import wandb

from am_v2 import config, score_main
from am_v2.optim_schedule import ScheduledOptim


class Trainer:
    """Trainer class pre-trains and tests the model"""

    def __init__(
        self,
        model,
        device,
        d_model,
        debug_mode,
        num_epochs,
        warmup_steps,
        weight_path,
        weighted_loss,
        learning_rate,
        generators,
        score_generators,
    ):
        self.device = device
        self._model = model
        self._debug_mode = debug_mode
        self._weight_path = weight_path
        self._weighted_loss = weighted_loss
        self._num_epochs = num_epochs
        (
            self._train_generator,
            self._dev_generator,
            self._test_generator,
        ) = generators
        self._score_generators = score_generators

        adam = torch.optim.Adam(self._model.parameters(), lr=learning_rate)
        self.optim_schedule = ScheduledOptim(adam, d_model, n_warmup_steps=warmup_steps)
        self._mse_loss = torch.nn.MSELoss(reduction="none")
        self._bce_loss = torch.nn.BCELoss(reduction="none")
        self._ce_loss = torch.nn.CrossEntropyLoss(reduction="none")

        self._loss_ratio = 2
        self._cur_epoch = 0
        self._max_acc = -99
        self._max_epoch = 0

        self._train_acc = -99
        self._train_time_acc = -99
        self._train_correct_acc = -99
        self._train_loss = -99
        self._train_et_mse = 0
        self._train_lt_mse = 0
        self._best_dev_metric = 0
        self._best_dev_time_acc = 0
        self._best_dev_correct_acc = 0
        self._best_dev_loss = 0
        self._best_epoch = 0
        self._val_min_maes = {}
        self._test_maes = {}
        self._best_mae = 100000
        self._dev_acc = 0
        self._dev_time_acc = 0
        self._dev_correct_acc = 0
        self._dev_add_task_acc = 0
        self._dev_loss = 0
        self._dev_et_mse = 0
        self._dev_lt_mse = 0

    def _update(self, loss):
        """Updates gradients and lr"""
        self.optim_schedule.zero_grad()
        loss.backward()
        self.optim_schedule.step_and_update_lr()

    def _write_wandb(self):
        """Writes train logs to WanDB"""
        wandb.log(
            {
                "Train bce_Loss": self._train_loss,
                "Train acc": self._train_acc,
                "Train time acc": self._train_time_acc,
                "Train correct acc": self._train_correct_acc,
                "Train ET mse": self._train_et_mse,
                "Train LT mse": self._train_lt_mse,
                "Dev MAE 0": self._val_min_maes[0],
                "Dev MAE 1": self._val_min_maes[1],
                "Dev MAE 2": self._val_min_maes[2],
                "Dev MAE 3": self._val_min_maes[3],
                "Dev MAE 4": self._val_min_maes[4],
                "Test MAE 0": self._test_maes[0],
                "Test MAE 1": self._test_maes[1],
                "Test MAE 2": self._test_maes[2],
                "Test MAE 3": self._test_maes[3],
                "Test MAE 4": self._test_maes[4],
                "5-mean Dev MAE": sum(self._val_min_maes.values())/len(self._val_min_maes),
                "5-mean Test MAE": sum(self._test_maes.values()) / len(self._test_maes),
                "Dev bce_Loss": self._dev_loss,
                "Dev acc": self._dev_acc,
                "Dev time acc": self._dev_time_acc,
                "Dev correct acc": self._dev_correct_acc,
                "Dev ET mse": self._dev_et_mse,
                "Dev LT mse": self._dev_lt_mse,
                "Best dev time acc": self._best_dev_time_acc,
                "Best dev correct acc": self._best_dev_correct_acc,
                "Best epoch": self._best_epoch,
            }
        )

    def _inference(self, generator, is_training_mode, epoch,):
        """
        Training code also used for inference when is_training_mode==False
        """
        start_time = time.time()
        total_count = 0
        prediction_counts = {'is_on_time': 0, 'is_correct': 0}
        loss_list = []
        et_mse_list = []
        lt_mse_list = []
        for (step, batch) in enumerate(tqdm(generator)):
            batch = {
                feature_group: {
                    name: feature.to(self.device)
                    for name, feature in features.items()
                }
                for feature_group, features in batch.items()
            }
            input_features = {
                'qid': batch['all']['qid'].long(),
                'part': batch['all']['part'].long(),
                # 'is_on_time': batch['masked']['is_on_time'].long(),
                'is_correct': batch['masked']['is_correct'].long(),
                'elapsed_time': batch['masked']['elapsed_time'].unsqueeze(-1),
                'lag_time': batch['masked']['lag_time'].unsqueeze(-1)
            }
            output = self._model(input_features)

            is_predicted = batch['all']['is_predicted']
            target_count = is_predicted.contiguous().view(-1).sum()
            total_count += target_count
            batch_mean_loss = torch.Tensor([0.0]).to(self.device)

            for pretrain_feature_name, model_output in output.items():
                model_output = torch.sigmoid(model_output)
                expected_output = batch['all'][pretrain_feature_name]

                if pretrain_feature_name in ['is_correct', 'is_on_time']:
                    feature_loss = self._bce_loss(model_output, expected_output)
                else:
                    feature_loss = self._mse_loss(model_output, expected_output)
                masked_feature_loss = feature_loss.masked_fill(
                    is_predicted == 0, 0
                )
                batch_mean_loss += masked_feature_loss.sum() / target_count

                if pretrain_feature_name == 'elapsed_time':
                    et_mse_list.append(masked_feature_loss.sum().item())
                elif pretrain_feature_name == 'lag_time':
                    lt_mse_list.append(masked_feature_loss.sum().item())

                if pretrain_feature_name in ['is_correct', 'is_on_time']:
                    prediction = (model_output >= 0.5).float()
                    masked_prediction = prediction.masked_fill(
                        is_predicted == 0, 999
                    )
                    prediction_counts[pretrain_feature_name] += (
                        (masked_prediction.float() == expected_output.float())
                        .sum()
                        .float()
                    )

            loss_list.append(batch_mean_loss.item())
            if is_training_mode:
                loss = batch_mean_loss
                self._update(loss)
                # Break after 0.N % of training data
                if step / len(generator) > config.ARGS.cut_point:
                    break
        acc = {}
        for feature_name in prediction_counts:
            acc[feature_name] = prediction_counts[feature_name]/total_count
        acc['mean'] = sum(acc.values())/len(acc)

        mean_loss = np.mean(loss_list)
        training_time = time.time() - start_time

        et_mse = np.sum(et_mse_list) / total_count
        lt_mse = np.sum(lt_mse_list) / total_count

        print_val = (
            f"\ntime: {training_time:.2f}, ce_loss: {mean_loss:.4f}, r_acc:"
            f" {acc['is_correct']:.4f}, t_acc: {acc['is_on_time']:.4f}"
        )
        if is_training_mode:
            self._train_time_acc = acc['is_on_time']
            self._train_correct_acc = acc['is_correct']
            self._train_acc = acc['mean']
            self._train_loss = mean_loss
            self._train_et_mse = et_mse
            self._train_lt_mse = lt_mse
            print(f"[Train]     {print_val}")
        else:
            self._dev_time_acc = acc['is_on_time']
            self._dev_correct_acc = acc['is_correct']
            self._dev_acc = acc['mean']
            self._dev_loss = mean_loss
            self._dev_et_mse = et_mse
            self._dev_lt_mse = lt_mse
            print(f"└───[Dev]  {print_val}")
            print(
                f"└──────[best] epoch: {self._best_epoch}, accuracy:"
                f" {self._best_dev_metric}"
            )
            self._record_best_epoch(epoch, acc['is_correct'], "acc")

    def _record_best_epoch(self, epoch, metric_val, metric_typ):
        """Records best epoch number and val acc"""
        if metric_typ == "mae":
            if metric_val < self._best_mae:
                self._best_epoch = epoch
                self._best_mae = metric_val
        elif metric_typ == "acc":
            if metric_val > self._best_dev_correct_acc:
                self._best_epoch = epoch
                self._best_dev_correct_acc = metric_val

    def train(self):
        """
        Main training loop that calls self._inference(*)
        Evals every epoch and test at end
        """
        for epoch in range(self._num_epochs):
            print(f"\nEpoch: {epoch:03d} re-shuffling...")
            self._cur_epoch = epoch

            torch.manual_seed(7 + epoch * 11)
            np.random.seed(7 + epoch * 11)

            # train
            self._model.train()
            self._inference(
                self._train_generator,
                is_training_mode=True,
                epoch=epoch,
            )

            # save_parameters
            cur_weight = self._model.state_dict()
            torch.save(cur_weight, f"{self._weight_path}/{epoch}.pt")

            if config.ARGS.use_score_val:
                print("└───[Val] Score data dev set fine-tuning evaluation")
                test_maes, val_min_maes = score_main.finetune_model(
                    self._model.embedding_by_feature['qid'].num_embeddings - 1,
                    f"{self._weight_path}/{epoch}.pt",
                    self._score_generators,
                    _num_epochs=100,
                )
                self._val_min_maes = val_min_maes
                self._test_maes = test_maes
                print(f'5-means dev mae:{sum(self._val_min_maes.values()) / len(self._val_min_maes)}')
                # self._record_best_epoch(epoch, trainer._validation_min_mae, "mae")

            print("Dev set evaluation")
            # validate with val set
            with torch.no_grad():
                self._model.eval()
                self._inference(
                    self._dev_generator,
                    is_training_mode=False,
                    epoch=epoch,
                )

            # write_wandb
            if not self._debug_mode:
                self._write_wandb()

        if config.ARGS.use_score_val:
            # test on score data
            print(
                "└───[Test] Score data test set fine-tuning evaluation with best val"
                " mae model"
            )
            test_mae, _ = score_main.finetune_model(
                self._model.embed_question.num_embeddings - 1,
                f"{self._weight_path}/{self._best_epoch}.pt",
                self._score_generators,
                _num_epochs=1000,
            )
            print(f"└───[Test] Best AM epoch: {self._best_epoch}")
            print(f"└───[Test] Best AM r_acc: {self._best_dev_correct_acc}")
            print(f"└───[Test] Best AM t_acc: {self._best_dev_time_acc}")
            print(f"└───[Test] Best test MAE: {test_mae}")

        # test on test set
        print("Test set evaluation")
        with torch.no_grad():
            self._model.eval()

            self._inference(
                self._test_generator,
                is_training_mode=False,
                epoch=epoch,
            )




import time
import wandb
import torch
import numpy as np
import dill as pkl
from am_v2.optim_schedule import ScheduledOptim
from tqdm import tqdm
from am_v2.config import ARGS

class BERTTrainer:
    """Trainer class for BERT baseline"""

    def __init__(
        self,
        model,
        device,
        d_model,
        debug_mode,
        num_epochs,
        learning_rate,
        warmup_steps,
        generators,
        qid_to_tokens,
    ):
        self.device = device
        self._model = model
        self._debug_mode = debug_mode
        self._num_epochs = num_epochs
        (
            self._train_generator,
            self._dev_generator,
            self._test_generator,
        ) = generators
        self._qid_to_tokens = qid_to_tokens

        adam = torch.optim.Adam(self._model.parameters(), lr=learning_rate)
        self.optim_schedule = ScheduledOptim(adam, d_model, n_warmup_steps=warmup_steps)
        self._ce_loss = torch.nn.CrossEntropyLoss(reduction="none")

        self._cur_epoch = 0
        self._max_epoch = 0

        self._train_acc = -99
        self._train_loss = -99
        self._best_dev_acc = 0
        self._best_epoch = 0
        self._dev_acc = 0
        self._dev_loss = 0

    def _update(self, loss):
        """Updates gradients and lr"""
        self.optim_schedule.zero_grad()
        loss.backward()
        self.optim_schedule.step_and_update_lr()

    def _write_wandb(self):
        """Writes train logs to WanDB"""
        wandb.log(
            {
                "Train Loss": self._train_loss,
                "Train acc": self._train_acc,
                "Dev Loss": self._dev_loss,
                "Dev acc": self._dev_acc,
                "Best dev acc": self._best_dev_acc,
                "Best epoch": self._best_epoch,
            }
        )

    def _inference(self, generator, is_training_mode, epoch,):
        """
        Training code also used for inference when is_training_mode==False
        """
        start_time = time.time()
        total_count = 0
        correct_count = 0
        loss_list = []

        for (step, batch) in enumerate(tqdm(generator)):
            input_feature = {'word_token': batch['masked']['word_token'].long().to(self.device)}

            model_output = self._model(input_feature)
            is_predicted = batch['all']['is_predicted'].to(self.device)

            target_count = is_predicted.contiguous().view(-1).sum()
            total_count += target_count

            expected_output = batch['all']['word_token'].long().to(self.device)
            model_output = model_output.transpose(-1, -2)
            feature_loss = self._ce_loss(model_output, expected_output)
            masked_feature_loss = feature_loss.masked_fill(
                is_predicted == 0, 0
            )
            batch_mean_loss = torch.mean(masked_feature_loss)
            loss_list.append(batch_mean_loss.item())

            prediction = torch.argmax(model_output,1).masked_fill(is_predicted == 0, -1)
            correct_count += (expected_output == prediction).sum()

            if is_training_mode:
                loss = batch_mean_loss
                self._update(loss)
                # Break after 0.N % of training data
                if step / len(generator) > ARGS.cut_point:
                    break
        acc = correct_count / total_count
        mean_loss = np.mean(loss_list)
        training_time = time.time() - start_time

        print_val = (
            f"\ntime: {training_time:.2f}, ce_loss: {mean_loss:.4f}, r_acc:"
            f" {acc:.4f}"
        )
        if is_training_mode:
            self._train_acc = acc
            self._train_loss = mean_loss
            print(f"[Train]     {print_val}")
        else:
            self._dev_acc = acc
            self._dev_loss = mean_loss
            print(f"└───[Dev]  {print_val}")
            print(
                f"└──────[best] epoch: {self._best_epoch}, accuracy:"
                f" {self._best_dev_acc}"
            )
            self._record_best_epoch(epoch, acc, "acc")

    def _record_best_epoch(self, epoch, metric_val, metric_typ):
        """Records best epoch number and val acc"""
        if metric_typ == "acc":
            if metric_val > self._best_dev_acc:
                self._best_epoch = epoch
                self._best_dev_acc = metric_val
        else:
            raise NotImplementedError

    def train(self):
        """
        Main training loop that calls self._inference(*)
        Evals every epoch and test at end
        """
        for epoch in range(self._num_epochs):
            print(f"\nEpoch: {epoch:03d} re-shuffling...")
            self._cur_epoch = epoch

            # train
            self._model.train()
            self._inference(
                self._train_generator,
                is_training_mode=True,
                epoch=epoch,
            )

            # save dictionary which takes a qid and returns the average of word vectors that appear in the question 
            qid_to_vector = {
                qid: self._model.embed_words(tokens)
                for qid, tokens in self._qid_to_tokens.items()
            }
            with open(f'BERT/qid_to_vector_{epoch}.pkl', 'wb') as output_file:
                pkl.dump(qid_to_vector, output_file)


            # torch.save()
            #
            # torch.save(cur_weight, f"{self._weight_path}{epoch}.pt")

            # write wandb
            if not self._debug_mode:
                self._write_wandb()

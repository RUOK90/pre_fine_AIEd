import sys
sys.path.append("..")
import numpy as np
import time
import torch
import torch.nn as nn
from tqdm import tqdm
from optim_schedule import ScheduledOptim
import wandb

TRAIN_MODE = 'train'
VALIDATION_MODE = 'validation'
TEST_MODE = 'test'

class Trainer:
    def __init__(self, model, debug_mode, num_epochs, weight_path, d_model, warmup_steps, learning_rate, device, train_generator, val_generator, test_generator, lambda1, lambda2):
        self._device = device
        self._model = model
        self._debug_mode = debug_mode
        self._weight_path = weight_path
        self._num_epochs = num_epochs
        self._train_generator = train_generator
        self._val_generator = val_generator
        self._test_generator = test_generator
        self._weighted_loss = False

        self._loss = nn.L1Loss()
        self._l2loss = nn.MSELoss()
        self._bce_loss = nn.BCELoss(reduce=False)
        self._optimizer = torch.optim.Adam(self._model.parameters(), lr=learning_rate, betas=(0.9, 0.98), eps=1e-9)
        self._schedule_optimizer = ScheduledOptim(self._optimizer, d_model, n_warmup_steps = warmup_steps)

        self.lambda1 = lambda1
        self.lambda2 = lambda2

        self._toeic_list = []
        self._lc_list = []
        self._rc_list = []

        self._count_list = []
        self._user_id_list = []

        self._predicted_toeic_list = []
        self._predicted_lc_list = []
        self._predicted_rc_list = []

        self._train_mae = -1
        self._train_loss = 0.
        self._validation_mae = -1
        self._validation_min_mae = 99999
        self._validation_min_mae_epoch = -1
        self._test_mae = -1

    def _set_min_loss(self, epoch):
        if self._validation_min_mae > self._validation_mae:
            self._validation_min_mae = self._validation_mae
            self._validation_min_mae_epoch = epoch

    def _update(self, loss):
        self._schedule_optimizer.zero_grad()
        loss.backward()
        self._schedule_optimizer.step_and_update_lr()

    def _write_wandb(self):
        import wandb
        wandb.log({
            'Train MAE': self._train_mae,
            'Train loss': self._train_loss,

            'Val MAE': self._validation_mae,
            'Best Val MAE': self._validation_min_mae,

            'Test MAE': self._test_mae
        })

    def get_weight(self, seq_size):

        a = 0.9 / math.log(seq_size)
        b = 0.1
        x = torch.arange(seq_size) + 1
        return a*x.float().log() + b

    def _inference(self, generator, mode):
        start_time = time.time()
        toeic_loss_list = []
        lc_loss_list = []
        rc_loss_list = []
        sum_lc_rc_loss_list = []

        self._toeic_list = []
        self._lc_list = []
        self._rc_list = []

        self._count_list = []
        self._user_id_list = []

        self._predicted_toeic_list = []
        self._predicted_lc_list = []
        self._predicted_rc_list = []

        for item, answer, is_on_time_list, part, lc, rc, user_id, count in tqdm(generator):

            predicted_lc, predicted_rc = self._model(item, answer, is_on_time_list, part)
            predicted_toeic = (predicted_lc + predicted_rc) * 990

            _toeic = (lc+rc).view(-1, 1).float().to(self._device)
            _lc = lc.view(-1, 1).float().to(self._device) / 990
            _rc = rc.view(-1, 1).float().to(self._device) / 990

            toeic_loss = self._loss(predicted_toeic, _toeic)
            lc_loss = self._l2loss(predicted_lc, _lc)
            rc_loss = self._l2loss(predicted_rc, _rc)

            if mode == 'train':
                # self._update(self.lambda1*(lc_loss+rc_loss)+self.lambda2*bce_loss)
                self._update(lc_loss+rc_loss)
            else:
                self._toeic_list += list((lc+rc).numpy().flatten())
                self._lc_list += list(lc.numpy().flatten())
                self._rc_list += list(rc.numpy().flatten())

                self._count_list += list(count.numpy())
                self._user_id_list += list(user_id)

                self._predicted_toeic_list += list(predicted_toeic.data.cpu().numpy().flatten())
                self._predicted_lc_list += list(predicted_lc.data.cpu().numpy().flatten())
                self._predicted_rc_list += list(predicted_rc.data.cpu().numpy().flatten())

            toeic_loss_list.append(toeic_loss.item())
            lc_loss_list.append(lc_loss.item())
            rc_loss_list.append(rc_loss.item())

        a = 0
        for index in range(len(self._toeic_list)):
            uid = self._user_id_list[index]
            count = self._count_list[index]

            score = self._toeic_list[index]
            lc = self._lc_list[index]
            rc = self._rc_list[index]

            predicted_score = self._predicted_toeic_list[index]
            predicted_lc = self._predicted_lc_list[index]
            predicted_rc = self._predicted_rc_list[index]

            a += abs(score-predicted_score)
            print_val = f'{uid}, {count}, {score}, {lc}, {rc}, {predicted_score}, {predicted_lc}, {predicted_rc}, {abs(score-predicted_score)}'
            print(print_val)



        print(toeic_loss_list)
        mae = np.mean(toeic_loss_list)
        mean_lc_loss = np.mean(rc_loss_list)
        mean_rc_loss = np.mean(lc_loss_list)
        loss = mean_lc_loss + mean_rc_loss

        training_time = time.time() - start_time

        print_val = f'time: {training_time:.2f}, MAE: {mae:.4f},   loss: {loss:.4f}'
        if mode == TRAIN_MODE:
            self._train_loss = loss
            self._train_mae = mae
            print(f'[Train]     {print_val}')
        elif mode == VALIDATION_MODE:
            self._validation_mae = mae
            print(f'└───[Val]  {print_val}')
        elif mode == TEST_MODE:
            self._test_mae = mae
            print(f'└───[Test]  {print_val}')
        else:
            assert(False)

    def train(self):
        for epoch in range(self._num_epochs):
            print(f'\nEpoch: {epoch:03d} re-shuffling...')

            # train
            self._model.train()
            self._inference(self._train_generator, mode=TRAIN_MODE)

            # test
            with torch.no_grad():
                self._model.eval()

                self._inference(self._val_generator, mode=VALIDATION_MODE)
                self._set_min_loss(epoch)

            #save_parameters and write_wandb
            cur_weight = self._model.state_dict()
            torch.save(cur_weight, f'{self._weight_path}{epoch}.pt')
            if not self._debug_mode:
                self._write_wandb()

        path = f'{self._weight_path}{self._validation_min_mae_epoch}.pt'
        self._model.load_state_dict(torch.load(path))
        self._inference(self._test_generator, mode=TEST_MODE)
        if not self._debug_mode:
            self._write_wandb()

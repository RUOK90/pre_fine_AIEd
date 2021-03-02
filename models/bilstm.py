import torch
import torch.nn as nn


class BiLSTM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.bilstm = nn.LSTM(
            input_size=config.embedding_size,
            hidden_size=config.hidden_size,
            num_layers=config.num_hidden_layers,
            batch_first=True,
            dropout=config.hidden_dropout_prob,
            bidirectional=True,
        )
        self.final = nn.Linear(2 * config.hidden_size, 2)

    def forward(self, x):
        x, _ = self.bilstm(x)
        x, _ = x.max(dim=1)
        logit = self.final(x)
        output = torch.sigmoid(logit)

        outputs = {"lc": (logit[:, 0], output[:, 0]), "rc": (logit[:, 1], output[:, 1])}

        return outputs

import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.fc1 = nn.Linear(config.embedding_size, config.hidden_size)
        self.fc2 = nn.Linear(config.hidden_size, config.hidden_size)
        self.fc3 = nn.Linear(config.hidden_size, config.hidden_size)
        self.fc4 = nn.Linear(config.hidden_size, config.hidden_size)
        self.final = nn.Linear(config.hidden_size, 2)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, x):
        x = x.sum(-2)
        x = self.dropout(self.act(self.fc1(x)))
        x = self.dropout(self.act(self.fc2(x)))
        x = self.dropout(self.act(self.fc3(x)))
        x = self.dropout(self.act(self.fc4(x)))
        logit = self.final(x)
        output = torch.sigmoid(logit)

        outputs = {"lc": (logit[:, 0], output[:, 0]), "rc": (logit[:, 1], output[:, 1])}

        return outputs

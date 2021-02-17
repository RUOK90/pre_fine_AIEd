from torch.utils import data
from itertools import chain, islice
import torch
import random
import numpy as np


class DummyDataSet(data.Dataset):
    def __init__(self):
        self._data = list(range(10))

    def __len__(self):
        return len(self._data)

    def __getitem__(self, idx):
        return self._data[idx]


dummy_dataloader = data.DataLoader(
    dataset=DummyDataSet(),
    batch_size=3,
    shuffle=True,
    num_workers=0,
)

chained_dataloader = chain.from_iterable([dummy_dataloader] * 10)

torch.manual_seed(1)
torch.cuda.manual_seed_all(1)
random.seed(1)
np.random.seed(1)

sliced_dataloader = islice(chained_dataloader, 5)
for batch in sliced_dataloader:
    print(batch)

print()

torch.manual_seed(2)
torch.cuda.manual_seed_all(2)
random.seed(2)
np.random.seed(2)

sliced_dataloader = islice(chained_dataloader, 5)
for batch in sliced_dataloader:
    print(batch)

print()

dummy_dataloader = data.DataLoader(
    dataset=DummyDataSet(),
    batch_size=3,
    shuffle=True,
    num_workers=0,
)

chained_dataloader = chain.from_iterable([dummy_dataloader] * 10)

torch.manual_seed(2)
torch.cuda.manual_seed_all(2)
random.seed(2)
np.random.seed(2)

sliced_dataloader = islice(chained_dataloader, 5, 10)
for batch in sliced_dataloader:
    print(batch)

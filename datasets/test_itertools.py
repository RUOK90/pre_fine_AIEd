from torch.utils import data
from itertools import chain, islice


class DummyDataSet(data.Dataset):
    def __init__(self):
        self._data = list(range(90))

    def __len__(self):
        return len(self._data)

    def __getitem__(self, idx):
        return self._data[idx]


dummy_dataset = DummyDataSet()
dummy_dataloader = data.DataLoader(
    dataset=DummyDataSet(),
    batch_size=3,
    shuffle=False,
    num_workers=0,
)

chained_dataloader = chain.from_iterable([dummy_dataloader] * 3)

for data in islice(chained_dataloader, 5):
    pass
print()

for data in islice(chained_dataloader, 5):
    pass
print()

sliced_dataloader = islice(chained_dataloader, 5)
for data in sliced_dataloader:
    print(data)

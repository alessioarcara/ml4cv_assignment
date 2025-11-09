import torch


class MultiEpochsDataLoader(torch.utils.data.DataLoader):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._DataLoader__initialized = False
        self.batch_sampler = _RepeatSampler(self.batch_sampler)
        self._DataLoader__initialized = True
        self.iterator = super().__iter__()

    def __len__(self):
        return len(self.batch_sampler.sampler)

    def __iter__(self):
        for i in range(len(self)):
            yield next(self.iterator)


class _RepeatSampler(object):
    """
    Sampler that repeats forever.

    Args:
        sampler (Sampler)
    """

    def __init__(self, sampler):
        self.sampler = sampler

    def __iter__(self):
        while True:
            yield from iter(self.sampler)


# Verifying whether the _RepeatSampler affects the shuffle functionality.
# If shuffle is working correctly, the printed indices should be different across epochs.
if __name__ == "__main__":

    class IndexDataset(torch.utils.data.Dataset):
        def __init__(self, size):
            self.size = size

        def __len__(self):
            return self.size

        def __getitem__(self, idx):
            return idx

    dataset = IndexDataset(100)

    loader = MultiEpochsDataLoader(dataset, batch_size=4, shuffle=True, num_workers=0)

    for epoch in range(3):
        indices = []
        for batch in loader:
            indices.extend(batch.numpy())
            break

        print(f"Epoch {epoch + 1}: {indices}")

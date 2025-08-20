import torch
from torch.utils.data import DataLoader
from torch.utils.data.sampler import Sampler

class PosNegBatchSampler(Sampler):
    """
    Yields batches of indices: half positives, half negatives (no replacement).
    Epoch length is limited by the number of positives.
    """
    def __init__(self, labels: torch.Tensor, pos_value: int, batch_size: int, generator=None):
        assert batch_size % 2 == 0, "batch_size must be even"
        self.labels = labels if torch.is_tensor(labels) else torch.as_tensor(labels)
        self.pos_value = pos_value
        self.batch_size = batch_size
        self.per_class = batch_size // 2
        self.generator = generator

        self.pos_idx = (self.labels == pos_value).nonzero(as_tuple=True)[0].tolist()
        self.neg_idx = (self.labels != pos_value).nonzero(as_tuple=True)[0].tolist()
        if len(self.pos_idx) == 0:
            raise ValueError("No positive samples for the chosen digit.")

    def __iter__(self):
        # shuffle indices each epoch
        g = self.generator if self.generator is not None else torch.Generator()
        pos = torch.tensor(self.pos_idx)[torch.randperm(len(self.pos_idx), generator=g)].tolist()
        neg = torch.tensor(self.neg_idx)[torch.randperm(len(self.neg_idx), generator=g)].tolist()

        # number of full batches we can form without reusing samples
        n_batches = min(len(pos) // self.per_class, len(neg) // self.per_class)
        for b in range(n_batches):
            p = pos[b*self.per_class : (b+1)*self.per_class]
            n = neg[b*self.per_class : (b+1)*self.per_class]
            yield p + n

    def __len__(self):
        # how many batches per epoch (bounded by whichever class runs out first;
        # with MNIST, negatives >> positives, so positives will usually bound)
        return min(len(self.pos_idx) // self.per_class, len(self.neg_idx) // self.per_class)

def make_contrastive_loader(dataset, digit: int, batch_size: int = 128, num_workers: int = 0, seed: int | None = None):
    # get labels (MNIST exposes .targets)
    if isinstance(dataset, torch.utils.data.Subset):
        dataset = dataset.dataset

    labels = dataset.targets if hasattr(dataset, "targets") else dataset.labels
    labels = labels if torch.is_tensor(labels) else torch.as_tensor(labels)

    g = torch.Generator().manual_seed(seed) if seed is not None else None
    batch_sampler = PosNegBatchSampler(labels, pos_value=digit, batch_size=batch_size, generator=g)

    # single loader over the full dataset; batch_sampler controls the composition
    loader = DataLoader(
        dataset,
        batch_sampler=batch_sampler,   # NOTE: don't pass batch_size or shuffle with batch_sampler
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )
    return loader

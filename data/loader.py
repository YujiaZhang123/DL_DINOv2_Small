import torch
from torch.utils.data import DataLoader, DistributedSampler


def create_dataloader(
    dataset,
    batch_size: int,
    num_workers: int = None,
    collate_fn=None,
    shuffle: bool = True,
    drop_last: bool = True,
):
    """
    DataLoader that automatically uses DistributedSampler when running under DDP.
    """

    if torch.distributed.is_available() and torch.distributed.is_initialized():
        sampler = DistributedSampler(dataset, shuffle=shuffle)
        shuffle_flag = False   
    else:
        sampler = None
        shuffle_flag = shuffle

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=drop_last,
        collate_fn=collate_fn,
    )

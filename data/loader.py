import torch
from torch.utils.data import DataLoader


def create_dataloader(
    dataset,
    batch_size: int,
    num_workers: int = None,
    collate_fn=None,
    shuffle: bool = True,
    drop_last: bool = True,
):
    """
    Basic DataLoader wrapper for SSL training.
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=drop_last,
        collate_fn=collate_fn,
    )

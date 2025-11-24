import torch


def collate_multicrop(batch):
    """
    Args:
        batch: list of dicts, each containing:
            {
              "global_crops": [g1, g2],
              "local_crops":  [l1, l2, ..., lN],
              ...
            }

    Returns:
        dict with:
            "global_crops": list of Tensors, each of shape (B, C, H, W)
            "local_crops":  list of Tensors, each of shape (B, C, H, W)
    """
    num_global = len(batch[0]["global_crops"])
    num_local = len(batch[0]["local_crops"])

    # ---- collate global crops ----
    global_crops = []
    for i in range(num_global):
        global_crops.append(
            torch.stack([sample["global_crops"][i] for sample in batch], dim=0)
        )

    # ---- collate local crops ----
    local_crops = []
    for j in range(num_local):
        local_crops.append(
            torch.stack([sample["local_crops"][j] for sample in batch], dim=0)
        )

    return {
        "global_crops": global_crops,   # list[Tensor(B,C,H,W), ...]
        "local_crops": local_crops,     # list[Tensor(B,C,H,W), ...]
    }

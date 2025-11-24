from typing import Sequence
from torchvision import transforms as T


class GaussianBlur(T.RandomApply):
    """
    Simple wrapper around torchvision's GaussianBlur.

    Args:
        p: probability of applying the blur to an image.
        sigma_range: (min_sigma, max_sigma) passed to GaussianBlur.
        kernel_size: odd kernel size for the blur filter.
    """

    def __init__(
        self,
        p: float = 0.5,
        sigma_range: tuple[float, float] = (0.1, 2.0),
        kernel_size: int = 9,
    ) -> None:
        blur_op = T.GaussianBlur(kernel_size=kernel_size, sigma=sigma_range)
        # RandomApply: with probability `p` apply `blur_op`, otherwise return input
        super().__init__([blur_op], p=p)


# Standard ImageNet statistics commonly used for ViT-style backbones
IMAGENET_DEFAULT_MEAN: tuple[float, float, float] = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD: tuple[float, float, float] = (0.229, 0.224, 0.225)


def make_normalize_transform(
    mean: Sequence[float] = IMAGENET_DEFAULT_MEAN,
    std: Sequence[float] = IMAGENET_DEFAULT_STD,
) -> T.Normalize:
    """
    Create a normalization transform given channel-wise mean and std.

    This should be used after converting images to tensors in [0, 1].
    """
    return T.Normalize(mean=list(mean), std=list(std))

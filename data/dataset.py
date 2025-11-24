from torch.utils.data import Dataset
from PIL import Image

class SSLDataset(Dataset):
    """
    dataset that loads images from a list of file paths.
    """

    def __init__(self, image_paths, augment):
        self.paths = image_paths
        self.augment = augment

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        return self.augment(img)

    def __len__(self):
        return len(self.paths)

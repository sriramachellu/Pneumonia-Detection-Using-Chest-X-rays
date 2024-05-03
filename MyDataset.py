from torch.utils.data import Dataset
import os
from torchvision.io import read_image

# Dataset class
class ChestXRayDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.images = []
        self.labels = []
        for label in ["NORMAL", "PNEUMONIA"]:
            dir_path = os.path.join(self.root_dir, label)
            self.images.extend([os.path.join(dir_path, file) for file in os.listdir(dir_path)])
            self.labels.extend([0 if label == "NORMAL" else 1 for _ in os.listdir(dir_path)])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        image = read_image(img_path)
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label
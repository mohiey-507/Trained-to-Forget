import os
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
from typing import Tuple

class HumanActivityDataset(Dataset):
    def __init__(self, csv_file: str, root_dir: str, transform=None):
        self.annotations_df = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        self.classes = self.annotations_df['label'].unique()
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}

    def __len__(self):
        return len(self.annotations_df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_name = os.path.join(self.root_dir, self.annotations_df.iloc[idx, 0])
        image = Image.open(img_name).convert("RGB")
        label_name = self.annotations_df.iloc[idx, 1]
        label_idx = self.class_to_idx[label_name]
        if self.transform:
            image = self.transform(image)
        return image, label_idx

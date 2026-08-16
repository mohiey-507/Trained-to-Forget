import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

class HumanActivityDataset(Dataset):
    def __init__(self, csv_path, img_dir, transform=None):
        self.df = pd.read_csv(csv_path)
        self.img_dir = img_dir
        self.transform = transform
        
        self.filename_col = self.df.columns[0] 
        self.label_col = self.df.columns[1]
        
        self.classes = sorted(self.df[self.label_col].unique())
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_name = row[self.filename_col]
        label_name = row[self.label_col]
        
        img_path = os.path.join(self.img_dir, img_name)
        if not os.path.exists(img_path):
            img_path = img_path + '.jpg'
            
        image = Image.open(img_path).convert("RGB")
        label = self.class_to_idx[label_name]
        
        if self.transform:
            image = self.transform(image)
            
        return image, label

import os
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import pandas as pd
from PIL import Image

class CustomData(Dataset):
    def __init__(self, root_dir:str=None, train_csv:str="sample.csv", target_col_name:str="y", transform=None):
        self.root_dir = root_dir
        self.train_csv = train_csv
        self.annotations = pd.read_csv(os.path.join(self.root_dir, self.train_csv) if self.root_dir is not None else train_csv)
        self.target = target_col_name
        self.transform = transform
        self.class_ = sorted(list(self.annotations[self.target].unique()))
        self.class2idx = {cls: i for i, cls in enumerate(self.class_)}
        self.idx2class = {i: cls for cls, i in self.class2idx.items()}

    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, index):
        img_path = os.path.join(self.root, self.annotations.loc[index, "image_path"])

        img = Image.open(img_path).convert("RGB")
        label = self.annotations.loc[index, self.target]
        label = torch.tensor(self.class2idx[label], dtype=torch.int8)

        if self.transform:
            img = self.transform(img)
        else:
            img = transforms.ToTensor()(img)

    def get_classes(self) -> tuple[dict, dict]:
        """
        Docstring for get_classes
        
        :return: Two dictionaries mapping class names. First one is class names to indices, second one is indices to class
        :rtype: tuple[dict, dict]
        """
        return self.class2idx, self.idx2class

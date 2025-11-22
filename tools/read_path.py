from PIL import Image
from torch.utils.data import Dataset
import os
import random
import numpy as np
import cv2



### --- 使用了DFGC-2022的数据增强
class MyDataset_or(Dataset):
    def __init__(self, txt_path, transform=None, target_transform=None):
        fh = open(txt_path, 'r')
        imgs = []
        for line in fh:
            line = line.rstrip()
            words = line.split()
            imgs.append((words[0], int(words[1])))

        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        image_path, label = self.imgs[index]
        image = cv2.cvtColor(cv2.imread(image_path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB) # 用A.compose构建

        img1 = []
        if self.transform is not None:
            img1 = self.transform(image=image)["image"]   # 因为transform里面有A.Compose，所以是这样写。

        return img1, label

    def __len__(self):
        return len(self.imgs)
    





import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image
import config


df_data = pd.read_csv(config.train_path)
leaves_labels = sorted(list(set(df_data['label'])))
class_to_num = dict(zip(leaves_labels, range(len(leaves_labels))))
num_to_class = {v : k for k, v in class_to_num.items()}

class LeavesDataset(Dataset):
    def __init__(self, df_data, img_path, mode='train', transform=None):
        self.transform = transform
        self.img_path = img_path
        self.mode = mode
        self.df_data = df_data
        self.image_arr = np.asarray(self.df_data['image'])
        self.length = len(self.image_arr)       
        
        if mode == 'train':           
            self.label_arr = np.asarray(self.df_data['label'])

    def __getitem__(self, index):
        single_image_name = self.image_arr[index]
        img_as_img = Image.open(self.img_path + single_image_name)
        if not self.transform:
            transform = transforms.ToTensor()
        else:
            transform = self.transform     
        img_as_img = transform(img_as_img)     
        if self.mode == 'test':
            return img_as_img
        else:
            label = self.label_arr[index]
            number_label = class_to_num[label]
            return img_as_img, number_label

    def __len__(self):
        return self.length

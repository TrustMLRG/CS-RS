import os
from glob import glob

import numpy as np
import pandas as pd
import torch
import pdb
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision import transforms
from collections import Counter
data_path='/home/c01yuxi/CISPA-projects/certification_robustness-2022/macer'

class DermDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        X = Image.open(self.df['path'].iloc[index])
        y = torch.tensor(int(self.df['y'].iloc[index]))
        if self.transform:
            X = self.transform(X)
        return X, y
 
        
def load_ham_data(transform):
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    transform = transforms.Compose(
            [
                transforms.Resize(299), #299
                transforms.CenterCrop(299), #299
                transforms.ToTensor(),
                normalize
            ]
    )
    bm_indicator = {"benign": 0, "malignant": 1}
    class_to_idx = {
        'nv': 0,
        'mel': 1,
        'bkl': 2,
        'bcc': 3,
        'akiec': 4,
        'vasc': 5,
        'df': 6
    }
    id_to_lesion = {
        'nv': 'Melanocytic nevi', # 6705
        'mel': 'dermatofibroma', #1113
        'bkl': 'Benign keratosis-like lesions ', #1099
        'bcc': 'Basal cell carcinoma',# 514
        'akiec': 'Actinic keratoses', # 327
        'vasc': 'Vascular lesions', # 142
        'df': 'Dermatofibroma'} # 115

    benign_malignant = {
        'nv': 'benign', # 0:6705
        'mel': 'malignant', # 1:1113
        'bkl': 'benign',# 2:1099
        'bcc': 'malignant',# 3:514
        'akiec': 'benign',# 4 :327
        'vasc': 'benign',# 5:142
        'df': 'benign'} # 6:115

    df = pd.read_csv(data_path+'/HAM10000/HAM10000_metadata')
    all_image_paths = glob(os.path.join(data_path+'/HAM10000/data/', '*.jpg'))
    id_to_path = {os.path.splitext(os.path.basename(x))[0]: x for x in all_image_paths}

    def path_getter(id):
        if id in id_to_path:
            return id_to_path[id]
        else:
            return "-1"

    df['path'] = df['image_id'].map(path_getter)
    df = df[df.path != "-1"]
    df['dx_name'] = df['dx'].map(lambda id: id_to_lesion[id])
    df['benign_or_malignant'] = df["dx"].map(lambda id: benign_malignant[id])

    df['y']=(df['benign_or_malignant']=='benign')
    df['y']=df['y'].astype(int)

    df_train,df_test=train_test_split(df,test_size=0.20,random_state=1,stratify=df['y'])
    trainset = DermDataset(df_train, transform)
    valset = DermDataset(df_test,transform)

    return trainset, valset

    


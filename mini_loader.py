import torch
from PIL import Image
import os
import pandas as pd
import math
import numpy as np

def pil_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


class novel_ImageLoader(torch.utils.data.Dataset):
    def __init__(self, root, give_idx, loader=pil_loader, transform=None, unlabeled=False,extend=False,aug=False):
        img_folder = os.path.join(root, 'miniImagenet_base_novel/novel')
        data = pd.read_csv(os.path.join(root, 'novel_data.txt'))
        novel_data = data[data['idx'].isin(give_idx)]
        fixed_class = novel_data['label'].unique()
        for kk in range(len(fixed_class)):
            novel_data.loc[novel_data['label'] == fixed_class[kk], 'label'] = kk
        data = novel_data
        imgs = data.reset_index(drop=True)
        if unlabeled==False and extend==True:
            hehe=imgs
            for kkk in range(19):
                hehe=pd.concat([hehe,imgs])
            imgs = hehe.reset_index(drop=True)
        if aug:
            tmp_data=pd.DataFrame()
            for i in range(aug):
                tmp_data=pd.concat([tmp_data,data])
            data=tmp_data
            imgs = data.reset_index(drop=True)

        if len(imgs) == 0:
            raise (RuntimeError("no csv file"))
        self.root = img_folder
        self.imgs = imgs
        self.transform = transform
        self.loader = loader
        self.unlabeled = unlabeled

    def __getitem__(self, index):
        item = self.imgs.iloc[int(index)]
        file_path = item['path']
        target = item['label']

        img = self.loader(os.path.join(self.root, file_path))
        if not self.unlabeled:
            if self.transform is not None:
                img = self.transform(img)

            return img, target
        else:  # unlabeled data part
            if self.transform is not None:
                img1 = self.transform(img)
                img2 = self.transform(img)

            target = -1
            return (img1, img2), target

    def __len__(self):
        return len(self.imgs)



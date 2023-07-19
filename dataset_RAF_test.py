
from mmap import mmap
import PIL
import torch
import torch.utils.data as data
import torchvision
import torchvision.transforms as transforms
import math
import scipy.io as io
import numpy as np
from PIL import Image
import cv2
import json
import os
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class RAF_test(data.Dataset):

    def __init__(self, opts):
        super(RAF_test, self).__init__()
     
        self.data = np.load(opts.test_data , mmap_mode='r')
        self.label = np.load(opts.test_label,mmap_mode='r')
        self.targets = np.full((7,len(self.label)), fill_value=0.01)
        self.point = np.load(opts.test_point, mmap_mode='r')

        for i in range(self.label.size) :
            j = self.label[i]
            self.targets[j][i]=0.94
        normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        self.transforms = transforms.Compose([
            transforms.ToTensor(),
            normalize
        ]) 

    def __getitem__(self, index):


        img = self.data[index]
        label = self.label[index].astype(np.float32)
        targets = self.targets[:,index].astype(np.float32)
        point = self.point[index,:,:].astype(np.float32)
        
        img = PIL.Image.fromarray(img)
        img = self.transforms(img)

        return img, targets , label,point


    def __len__(self):

        return len(self.label)

def rafdb_data_loader(opts):
    dataset = RAF_test(opts)
   
    if device.type == 'cuda':
        train_loader = torch.utils.data.DataLoader(
            dataset, batch_size=opts.batch_size_test, shuffle=True, num_workers=16, pin_memory=True
        )
    else:
        train_loader = torch.utils.data.DataLoader(
            dataset, batch_size=opts.batch_size_test, shuffle=True
        )

    return train_loader



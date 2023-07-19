
import PIL
import torch
import torch.utils.data as data
import torchvision
import torchvision.transforms as transforms

import scipy.io as io
import numpy as np
from PIL import Image
import cv2
import json
import os
import matplotlib.pyplot as plt
import math
from sklearn.model_selection import KFold

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class RAF(data.Dataset):

    def __init__(self, opts,transform=None):
        super(RAF, self).__init__()
     
        self.data = np.load(opts.train_data , mmap_mode='r')

        self.label = np.load(opts.train_label, mmap_mode='r')

        self.targets = np.full((7,len(self.label)), fill_value=0.01)
        for i in range(self.label.size) :
            j = self.label[i]
            self.targets[j][i]=0.94


        self.targets = self.targets.astype(np.float32)
        self.point = np.load(opts.train_point, mmap_mode='r')

        normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

        color_jitter = transforms.ColorJitter(0.2, 0.2, 0.2, 0.05)
        self.transforms1 = transforms.Compose([
            transforms.RandomApply([color_jitter], p=0.8),
            transforms.Grayscale(num_output_channels=3),
            transforms.RandomRotation((8, 8), Image.BILINEAR)
        ])  
        self.transforms2 = transforms.Compose([
            transforms.RandomResizedCrop(size=opts.crop_size, scale=(0.75, 1.0), ratio=(1., 1.))
        ])  
        self.transforms3 = transforms.Compose([
            transforms.RandomHorizontalFlip()
        ])   
        self.transforms4 = transforms.Compose([
            transforms.ToTensor(),
            normalize
        ]) 
    

    def __getitem__(self, index):


        img = self.data[index]
        label = self.label[index]

        targets = self.targets[:,index].astype(np.float32)
        point = self.point[index,:,:].astype(np.float32)
        img_size = img.shape[1]

        img = PIL.Image.fromarray(img)
        img = self.transforms1(img)
        if (img[1] == 8.0):
            center_x = img_size/2
            center_y = img_size/2
            ang = math.radians(360-img[1])
            new_x = (point[:,0]-center_x)*math.cos(ang) - (point[:,1]-center_y)*math.sin(ang) + center_x
            new_y = (point[:,0]-center_x)*math.sin(ang) + (point[:,1]-center_y)*math.cos(ang) + center_y
            point[:,0] = new_x
            point[:,1] = new_y
            
        img = self.transforms2(img[0])
        crop_size_x = img[1][2] - img[1][0]
        crop_size_y = img[1][2] - img[1][0]

        point[:,0] = point[:,0] - img[1][0]
        point[:,1] = point[:,1] - img[1][1]
        point[:,0] = (img_size/crop_size_x) * point[:,0]
        point[:,1] = (img_size/crop_size_y) * point[:,1]

        img = self.transforms3(img[0])
        if (img[1] == 'FLIP_LEFT_RIGHT'):
            point[:,0]=img_size - point[:,0]
            point[:,1]=point[:,1]

        img = self.transforms4(img[0])
            
        return img, targets, label,point


    def __len__(self):

        return len(self.label)


def get_train_loader(opts):
    dataset = RAF(opts)
    if device.type == 'cuda':
        train_loader = torch.utils.data.DataLoader(
            dataset, batch_size=opts.batch_size_train, shuffle=True, num_workers=16, pin_memory=True
        )
    else:
        train_loader = torch.utils.data.DataLoader(
            dataset, batch_size=opts.batch_size_train, shuffle=True
        )

    return train_loader
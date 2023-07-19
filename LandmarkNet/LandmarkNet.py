import torch
import time

import torch.nn as nn
import numpy as np

from .Transformer import Transformer
from .interpolation import interpolation_layer
from .get_roi import get_roi
from fpn import LResNet50E_IR


import os
import cv2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class landmark_network(nn.Module):
    def __init__(self, num_point, d_model, trainable,
                 return_interm_layers, dilation, nhead,  feedforward_dim, cfg,args):
        super(landmark_network, self).__init__()
        self.num_point = num_point
        self.d_model = 256

        self.trainable = trainable
        self.return_interm_layers = return_interm_layers
        self.dilation = dilation
        self.nhead = nhead
        self.feedforward_dim = feedforward_dim
        self.Sample_num = int(args.patch_half_length*2)
        self.patch_half_length = int(args.patch_half_length)

        # ROI_creator
        ##  Just calculate the ratio, don't pay too much attention
        self.ROI_1 = get_roi(self.Sample_num, args.patch_half_length, 28)

        self.interpolation = interpolation_layer()

        # feature_extractor
        self.feature_extracto_10 = nn.Conv2d(d_model, d_model, kernel_size=self.Sample_num, bias=False)


        self.feature_norm = nn.LayerNorm(d_model)

        # Transformer
        self.Transformer = Transformer(num_point, d_model, nhead, 6,
                                       feedforward_dim, dropout=0.1)        
        self._reset_parameters()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
                
        self.backbone = LResNet50E_IR(args)
        self.output = nn.Linear(256, 7)
        self.output.weight = nn.init.xavier_uniform(self.output.weight)
        self.output.bias = nn.init.constant(self.output.bias, 0)
        self.deconv = nn.ConvTranspose2d(256, 256, self.Sample_num).cuda()
        
    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward_head(self, src ):

        src = src.mean(dim = 1)

        return src

    def forward(self, image,point,opts,mode = None):
        bs = image.size(0)
        feature_map = self.backbone(image).cuda()


        initial_points = torch.from_numpy(point / 112.0).view(bs, 68, 2).float()
        initial_points = initial_points.to(device)

        ROI_anchor = self.ROI_1(initial_points.detach())
        ROI_anchor = ROI_anchor.view(bs, self.num_point * self.Sample_num * self.Sample_num, 2)

        ROI_feature = self.interpolation(feature_map, ROI_anchor).view(bs, self.num_point, self.Sample_num,
                                                                            self.Sample_num, self.d_model)
        ROI_feature = ROI_feature.view(bs , self.num_point, self.d_model,self.Sample_num, self.Sample_num )
        

        if(mode == 'train'):
            patches,mask_sequences,patch_loss= Get_ids(ROI_feature,opts)
            mask_patches = patches.view(bs * self.num_point, self.d_model,self.Sample_num, self.Sample_num )
            transformer_feature = self.feature_extracto_10(mask_patches).view(bs, self.num_point, self.d_model)
            offset = self.Transformer(transformer_feature)
            offset_nocls = offset[:,-1,1:,:]
            batch_range = torch.arange(bs, device = device)[:, None]
            pred_masked_patch = offset_nocls[batch_range,mask_sequences]
            cls_token = offset[:,-1,:1,:].view(bs,self.d_model)
            classification = self.output(cls_token).cuda()
            # # #(bs,68,256,1,1)->(bs,68,256,8,8)
            pred_masked_patch1 =  self.deconv((pred_masked_patch).contiguous().view(bs * mask_sequences.shape[1], self.d_model,1,1)
                                    ).view(bs , mask_sequences.shape[1], self.d_model,self.Sample_num, self.Sample_num)

            return classification,patch_loss,pred_masked_patch1
        
        elif(mode == 'test'):
            ROI_feature = ROI_feature.view(bs * self.num_point, self.d_model,self.Sample_num, self.Sample_num )
            transformer_feature = self.feature_extracto_10(ROI_feature).view(bs, self.num_point, self.d_model)
            offset = self.Transformer(transformer_feature)
            cls_token = offset[:,-1,:1,:].view(bs,self.d_model)
            classification = self.output(cls_token).cuda()
            
            return classification
       

def Get_ids(transformer_feature,opts):
    Bs = transformer_feature.shape[0]
    batch_range = torch.arange(Bs, device=device)[:, None].cuda()
    len_mask = int(opts.num_patch * opts.mask_ratio)

    symmetric_points = [0, 1, 2, 3, 4, 5, 6, 7, 17, 18, 19, 20, 21, 31, 32, 36, 37, 38, 39, 40, 41, 48, 49, 50, 55, 56, 60, 61, 65]
    symmetric_points = torch.Tensor(symmetric_points).long().cuda()

    unsymmetric_points = [8, 27, 28, 29, 30, 33, 51, 57, 62, 66]
    unsymmetric_points = torch.Tensor(unsymmetric_points).long().cuda()

    symmetric_idx = [
        (0, 16), (1, 15), (2, 14), (3, 13), (4, 12), (5, 11), (6, 10), (7, 9),
        (17, 26), (18, 25), (19, 24), (20, 23), (21, 22), (31, 35), (32, 34),
        (36, 45), (37, 44), (38, 43), (39, 42), (40, 47), (41, 46), (48, 54),
        (49, 53), (50, 52), (55, 59), (56, 58), (60, 64), (61, 63), (65, 67)
    ]
    torch.manual_seed(int(time.time()))
    symmetric_idx = torch.Tensor(symmetric_idx).long().cuda()
    mask_sequences = []

    for q in range(Bs): 
        if len_mask % 2 == 0:
            # Randomly select symmetric points
            random_index = torch.randperm(29, device=device)[:int(len_mask/2)].cuda()
            mask_idx = torch.cat([symmetric_idx[random_index, 0], symmetric_idx[random_index, 1]], dim=-1)
            mask_idx = torch.sort(mask_idx)[0]
        elif len_mask % 2 == 1 and len_mask != 1:
            # Randomly select symmetric points
            random_index = torch.randperm(29, device=device)[:int(len_mask/2)].cuda()
            mask_idx = torch.cat([symmetric_idx[random_index, 0], symmetric_idx[random_index, 1]], dim=-1)
            # Randomly select an unsymmetric point
            random_index = torch.randint(0, 10, (1,), device=device)
            mask_idx = torch.cat([mask_idx, unsymmetric_points[random_index]], dim=-1)
            mask_idx = torch.sort(mask_idx)[0]
        elif len_mask == 1:
            # Randomly select an unsymmetric point
            random_index = torch.randint(0, 10, (1,), device=device).cuda()
            mask_idx = unsymmetric_points[random_index]
        mask_sequences.append(mask_idx.unsqueeze(0))
    mask_sequences = torch.stack(mask_sequences, dim=0).squeeze(1)

    patches = transformer_feature.clone()
    patch_loss = patches[batch_range,mask_sequences]
    patches[batch_range,mask_sequences] = 1
    return patches,mask_sequences,patch_loss

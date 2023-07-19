from pathlib import Path
from sklearn import utils
import torch
from torch.nn import DataParallel
import torch.nn.functional as F
from Config import cfg
from Config import update_config
import torchvision
from torch.autograd import Variable

import cv2
import os
import csv
import argparse
from datetime import datetime
import time
import json
import torch.nn.functional as F
import numpy as np
import pickle
from fpn import LResNet50E_IR
from focalloss import FocalLoss
from params import ParamsControl
from LandmarkNet import landmark_network
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sns
import itertools
from sklearn.model_selection import KFold

from dataset_RAF_train import get_train_loader
from dataset_RAF_test import rafdb_data_loader
from thop import profile

import utils


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
best_fer2013_val_top1 = 0
best_fer2013_test_top1 = 0
best_rafdb_test_top1 = 0
best_rafdb_val_top1 = 0
iter_num = 0


def main(opts):
    global best_fer2013_val_top1
    global best_fer2013_test_top1
    global best_rafdb_test_top1
    global best_rafdb_val_top1


    if not os.path.exists(opts.save_dir):
        os.makedirs(opts.save_dir)

    logging = utils.init_log(os.path.join(opts.save_dir, '{}.log'.format(opts.snapshot)))
    _print = logging.info
    _print(opts)

    opts.train_num = '{:0>4}'.format(opts.train_num)
    opts.save_dir = os.path.join(opts.save_dir, opts.train_num)
    if not os.path.exists(opts.save_dir):
        os.makedirs(opts.save_dir)
    _print(opts.train_num)

    with open(os.path.join(opts.save_dir, 'opts_setting.txt'), 'w') as f:
        json.dump(opts.__dict__, f, indent=2)

    if not os.path.exists(opts.save_dir):
        os.makedirs(opts.save_dir)

    update_config(cfg, opts)

    net = landmark_network(cfg.WFLW.NUM_POINT, cfg.MODEL.OUT_DIM,
                                    cfg.MODEL.TRAINABLE, cfg.MODEL.INTER_LAYER,
                                    cfg.MODEL.DILATION, cfg.TRANSFORMER.NHEAD,
                                    cfg.TRANSFORMER.FEED_DIM, cfg,opts)
    
    params = ParamsControl(opts, net)
    net = net.to(device)
    train_loader = get_train_loader(opts)
    test_loader_rafdb = rafdb_data_loader(opts)
    test_loader = {
        'rafdb_test': test_loader_rafdb,
    }

    _print('train batch_num: {},  raf test batch_num: {}'.format(len(train_loader), len(test_loader_rafdb)))
    
    _print('train batch_size: {},  test batch_size: {}'.format(opts.batch_size_train, opts.batch_size_test))

    loss_func = FocalLoss(opts.loss_type, opts.num_classes, opts.smooth_factor).to(device)
    
    
    for epoch in range(opts.start_epoch, opts.last_epoch):
        params.update(epoch)
        train(epoch, train_loader, test_loader, net, loss_func, params, opts, _print)
        

pixelwise_loss = torch.nn.L1Loss().cuda()

def train(epoch, train_loader, test_loader, net, loss_func, params, opts, _print):
    global iter_num
    global best_fer2013_val_top1
    global best_fer2013_test_top1
    global best_rafdb_test_top1
    global best_rafdb_val_top1

    batch_time = utils.AverageMeter()
    losses_total = utils.AverageMeter()
    losses_cls = utils.AverageMeter()
    losses_rec = utils.AverageMeter()
    top1 = utils.AverageMeter()
    top3 = utils.AverageMeter()

    net.train()

    end = time.time()
    for i, data in enumerate(train_loader):
        img, label , target,point= data[0].to(device), data[1].to(device), data[2].to(device),data[3].numpy()
        N = label.size(0)        
        classification,ROI_feature_loss,pred_masked_patch = net(img,point,opts,mode = "train")
        
        loss = loss_func(classification, label, target)
        loss_mask = pixelwise_loss(pred_masked_patch, ROI_feature_loss)
        Total_loss = loss + opts.m*loss_mask

        prec1, prec3 = utils.accuracy(classification, target, topk=(1, 3))
        losses_total.update(Total_loss.item(), N)
        losses_cls.update(loss.item(), N)
        losses_rec.update(loss_mask.item(), N)
        top1.update(prec1[0].item(), N)
        top3.update(prec3[0].item(), N)

        params.zero_grad()
        Total_loss.backward()
        params.back_grad()
        batch_time.update(time.time() - end)
        end = time.time()
        iter_num += 1

        if iter_num % opts.print_freq == 0:
            _print('{0}_Iter: [{1}][{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})   '
                  'Loss {losses.val:.4f} ({losses.avg:.4f})   '
                  'Loss_cls {loss_cls.val:.4f} ({loss_cls.avg:.4f})   '
                  'Loss_mask {loss_mask.val:.4f} ({loss_mask.avg:.4f})  '
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})   '
                  'Prec@3 {top3.val:.3f} ({top3.avg:.3f})'.
                  format('train', iter_num, epoch+1,
                  batch_time=batch_time, losses=losses_total,loss_cls =losses_cls,loss_mask= losses_rec,top1=top1,top3=top3 ))

        if iter_num % opts.evaluate_freq == 0:
            Total_loss_test,top1_rafdb_test, top3_rafdb_test= test('test_raftest', epoch, test_loader['rafdb_test'], net, loss_func, opts, _print)
            _print('{0}_Iter: [{1}][{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})   '
                  'Loss {losses_test.val:.4f} ({losses_test.avg:.4f})   '
                  'Prec@1 {top1:.3f} '
                  'Prec@3 {top3:.3f} '.
                  format('test', iter_num, epoch+1,
                  batch_time=batch_time, losses_test=Total_loss_test,top1=top1_rafdb_test,top3=top3_rafdb_test ))
            if top1_rafdb_test > best_rafdb_test_top1:
                best_rafdb_test_top1 = top1_rafdb_test
            _print('RAF-DB TEST-ACC:{}'.format(top1_rafdb_test))

def test(test_type, epoch, data_loader, net, loss_func, opts, _print):
    global iter_num
    global best_fer2013_val_top1
    global best_fer2013_test_top1
    global best_rafdb_test_top1
    global best_rafdb_val_top1
    losses_total = utils.AverageMeter()
    top1 = utils.AverageMeter()
    top3 = utils.AverageMeter()

    net.eval()

    end = time.time()
    with torch.no_grad():
        for i, data in enumerate(data_loader):
            img, label, target,point = data[0].to(device), data[1].to(device), data[2].to(device),data[3].numpy()
            N = label.size(0)
            classification = net(img,point,opts,mode = "test")
            loss = loss_func(classification, label, target)

            Total_loss = loss 
            prec1, prec3 = utils.accuracy(classification, target, topk=(1, 3))
            top1.update(prec1[0].item(), N)
            top3.update(prec3[0].item(), N)
            losses_total.update(Total_loss.item(), N)


    return losses_total,top1.avg, top3.avg

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='RAF-DB')
    parser.add_argument('--snapshot', type=str, default='debug')
    parser.add_argument('--train_num', type=int, default=1)
    parser.add_argument('--train_dataset', type=str, default='RAF')
    parser.add_argument('--backbone_net', type=str,default='LResNet50E_IR')


    parser.add_argument('--evaluate_freq', type=int, default=628)
    parser.add_argument('--print_freq', type=int, default=628)
    
    parser.add_argument('--batch_size_train', type=int, default=16)
    parser.add_argument('--batch_size_test', type=int, default=16)
 


    parser.add_argument('--lr', type=str, default='0.01,0.001,0.0001,1e-4')
    parser.add_argument('--milestone', type=str, default='15,30,50')


    parser.add_argument('--start_epoch', type=int, default=0)
    parser.add_argument('--last_epoch', type=int, default=80)


    parser.add_argument('--loss_type', type=str, default='KL')
    # parser.add_argument('--loss_type', type=str, default='CE')

    parser.add_argument('--smooth_factor', type=float, default=8)
    parser.add_argument('--crop_size', type=int, default=112)
    parser.add_argument("--mask_size", type=int, default=10)

    parser.add_argument('--spatial_size', type=int, default=5)
    parser.add_argument('--feature_dim', type=int, default=256)
    parser.add_argument('--drop_ratio', type=float, default=0.4)
    parser.add_argument('--num_classes', type=int, default=7)
    parser.add_argument('--mask_ratio', type=float, default=0.5)
    parser.add_argument('--patch_half_length', type=float, default=2)
    parser.add_argument('--m', type=float, default=0.5)#loss weight coefficient
    parser.add_argument('--num_patch', type=int, default=68)
    parser.add_argument('--resume_net', type=str, default='')
    
    parser.add_argument('--train_data', type=str, default='/RAF_train_images.npy')
    parser.add_argument('--train_label', type=str, default='/RAF_train_labels.npy')
    parser.add_argument('--train_point', type=str, default='/RAF_train_points.npy')
    parser.add_argument('--test_data', type=str, default='/RAF_test_images.npy')
    parser.add_argument('--test_label', type=str, default='/RAF_test_labels.npy')
    parser.add_argument('--test_point', type=str, default='/RAF_test_points.npy')
    

    parser.add_argument('--save_dir', type=str, default='./results1')
    parser.add_argument('--modelDir', type=str, default='')
    parser.add_argument('--dataDir', help='data directory', type=str, default='/submit_code')
    parser.add_argument('--logDir', help='log directory', type=str, default='/submit_code/results1')
    parser.add_argument('--path_model', help='model path', type=str, default='/model/model_88.44.pkl')

    opts = parser.parse_args()


    main(opts)

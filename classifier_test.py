import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
from torch.optim import SGD, Adam
from utils import *
from tqdm import tqdm
from config import *
from dataset import *
import logging
import timm

import argparse
parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')

parser.add_argument('--img_dir', type=str, help='folder of images')
parser.add_argument('--outdir', type=str, help='folder to save denoiser and training log')
parser.add_argument('--batch', default=10, type=int, metavar='N')
parser.add_argument('--dataset', type=str, default="SIPADMEK")
parser.add_argument('--gpu', default=None, type=str,
                    help='id(s) for CUDA_VISIBLE_DEVICES')

args = parser.parse_args()
model = torch.load(r"experiment\sipadmek\best.pth").eval().cuda()


os.environ['CUDA_VISIBLE_DEVICES'] = "0"
test_dataset = get_dataset(args.dataset, mode="Train", img_dir=args.img_dir)
test_loader = DataLoader(test_dataset, batch_size=args.batch, shuffle=False)
acc_meter = AverageMeter()

with torch.no_grad():
    model.eval()
    for data in tqdm(test_loader):
        imgs, labels = data
        imgs = imgs.cuda()
        labels = labels.cuda()
        output = model(imgs)
        acc = accuracy(output, labels)

        acc_meter.update(acc[0].item(), imgs.shape[0])

print(f"Acc: {acc_meter.avg:.4f}")


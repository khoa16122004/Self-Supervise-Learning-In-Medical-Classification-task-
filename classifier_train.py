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

# Uncomment and use argparse if needed
import argparse
parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--lr', '--learning-rate', default=1e-3, type=float,
                    help='initial learning rate', dest='lr')
parser.add_argument('--img_dir', type=str, help='folder of images')
parser.add_argument('--outdir', type=str, help='folder to save denoiser and training log')
parser.add_argument('--epochs', default=10, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--batch', default=10, type=int, metavar='N')
parser.add_argument('--dataset', type=str, default="SIPADMEK")
parser.add_argument('--optimizer', default='Adam', type=str,
                    help='SGD, Adam, or Adam then SGD', choices=['SGD', 'Adam', 'AdamThenSGD'])
parser.add_argument('--gpu', default=None, type=str,
                    help='id(s) for CUDA_VISIBLE_DEVICES')

args = parser.parse_args()

# Use the config or argparse arguments to set these variables
# lr = args.lr
# epochs = args.epochs
# device = args.gpu if args.gpu else 'cuda:0'
# OUTDIR_TRAIN = args.outdir

os.environ['CUDA_VISIBLE_DEVICES'] = "0"

if not os.path.exists(args.outdir):
    os.mkdir(args.outdir)

# Set up logging
logging.basicConfig(filename=f'{args.outdir}/training_log.txt', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logging.info('Starting training process')

train_dataset = get_dataset(args.dataset, mode="Train", img_dir=args.img_dir)
val_dataset = get_dataset(args.dataset, mode="Val", img_dir=args.img_dir)
test_dataset = get_dataset(args.dataset, mode="Test", img_dir=args.img_dir)

train_loader = DataLoader(train_dataset, batch_size=args.batch, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=args.batch, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=args.batch, shuffle=False)


model = torch.load("experiment/sipadmek/best_vits.pth")
model.fc = nn.Linear(in_features=2048, out_features=3, bias=True)
model.cuda()

criterion = nn.CrossEntropyLoss(size_average=None, reduce=None, reduction='mean').cuda()
optimizer = Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

# Optionally, add a learning rate scheduler
scheduler = StepLR(optimizer, step_size=30, gamma=0.1)

acc_meter = AverageMeter()
losses_meter = AverageMeter()

best_acc = None

for epoch in tqdm(range(args.epochs)):
    losses_meter.reset()
    acc_meter.reset()
    
    model.train()
    for data in tqdm(train_loader):
        imgs, labels = data
        imgs = imgs.cuda()
        labels = labels.cuda()
        output = model(imgs)
        print(labels.shape)
        loss = criterion(output, labels)
        acc = accuracy(output, labels)


        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    scheduler.step()  # Step the scheduler at the end of the epoch if using

    with torch.no_grad():
        model.eval()
        for data in val_loader:
            imgs, labels = data
            imgs = imgs.cuda()
            labels = labels.cuda()
            output = model(imgs)
            loss = criterion(output, labels)
            acc = accuracy(output, labels)

            losses_meter.update(loss.item(), imgs.shape[0])
            acc_meter.update(acc[0].item(), imgs.shape[0])
    
    
    # Log epoch metrics
    logging.info(f"Epoch {epoch} Loss: {losses_meter.avg:.4f} Acc: {acc_meter.avg:.4f}")
    print(f"Epoch {epoch} Loss: {losses_meter.avg:.4f} Acc: {acc_meter.avg:.4f}")

    if not best_acc or acc_meter.avg > best_acc:
        best_acc = acc_meter.avg
        torch.save(model, f"{args.outdir}/best.pth")
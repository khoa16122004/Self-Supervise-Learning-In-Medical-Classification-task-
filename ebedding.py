from tqdm import tqdm
import torch.nn as nn
import os
import timm
import config
# from pytorch_lightning import seed_everything
from utils import init_logfile, log, AverageMeter
import torch
from torchcontrib.optim import SWA
import argparse
from dataset import get_dataset
from torch.utils.data import DataLoader

parser = argparse.ArgumentParser(description="DINO Train")
parser.add_argument('--mode', type=str, default='train')
parser.add_argument('--dataset', type=str, default="COVIDGR")
parser.add_argument('--batch', default=32 , type=int)
parser.add_argument('--outdir', type=str, default=".")
parser.add_argument('--lr', '--learning-rate', default=1e-3, type=float,
                    help='initial learning rate', dest='lr')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--lr_step_size', type=int, default=100,
                    help='How often to decrease learning by gamma.')
parser.add_argument('--gamma', type=float, default=0.1)
parser.add_argument('--epochs', type=int, default=200)
parser.add_argument('--img_path', type=str, default=None)
args = parser.parse_args()


def sim_loss(x, y): # similarity loss
    x = nn.functional.normalize(x, dim=-1, p=2)
    y = nn.functional.normalize(y, dim=-1, p=2)

    return 2 - 2 * (x * y).sum(dim=-1)

def train(train_loader,
          val_loader,
          model_vits,
          optimizer_vits,
          scheduler_vits,
          model_cnn,
          optimizer_cnn,
          scheduler_cnn,
          epochs,
          log_file_name):
    
    loss_train_meter = AverageMeter()
    loss_val_meter = AverageMeter()
    
    best_score = 0.0
    for i in tqdm(range(epochs)):
        model_cnn.train()
        model_vits.train()
        
        for imgs, _ in train_loader:
            imgs = imgs.cuda()
            
            pred_vits = model_vits(imgs)
            pred_cnn = model_cnn(imgs)
            loss = sim_loss(pred_vits, pred_cnn).mean()
            loss.backward()
            optimizer_cnn.step()
            optimizer_vits.step()
            scheduler_cnn.step()
            scheduler_vits.step()
            
            loss_train_meter.update(loss.item(), imgs.shape[0])
        
        with torch.no_grad():
            model_cnn.eval()
            model_vits.eval()
            for imgs, _ in val_loader:
                imgs = imgs.cuda()
            
                pred_vits = model_vits(imgs)
                pred_cnn = model_cnn(imgs)
                loss = sim_loss(pred_vits, pred_cnn).mean()
                loss_val_meter.update(loss.item(), imgs.shape[0])

        if not best_score or loss_val_meter.avg < best_score:
            best_score = loss_train_meter.avg
            torch.save(model_vits, os.path.join(args.outdir,"best_vits.pth"))
            torch.save(model_cnn, os.path.join(args.outdir,"best_cnn.pth"))
            
        log(log_file_name, f"{i}\t{loss_train_meter.avg}\t{loss_val_meter.avg}")
        loss_train_meter.reset()
        loss_val_meter.reset()
        
        
            
def main():
    os.environ["CUDA_VISIBLE_DEVICE"] = "0"
    # seed_everything(config.seed)
    os.makedirs(args.outdir, exist_ok=True)

    # dataset
    train_dataset = get_dataset(args.dataset, mode="Train")
    val_dataset = get_dataset(args.dataset, mode="Val")
    test_dataset = get_dataset(args.dataset, mode="Test")

    train_loader = DataLoader(train_dataset, batch_size=args.batch, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch, shuffle=False)

    log_file_name = os.path.join(args.outdir, 'log.txt')
    if not os.path.isfile(log_file_name):
        init_logfile(log_file_name, "epoch\Train_loss\Test_loss")
    
    # model
    model_cnn = timm.create_model(config.cnn_name, pretrained=True).cuda()
    model_vits = timm.create_model(config.vits_name, pretrained=True).cuda()
    
    # optimizer
    optimizer_cnn = SWA(torch.optim.Adam(model_cnn.parameters(), lr=1e-3))
    optimizer_vits = SWA(torch.optim.Adam(model_vits.parameters(), lr=1e-3))
    
    scheduler_cnn = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_cnn, T_max=16, eta_min=1e-6)
    scheduler_vits = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_vits, T_max=16, eta_min=1e-6)
    
    train(train_loader, val_loader, model_vits, optimizer_vits, scheduler_vits, model_cnn, optimizer_cnn, scheduler_cnn, args.epochs, log_file_name)

    
if __name__ == '__main__':
    main()
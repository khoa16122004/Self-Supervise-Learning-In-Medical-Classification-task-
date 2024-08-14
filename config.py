import os
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms

IMG_SIZE = (384, 384)

image_transform = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.RandomApply([transforms.ColorJitter(0.2, 0.2, 0.2), 
                            transforms.RandomPerspective(distortion=0.2)], p=0.3),
    transforms.RandomApply([transforms.ColorJitter(0.2, 0.2, 0,2), transforms.RandomAffine(degree=10), ], p=3),
    transforms.RandomVerticalFlip(p=0.3),
    transforms.RandomHorizontalFlip(p=0.3),
    transforms.ToTensor(),
])

valid_transform = transforms.Compose([transforms.Resize(IMG_SIZE), 
                                      transforms.ToTensor()])

vit_name = "resnet50" 
cnn_name = "vit_base_patch16_384"
seed = 16122004
num_class = 2


# COVID_CONFIG
covid_img_dir = r"Dataset\COVIDGR_1.0"
covid_label_str2num = {
    "P": 0,
    "N": 1,
}




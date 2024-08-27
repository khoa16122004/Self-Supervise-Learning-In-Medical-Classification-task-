import os
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms

IMG_SIZE = (384, 384)

image_transform = transforms.Compose([transforms.Resize(IMG_SIZE),
                                transforms.RandomApply([transforms.ColorJitter(0.2, 0.2, 0.2),transforms.RandomPerspective(distortion_scale=0.2),], p=0.3),
                                transforms.RandomApply([transforms.ColorJitter(0.2, 0.2, 0.2),transforms.RandomAffine(degrees=10),], p=0.3),
                                transforms.RandomVerticalFlip(p=0.3),
                                transforms.RandomHorizontalFlip(p=0.3),
                                transforms.ToTensor(),
#                                 transforms.Normalize(DATASET_IMAGE_MEAN, DATASET_IMAGE_STD), 
                               ])

valid_transform = transforms.Compose([transforms.Resize(IMG_SIZE), 
                                      transforms.ToTensor()])

vits_name = "resnet50" 
cnn_name = "vit_base_patch16_384"
seed = 16122004
num_class = 2




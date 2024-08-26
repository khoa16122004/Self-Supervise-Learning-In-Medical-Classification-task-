from torch.utils.data import Dataset
import os
from config import covid_label_str2num, seed, image_transform, valid_transform, covid_img_dir
from sklearn.model_selection import train_test_split
from PIL import Image

def get_dataset(dataset, 
                mode="Train"):
    if dataset == "COVIDGR":
        return COVIDGR(img_dir=covid_img_dir,
                       mode=mode)
    
class COVIDGR(Dataset):
    def __init__(self, img_dir, mode="Train"):
        
        img_paths = []
        labels = []
        
        for class_name in os.listdir(img_dir):
            class_folder = os.path.join(img_dir, class_name)
            for file_name in os.listdir(class_folder):
                file_path = os.path.join(class_folder, file_name)
                img_paths.append(file_path)
                labels.append(covid_label_str2num[class_name])
    
        train_imgs, test_imgs, train_labels, test_labels = train_test_split(
            img_paths, labels, test_size=0.1, random_state=seed
        )

        train_imgs, val_imgs, train_labels, val_labels = train_test_split(
            train_imgs, train_labels, test_size=0.1, random_state=seed
        )
        
        if mode == "Train":
            self.img_paths = train_imgs
            self.labels = train_labels
            self.transform = image_transform
        
        elif mode == "Val":
            self.img_paths = val_imgs
            self.labels = val_labels
            self.transform = valid_transform
        else:
            self.img_paths = test_imgs
            self.labels = test_labels
            self.transform = valid_transform

            
    def __len__(self):
        return len(self.img_paths)
    
    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        
        label = self.labels[idx]
        return img, label
    
# train_dataset = get_dataset("COVIDGR", mode="Val")
# print(len(train_dataset))

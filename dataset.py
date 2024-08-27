from torch.utils.data import Dataset
import os
from config import *
from sklearn.model_selection import train_test_split
from PIL import Image
from tqdm import tqdm
import random
from torchvision import transforms
import torch
import shutil



def get_dataset(dataset, 
                img_dir,
                mode="Train"):

    
    if dataset == "SIPADMEK":
        if mode == "Train":
            return SIPADMEK(img_dir=img_dir,
                            mode=mode)
        else:
            return SIPADMEK(img_dir=img_dir,
                            mode=mode,
                            transform=transforms.Compose([transforms.Resize((384, 384)), 
                                                          transforms.ToTensor()])
                            )
    
                

class SIPADMEK(Dataset):
    def extract_data(self, image_dir: str,
                class_name: str,
                output_dir="Dataset\SIPADMEK\process",
                class_map = {
                    "im_Dyskeratotic": 0, # abnormal
                    "im_Koilocytotic": 0, # abnormal
                    "im_Metaplastic": 1, # Benign
                    "im_Parabasal": 2, # normal
                    "im_Superficial-Intermediate": 2, # normal
                        }
                ):
    
    
        os.makedirs(output_dir, exist_ok=True) # check exist
        class_label = class_map[class_name]
        
        label_dir = os.path.join(output_dir, str(class_label))
        os.makedirs(label_dir, exist_ok=True)
        
        count = 0
        for file_name in tqdm(os.listdir(image_dir)):
            if "bmp" in file_name:
                count += 1
                file_path = os.path.join(image_dir, file_name)
                img = Image.open(file_path).convert("RGB")
                base_name = file_name.split(".")[0]
                output_path = os.path.join(label_dir, f"{class_name}{base_name}.png")
                img.save(output_path)
        print(count)


    def split_data(self, img_dir, train_size=0.7, val_size=0.1, test_size=0.2):
        random.seed("22520691")
        train_img, val_img, test_img = [], [], []
        train_label, val_label, test_label = [], [], []
        
        for label_name in os.listdir(img_dir):
            label_folder = os.path.join(img_dir, label_name)
            tmp = []
            tmp_label = []
            
            for file_name in os.listdir(label_folder):
                file_path = os.path.join(label_folder, file_name)
                tmp.append(file_path)
                tmp_label.append(label_name)
            
            combined = list(zip(tmp, tmp_label))
            random.shuffle(combined)
            tmp, tmp_label = zip(*combined)
            
            n_train = int(len(tmp) * train_size)
            n_val = int(len(tmp) * val_size)
            
            train_img += tmp[:n_train]
            val_img += tmp[n_train:n_train + n_val]
            test_img += tmp[n_train + n_val:]
            
            train_label += tmp_label[:n_train]
            val_label += tmp_label[n_train:n_train + n_val]
            test_label += tmp_label[n_train + n_val:]
            
        return train_img, val_img, test_img, train_label, val_label, test_label

    def save_to_txt(self, image_paths, labels, split_name, output_dir):
        txt_file_path = os.path.join(output_dir, f"{split_name}.txt")
        with open(txt_file_path, 'w') as f:
            for img_path, label_name in zip(image_paths, labels):
                f.write(f"{img_path}, {label_name}\n")
    

    
    
    def __init__(self, img_dir, mode="Train",
                 transform=transforms.Compose([transforms.Resize((384, 384)),
                                               transforms.RandomApply([transforms.ColorJitter(0.2, 0.2, 0.2),transforms.RandomPerspective(distortion_scale=0.2),], p=0.3),
                                               transforms.RandomApply([transforms.ColorJitter(0.2, 0.2, 0.2),transforms.RandomAffine(degrees=10),], p=0.3),
                                               transforms.RandomVerticalFlip(p=0.3),
                                               transforms.RandomHorizontalFlip(p=0.3),
                                               transforms.ToTensor(),
                                               ])):
        
        self.transform = transform
        
        # train_img, val_img, test_img, train_label, val_label, test_label = self.split_data(img_dir)
        if mode == "Train":
            path_file = os.path.join(img_dir, "train.txt")
        elif mode == "Val":
            path_file = os.path.join(img_dir, "val.txt")
        else:
            path_file = os.path.join(img_dir, "test.txt")
        
        with open(path_file, 'r') as f:
            lines = f.readlines()
            lines = [line.strip() for line in lines]
            img_paths = [line.split(", ")[0] for line in lines]
            labels = [line.split(", ")[1] for line in lines]
        self.img_paths, self.labels = img_paths, labels

    def __len__(self) -> int:
        return len(self.img_paths)
    
    def __getitem__(self, index):
        img_path = self.img_paths[index]
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        label = int(self.labels[index])
        return img, label
    
a = SIPADMEK(img_dir="Dataset\SIPADMEK\process", mode="Train")

# train_img, val_img, test_img, train_label, val_label, test_label = a.split_data("Dataset\SIPADMEK\process")
# a.save_to_txt(train_img, train_label, "train", "Dataset\SIPADMEK\process")
# a.save_to_txt(test_img, test_label, "test", "Dataset\SIPADMEK\process")
# a.save_to_txt(val_img, val_label, "val", "Dataset\SIPADMEK\process")
           

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



# extract_data("D:\Self-Supervise-Learning-In-Medical-Classification-task-\Dataset\SIPADMEK\im_Dyskeratotic\CROPPED",
#              "im_Dyskeratotic")

# extract_data("D:\Self-Supervise-Learning-In-Medical-Classification-task-\Dataset\SIPADMEK\im_Koilocytotic\CROPPED", 
#              "im_Koilocytotic")

# extract_data("D:\Self-Supervise-Learning-In-Medical-Classification-task-\Dataset\SIPADMEK\im_Metaplastic\im_Metaplastic\CROPPED", 
#              "im_Metaplastic")

# extract_data("D:\Self-Supervise-Learning-In-Medical-Classification-task-\Dataset\SIPADMEK\im_Parabasal\im_Parabasal\CROPPED", 
#              "im_Parabasal")

# extract_data("D:\Self-Supervise-Learning-In-Medical-Classification-task-\Dataset\SIPADMEK\im_Superficial-Intermediate\im_Superficial-Intermediate\CROPPED", 
#              "im_Superficial-Intermediate")

##### step03_dataset_and_model.py

import os
import random
from dataclasses import dataclass
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torchvision import transforms, models
from torch.utils.data import Dataset

import torch.nn.functional as F

SEED = 42

def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

set_seed()

PROJECT_ROOT = r"C:\Users\holyb\OneDrive\Desktop\shoe_wear_project"

IMAGE_DIR = os.path.join(PROJECT_ROOT, "data", "images", "train")
ANNOT_DIR = os.path.join(PROJECT_ROOT, "data", "annotations")
OUTPUT_MODEL_DIR = os.path.join(PROJECT_ROOT, "outputs", "models")
OUTPUT_LOG_DIR = os.path.join(PROJECT_ROOT, "outputs", "logs")

os.makedirs(OUTPUT_MODEL_DIR, exist_ok=True)
os.makedirs(OUTPUT_LOG_DIR, exist_ok=True)

@dataclass
class Config:
    project_root: str = PROJECT_ROOT

    image_dir: str = IMAGE_DIR
    annot_dir: str = ANNOT_DIR

    all_images_csv: str = os.path.join(ANNOT_DIR, "all_images.csv")
    seed_label_csv: str = os.path.join(ANNOT_DIR, "labels_seed.csv")

    binary_ckpt: str = os.path.join(OUTPUT_MODEL_DIR, "binary_stage1.pt")
    three_ckpt_tmpl: str = os.path.join(OUTPUT_MODEL_DIR, "three_stage{}.pt")

    batch_size: int = 32
    num_workers: int = 0         
    lr: float = 3e-4
    epochs_binary: int = 20
    epochs_three: int = 20

    w_seed: float = 1.0
    w_pseudo_core: float = 0.8
    w_pseudo_mid: float = 0.5

    log_dir: str = OUTPUT_LOG_DIR

    def __post_init__(self):
        os.makedirs(self.log_dir, exist_ok=True)


cfg = Config()

class ShoeWearDataset(Dataset):
    """
    df는 반드시 다음 컬럼을 포함:
      - image_path
      - label
      - weight (없으면 1.0 처리)
    """
    def __init__(self, df, transform=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        img_path = row["image_path"]
        label = int(row["label"])
        weight = float(row.get("weight", 1.0))

        img = Image.open(img_path).convert("RGB")

        if self.transform:
            img = self.transform(img)

        return img, label, weight

train_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=5),
    transforms.CenterCrop((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5]*3, std=[0.25]*3),
])

val_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5]*3, std=[0.25]*3),
])

def build_binary_model():
    """
    output: 2 (class 0 vs class 2)
    """
    base = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    in_features = base.fc.in_features
    base.fc = nn.Linear(in_features, 2)
    return base


def build_three_class_model():
    """
    output: 3 (class 0, 1, 2)
    """
    base = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    in_features = base.fc.in_features
    base.fc = nn.Linear(in_features, 3)
    return base


print("step03_dataset_and_model.py 로드 완료")

class ShoeWearNet(nn.Module):
    def __init__(self, num_classes: int = 3, fpn_channels: int = 256):
        super().__init__()
        base = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

        self.stem = nn.Sequential(
            base.conv1,
            base.bn1,
            base.relu,
            base.maxpool,
        )
        self.layer1 = base.layer1  
        self.layer2 = base.layer2  
        self.layer3 = base.layer3  

        self.lateral2 = nn.Conv2d(64, fpn_channels, kernel_size=1)    
        self.lateral3 = nn.Conv2d(128, fpn_channels, kernel_size=1)   
        self.lateral4 = nn.Conv2d(256, fpn_channels, kernel_size=1)   

        self.smooth2 = nn.Conv2d(fpn_channels, fpn_channels, kernel_size=3, padding=1)
        self.smooth3 = nn.Conv2d(fpn_channels, fpn_channels, kernel_size=3, padding=1)
        self.smooth4 = nn.Conv2d(fpn_channels, fpn_channels, kernel_size=3, padding=1)

        self.att_mlp = nn.Sequential(
            nn.Linear(fpn_channels, fpn_channels // 2),
            nn.ReLU(inplace=True),
            nn.Linear(fpn_channels // 2, 1),
        )

        self.fc = nn.Sequential(
            nn.Linear(fpn_channels, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.stem(x)   
        c2 = self.layer1(x) 
        c3 = self.layer2(c2) 
        c4 = self.layer3(c3)

        p4 = self.lateral4(c4)  
        p3 = self.lateral3(c3) 
        p2 = self.lateral2(c2) 

        p3 = p3 + F.interpolate(p4, size=p3.shape[-2:], mode="nearest")
        p2 = p2 + F.interpolate(p3, size=p2.shape[-2:], mode="nearest")

 
        p4 = self.smooth4(p4)
        p3 = self.smooth3(p3)
        p2 = self.smooth2(p2)

        b, c, h, w = p2.shape
        feat = p2.view(b, c, h * w).permute(0, 2, 1) 

        scores = self.att_mlp(feat)     
        scores = scores.squeeze(-1)     
        attn = torch.softmax(scores, dim=1)  

        attn = attn.unsqueeze(-1)   
        weighted_feat = (feat * attn).sum(dim=1)  

        logits = self.fc(weighted_feat)  
        return logits


def build_wear_model():
    return ShoeWearNet(num_classes=3)

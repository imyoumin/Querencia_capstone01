# step03_dataset_and_model.py
# FastAPI 추론 서버 전용 (불필요한 경로/CSV/로그/학습 설정 제거)

import torch
import torch.nn as nn
from torchvision import transforms, models
import torch.nn.functional as F


# ========================================================
# 1) 이미지 Transform (학습 때와 동일해야 함)
# ========================================================
val_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5]*3, std=[0.25]*3),
])


# ========================================================
# 2) 최종 모델 구조 (ShoeWearNet)
# ========================================================
class ShoeWearNet(nn.Module):
    def __init__(self, num_classes: int = 3, fpn_channels: int = 256):
        super().__init__()
        base = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

        # ResNet18 backbone
        self.stem = nn.Sequential(
            base.conv1,
            base.bn1,
            base.relu,
            base.maxpool,
        )
        self.layer1 = base.layer1  # C2
        self.layer2 = base.layer2  # C3
        self.layer3 = base.layer3  # C4

        # FPN lateral conv
        self.lateral2 = nn.Conv2d(64, fpn_channels, kernel_size=1)
        self.lateral3 = nn.Conv2d(128, fpn_channels, kernel_size=1)
        self.lateral4 = nn.Conv2d(256, fpn_channels, kernel_size=1)

        # smoothing conv
        self.smooth2 = nn.Conv2d(fpn_channels, fpn_channels, kernel_size=3, padding=1)
        self.smooth3 = nn.Conv2d(fpn_channels, fpn_channels, kernel_size=3, padding=1)
        self.smooth4 = nn.Conv2d(fpn_channels, fpn_channels, kernel_size=3, padding=1)

        # Local Attention
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

        # FPN
        p4 = self.lateral4(c4)
        p3 = self.lateral3(c3)
        p2 = self.lateral2(c2)

        p3 = p3 + nn.functional.interpolate(p4, size=p3.shape[-2:], mode="nearest")
        p2 = p2 + nn.functional.interpolate(p3, size=p2.shape[-2:], mode="nearest")

        p4 = self.smooth4(p4)
        p3 = self.smooth3(p3)
        p2 = self.smooth2(p2)

        # Local Attention Pooling
        b, c, h, w = p2.shape
        feat = p2.view(b, c, h*w).permute(0, 2, 1)
        scores = self.att_mlp(feat).squeeze(-1)
        attn = torch.softmax(scores, dim=1).unsqueeze(-1)
        weighted_feat = (feat * attn).sum(dim=1)

        logits = self.fc(weighted_feat)
        return logits


# ========================================================
# 3) 빌드 함수
# ========================================================
def build_wear_model():
    return ShoeWearNet(num_classes=3)

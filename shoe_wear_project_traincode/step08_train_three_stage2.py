##### step08_train_three_stage2.py

import os
import pandas as pd
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from datetime import datetime

from tqdm import tqdm

from step03_dataset_and_model import (
    cfg, ShoeWearDataset,
    train_transform, val_transform,
    build_three_class_model, set_seed
)
from logging_utils import append_log_csv


# ================================================================
# 1) 기본 설정
# ================================================================
print("===============================================================")
print("[Step08] 3-Class Training Stage 2 시작 (merged_train_stage3.csv)")
print("===============================================================")

set_seed()
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"▶ Device: {device}")

MERGED_STAGE3_CSV = os.path.join(cfg.annot_dir, "merged_train_stage3.csv")

THREE_STAGE1_CKPT = cfg.three_ckpt_tmpl.format(1)  # 초기 가중치
THREE_STAGE2_CKPT = cfg.three_ckpt_tmpl.format(2)  # Stage2 최종 모델 저장

LOG_CSV = os.path.join(cfg.log_dir, "three_stage2_train_log.csv")
log_fields = [
    "time",
    "epoch",
    "train_loss",
    "val_loss",
    "val_acc",
    "val_f1_macro",
]

print("Using device:", device)
if device == "cuda":
    print("GPU 이름:", torch.cuda.get_device_name(0))

# ================================================================
# 2) 데이터 불러오기
# ================================================================
print("\n[1] Stage3 병합 데이터(Seed + 3-Class Pseudo) 불러오는 중...")

if not os.path.isfile(MERGED_STAGE3_CSV):
    raise FileNotFoundError(f"merged_train_stage3.csv 를 찾을 수 없습니다: {MERGED_STAGE3_CSV}")

df = pd.read_csv(MERGED_STAGE3_CSV)

print(f"    - 총 데이터 개수: {len(df)}")
print("    - 라벨 분포:")
print(df["label"].value_counts())

# Train / Validation Split
print("\n[2] Train / Validation 분리...")
train_df, val_df = train_test_split(
    df,
    test_size=0.2,
    stratify=df["label"],
    random_state=42
)

print(f"    → Train: {len(train_df)}개 | Val: {len(val_df)}개")

# ================================================================
# 3) Dataset / Loader
# ================================================================
print("\n[3] Dataset / DataLoader 구성...")

train_ds = ShoeWearDataset(train_df, transform=train_transform)
val_ds = ShoeWearDataset(val_df, transform=val_transform)

train_loader = DataLoader(
    train_ds,
    batch_size=cfg.batch_size,
    shuffle=True,
    num_workers=cfg.num_workers
)
val_loader = DataLoader(
    val_ds,
    batch_size=cfg.batch_size,
    shuffle=False,
    num_workers=cfg.num_workers
)

# ================================================================
# 4) Model / Loss / Optimizer
#    - Stage1에서 학습된 three_stage1.pt를 로드 후 이어서 학습
# ================================================================
print("\n[4] 모델 생성 및 Optimizer 설정...")

model = build_three_class_model().to(device)

if not os.path.isfile(THREE_STAGE1_CKPT):
    print(f"    - 경고: Stage1 체크포인트를 찾지 못했습니다. ({THREE_STAGE1_CKPT})")
    print("      → 랜덤 초기화된 모델로 Stage2를 학습합니다.")
else:
    print(f"    - Stage1 체크포인트 로드: {THREE_STAGE1_CKPT}")
    state = torch.load(THREE_STAGE1_CKPT, map_location=device)
    model.load_state_dict(state)
    print("      → Stage1 가중치 로드 완료.")

criterion = nn.CrossEntropyLoss(reduction="none")
optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr)

best_val_loss = float("inf")
patience = 5
no_improve = 0

# ================================================================
# 5) Train / Eval 함수 정의
# ================================================================
def train_one_epoch():
    model.train()
    total_loss = 0.0

    for imgs, labels, weights in tqdm(train_loader, desc="    [Train] ", ncols=100):
        imgs = imgs.to(device)
        labels = labels.to(device)
        weights = weights.to(device)

        optimizer.zero_grad()
        logits = model(imgs)

        loss_raw = criterion(logits, labels)
        loss = (loss_raw * weights).mean()

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * len(imgs)

    return total_loss / len(train_loader.dataset)


def evaluate():
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    preds_all = []
    labels_all = []

    for imgs, labels, weights in tqdm(val_loader, desc="    [Val]   ", ncols=100):
        imgs = imgs.to(device)
        labels = labels.to(device)
        weights = weights.to(device)

        with torch.no_grad():
            logits = model(imgs)

        loss_raw = criterion(logits, labels)
        loss = (loss_raw * weights).mean()

        total_loss += loss.item() * len(imgs)

        preds = logits.argmax(dim=1)
        preds_all.extend(preds.cpu().tolist())
        labels_all.extend(labels.cpu().tolist())

        correct += (preds == labels).sum().item()
        total += len(imgs)

    from sklearn.metrics import f1_score
    f1_macro = f1_score(labels_all, preds_all, average="macro")

    acc = correct / total
    return total_loss / len(val_loader.dataset), acc, f1_macro

# ================================================================
# 6) Epoch 루프
# ================================================================
print("\n===============================================================")
print("▶▶ 3-Class Training Stage 2 시작")
print("===============================================================")

for epoch in range(cfg.epochs_three):
    print(f"\n------ Epoch {epoch+1}/{cfg.epochs_three} ------")

    train_loss = train_one_epoch()
    val_loss, val_acc, val_f1_macro = evaluate()

    print(f"    Epoch {epoch+1}")
    print(f"    - Train Loss: {train_loss:.4f}")
    print(f"    - Val Loss:   {val_loss:.4f}")
    print(f"    - Val Acc:    {val_acc:.3f}")
    print(f"    - Val F1(M):  {val_f1_macro:.3f}")

    # 로그 저장
    append_log_csv(
        LOG_CSV,
        {
            "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "val_f1_macro": val_f1_macro,
        },
        log_fields
    )

    # Best model 저장
    if val_loss < best_val_loss - 1e-4:
        best_val_loss = val_loss
        no_improve = 0
        torch.save(model.state_dict(), THREE_STAGE2_CKPT)
        print("    → Best model 갱신됨")
    else:
        no_improve += 1
        print(f"    (개선 없음 {no_improve}/{patience})")

        if no_improve >= patience:
            print("    → Early Stopping 발동")
            break

print("\n===============================================================")
print("▶▶ 3-Class Training Stage 2 완료")
print("===============================================================")
print(f"Best Val Loss: {best_val_loss:.4f}")
print(f"Model Saved: {THREE_STAGE2_CKPT}")
print(f"Log Saved: {LOG_CSV}")

##### step04_train_binary.py

import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from step03_dataset_and_model import (
    cfg, ShoeWearDataset,
    train_transform, val_transform,
    build_binary_model, set_seed
)

from logging_utils import append_log_csv, save_confusion_matrix_txt
from datetime import datetime

# ============================================================
# 1) 설정
# ============================================================
set_seed()

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", device)

LOG_CSV = os.path.join(cfg.log_dir, "binary_train_log.csv")
CM_TXT = os.path.join(cfg.log_dir, "binary_best_confusion_matrix.txt")

# 로그 CSV 컬럼 정의
log_fields = ["epoch", "train_loss", "val_loss", "val_acc", "time"]

# ============================================================
# 2) Seed Label CSV 불러오기 (0과 2만 있음)
# ============================================================
df_seed = pd.read_csv(cfg.seed_label_csv)

print("\n==== Seed Label 분포 (원본) ====")
print(df_seed["label"].value_counts())

# Binary 형태 (0 → 0, 2 → 1)
df_seed_binary = df_seed.copy()
df_seed_binary["label"] = df_seed_binary["label"].map({0: 0, 2: 1})

print("\n==== Binary Label 분포 (0 vs 1) ====")
print(df_seed_binary["label"].value_counts())

# ============================================================
# 3) Train/Val Split
# ============================================================
train_df, val_df = train_test_split(
    df_seed_binary,
    test_size=0.2,
    stratify=df_seed_binary["label"],
    random_state=42
)

print("\nTrain size:", len(train_df), "Val size:", len(val_df))

train_df["weight"] = 1.0
val_df["weight"] = 1.0

# ============================================================
# 4) Dataset / Loader
# ============================================================
train_ds = ShoeWearDataset(train_df, transform=train_transform)
val_ds = ShoeWearDataset(val_df, transform=val_transform)

train_loader = DataLoader(train_ds, batch_size=cfg.batch_size,
                          shuffle=True, num_workers=cfg.num_workers)
val_loader = DataLoader(val_ds, batch_size=cfg.batch_size,
                        shuffle=False, num_workers=cfg.num_workers)

# ============================================================
# 5) Model / Loss / Optimizer
# ============================================================
model = build_binary_model().to(device)
criterion = nn.CrossEntropyLoss(reduction="none")
optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr)

best_val_loss = float("inf")
patience = 5
no_improve = 0


# ============================================================
# 6) Train/Val Loop
# ============================================================
def train_one_epoch(model, loader, optimizer):
    model.train()
    total_loss = 0

    for imgs, labels, weights in loader:
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

    return total_loss / len(loader.dataset)


def evaluate(model, loader):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for imgs, labels, weights in loader:
            imgs = imgs.to(device)
            labels = labels.to(device)
            weights = weights.to(device)

            logits = model(imgs)
            loss_raw = criterion(logits, labels)
            loss = (loss_raw * weights).mean()

            total_loss += loss.item() * len(imgs)

            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += len(imgs)

            all_labels.extend(labels.cpu().numpy().tolist())
            all_preds.extend(preds.cpu().numpy().tolist())

    acc = correct / total
    cm = confusion_matrix(all_labels, all_preds)

    return total_loss / len(loader.dataset), acc, cm


# ============================================================
# 7) Epoch 반복
# ============================================================
print("\n==== Binary Training Start ====")

for epoch in range(cfg.epochs_binary):
    train_loss = train_one_epoch(model, train_loader, optimizer)
    val_loss, val_acc, cm = evaluate(model, val_loader)

    print(f"[Epoch {epoch+1:02d}] "
          f"train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | val_acc={val_acc:.3f}")

    # ------------------------------------------------------------
    # 로그 기록 (CSV)
    # ------------------------------------------------------------
    append_log_csv(
        LOG_CSV,
        {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        },
        log_fields
    )

    # Best model 저장
    if val_loss < best_val_loss - 1e-4:
        best_val_loss = val_loss
        no_improve = 0

        torch.save(model.state_dict(), cfg.binary_ckpt)
        print("  → best model updated")

        # Confusion Matrix 저장
        save_confusion_matrix_txt(
            CM_TXT,
            cm,
            info={"epoch": epoch + 1, "val_loss": val_loss, "val_acc": val_acc}
        )

    else:
        no_improve += 1
        if no_improve >= patience:
            print("  → Early stopping triggered")
            break


print("\n==== Binary Training Finished ====")
print("Best Val Loss:", best_val_loss)
print("Model saved to:", cfg.binary_ckpt)
print("Log saved to:", LOG_CSV)

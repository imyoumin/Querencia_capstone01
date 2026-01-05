##### step09_generate_pseudo_final.py

import os
import pandas as pd
import torch
from torch.utils.data import DataLoader
from datetime import datetime
from tqdm import tqdm

from step03_dataset_and_model import (
    cfg, ShoeWearDataset,
    val_transform, build_three_class_model, set_seed
)
from logging_utils import append_log_csv


# ================================================================
# 1) 기본 설정
# ================================================================
set_seed()
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

# 최종 3-class Stage2 모델 (step08에서 저장한 것)
THREE_STAGE2_CKPT = cfg.three_ckpt_tmpl.format(2)

ALL_IMAGES_CSV = cfg.all_images_csv
SEED_LABEL_CSV = cfg.seed_label_csv

# 확률 + argmax 라벨 저장용
PSEUDO_FINAL_CSV = os.path.join(cfg.annot_dir, "pseudo_three_stage2_probs.csv")
# Seed + Final pseudo 병합 CSV
MERGED_STAGE4_CSV = os.path.join(cfg.annot_dir, "merged_train_stage4.csv")

# 로그 파일
LOG_CSV = os.path.join(cfg.log_dir, "pseudo_three_stage2_log.csv")
log_fields = [
    "time",
    "total_images",
    "seed_count",
    "pseudo_0_all",
    "pseudo_1_all",
    "pseudo_2_all",
    "pseudo_0_after_seed",
    "pseudo_1_after_seed",
    "pseudo_2_after_seed",
    "merged_total",
    "merged_0",
    "merged_1",
    "merged_2",
]

print("ALL_IMAGES_CSV:", ALL_IMAGES_CSV)
print("SEED_LABEL_CSV:", SEED_LABEL_CSV)
print("PSEUDO_FINAL_CSV:", PSEUDO_FINAL_CSV)
print("MERGED_STAGE4_CSV:", MERGED_STAGE4_CSV)
print("CKPT(three_stage2):", THREE_STAGE2_CKPT)
print("LOG_CSV:", LOG_CSV)


# ================================================================
# 2) 데이터 불러오기
# ================================================================
print("\n[1] 전체 이미지 / seed 라벨 로딩 중...")
if not os.path.isfile(ALL_IMAGES_CSV):
    raise FileNotFoundError(f"all_images.csv 를 찾을 수 없습니다: {ALL_IMAGES_CSV}")
if not os.path.isfile(SEED_LABEL_CSV):
    raise FileNotFoundError(f"labels_seed.csv 를 찾을 수 없습니다: {SEED_LABEL_CSV}")

df_all = pd.read_csv(ALL_IMAGES_CSV)
df_seed = pd.read_csv(SEED_LABEL_CSV)

seed_paths_set = set(df_seed["image_path"].tolist())

print(f"    - 전체 이미지 개수: {len(df_all)}")
print(f"    - Seed 라벨 개수: {len(df_seed)}")
print("    - Seed 라벨 분포:")
print(df_seed["label"].value_counts())


# ================================================================
# 3) 최종 3-Class Stage2 모델 로드
# ================================================================
print("\n[2] 3-Class Stage2 모델 로드 중...")
if not os.path.isfile(THREE_STAGE2_CKPT):
    raise FileNotFoundError(f"3-Class Stage2 체크포인트를 찾을 수 없습니다: {THREE_STAGE2_CKPT}")

model = build_three_class_model().to(device)
state = torch.load(THREE_STAGE2_CKPT, map_location=device)
model.load_state_dict(state)
model.eval()
print("    → 3-Class Stage2 모델 로드 완료.")


# ================================================================
# 4) 전체 이미지 확률 추론 + argmax 라벨
# ================================================================
print("\n[3] 전체 이미지에 대해 p0/p1/p2 + argmax 라벨 추론 중...")

df_all_for_ds = df_all.copy()
df_all_for_ds["label"] = 0
df_all_for_ds["weight"] = 1.0

ds_all = ShoeWearDataset(df_all_for_ds, transform=val_transform)
loader_all = DataLoader(
    ds_all,
    batch_size=64,
    shuffle=False,
    num_workers=cfg.num_workers,
)

p0_list, p1_list, p2_list = [], [], []
pred_list = []

with torch.no_grad():
    for imgs, labels, weights in tqdm(loader_all, desc="[3] Inference", ncols=80):
        imgs = imgs.to(device)
        logits = model(imgs)
        probs = torch.softmax(logits, dim=1)  # [B,3]

        p0_list.extend(probs[:, 0].cpu().numpy().tolist())
        p1_list.extend(probs[:, 1].cpu().numpy().tolist())
        p2_list.extend(probs[:, 2].cpu().numpy().tolist())

        preds = probs.argmax(dim=1)  # argmax: 가장 확률 높은 클래스
        pred_list.extend(preds.cpu().numpy().tolist())

df_probs = df_all.copy()
df_probs["p0"] = p0_list
df_probs["p1"] = p1_list
df_probs["p2"] = p2_list
df_probs["pred_label"] = pred_list

df_probs.to_csv(PSEUDO_FINAL_CSV, index=False, encoding="utf-8-sig")
print(f"    → 확률 + argmax 라벨 CSV 저장 완료: {PSEUDO_FINAL_CSV}")
print("    예시 5개:")
print(df_probs.head())


# ================================================================
# 5) argmax 기반 pseudo-label 적용 (전체 이미지에 대해)
# ================================================================
print("\n[4] argmax 기반 pseudo-label 적용 중...")

df_pseudo = df_probs.copy()
df_pseudo["label"] = df_pseudo["pred_label"].astype(int)
df_pseudo["weight"] = cfg.w_pseudo_mid       # 최종 pseudo는 mid weight로 설정
df_pseudo["source"] = "pseudo_from_three_stage2"

pseudo_counts_all = df_pseudo["label"].value_counts()
print("    - 전체 이미지에 대한 argmax 라벨 분포:")
print(pseudo_counts_all)


# ================================================================
# 6) Seed 이미지 제외
#    (Seed는 사람이 라벨링한 값 그대로 사용)
# ================================================================
print("\n[5] Seed 이미지 제외 중...")

mask_not_seed = ~df_pseudo["image_path"].isin(seed_paths_set)
df_pseudo_clean = df_pseudo[mask_not_seed].copy()

pseudo_counts_after = df_pseudo_clean["label"].value_counts()
print("    - Seed 제외 후 pseudo 라벨 분포:")
print(pseudo_counts_after)


# ================================================================
# 7) Seed + 최종 pseudo 병합 → Stage4 학습용 CSV
# ================================================================
print("\n[6] Seed + Final pseudo 병합 중...")

# Seed 쪽: 사람이 라벨링한 데이터
df_seed_stage = df_seed.copy()
df_seed_stage["weight"] = cfg.w_seed
df_seed_stage["source"] = "seed"
df_seed_stage = df_seed_stage[["image_path", "label", "weight", "source"]]

# Pseudo 쪽: 최종 Stage2 모델 argmax 결과
df_pseudo_final = df_pseudo_clean[["image_path", "label", "weight", "source"]]

merged = pd.concat([df_seed_stage, df_pseudo_final], ignore_index=True)

merged_counts = merged["label"].value_counts()
print("    - 병합 후 전체 학습 데이터 개수:", len(merged))
print("    - 병합 후 라벨 분포:")
print(merged_counts)
print("    - 병합 후 라벨 비율(%):")
print((merged_counts / len(merged) * 100).round(2))

merged.to_csv(MERGED_STAGE4_CSV, index=False, encoding="utf-8-sig")
print(f"\n✅ Stage4 학습용 merged CSV 저장 완료: {MERGED_STAGE4_CSV}")


# ================================================================
# 8) 로그 기록
# ================================================================
append_log_csv(
    LOG_CSV,
    {
        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "total_images": len(df_all),
        "seed_count": len(df_seed),

        "pseudo_0_all": int(pseudo_counts_all.get(0, 0)),
        "pseudo_1_all": int(pseudo_counts_all.get(1, 0)),
        "pseudo_2_all": int(pseudo_counts_all.get(2, 0)),

        "pseudo_0_after_seed": int(pseudo_counts_after.get(0, 0)),
        "pseudo_1_after_seed": int(pseudo_counts_after.get(1, 0)),
        "pseudo_2_after_seed": int(pseudo_counts_after.get(2, 0)),

        "merged_total": int(len(merged)),
        "merged_0": int(merged_counts.get(0, 0)),
        "merged_1": int(merged_counts.get(1, 0)),
        "merged_2": int(merged_counts.get(2, 0)),
    },
    log_fields
)

print(f"\n📝 로그 기록 완료: {LOG_CSV}")
print("🎯 step09_generate_pseudo_final 완료.")

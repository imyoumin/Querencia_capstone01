##### step05_generate_pseudo_binary.py

import os
import pandas as pd
import torch
from torch.utils.data import DataLoader
from datetime import datetime

from tqdm import tqdm  # ← 진행바 표시용

from step03_dataset_and_model import (
    cfg, ShoeWearDataset,
    val_transform, build_binary_model, set_seed
)
from logging_utils import append_log_csv

# ==============================
# 0) 기본 설정
# ==============================
set_seed()
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", device)

# 파일 경로
ALL_IMAGES_CSV = cfg.all_images_csv
SEED_LABEL_CSV = cfg.seed_label_csv
BINARY_CKPT = cfg.binary_ckpt

# 출력파일 경로
PSEUDO_BINARY_CSV = os.path.join(cfg.annot_dir, "pseudo_stage1_binary_probs.csv")
MERGED_STAGE2_CSV = os.path.join(cfg.annot_dir, "merged_train_stage2.csv")

# 로그 경로
LOG_CSV = os.path.join(cfg.log_dir, "pseudo_binary_stage1_log.csv")
log_fields = [
    "time",
    "total_images",
    "seed_count",
    "pseudo_0_before_seed",
    "pseudo_1_before_seed",
    "pseudo_2_before_seed",
    "pseudo_0_after_seed",
    "pseudo_1_after_seed",
    "pseudo_2_after_seed",
    "merged_total",
    "merged_0",
    "merged_1",
    "merged_2",
    "P0_STRICT",
    "P2_STRICT",
    "MARGIN_STRICT",
    "MID_LOW",
    "MID_HIGH",
    "MAX_CONF_FOR_MID"
]

print("ALL_IMAGES_CSV:", ALL_IMAGES_CSV)
print("SEED_LABEL_CSV:", SEED_LABEL_CSV)
print("BINARY_CKPT:", BINARY_CKPT)
print("PSEUDO_BINARY_CSV:", PSEUDO_BINARY_CSV)
print("MERGED_STAGE2_CSV:", MERGED_STAGE2_CSV)
print("LOG_CSV:", LOG_CSV)

# ==============================
# 1) 데이터 불러오기
# ==============================
print("\n[1] 전체 이미지 / seed 라벨 로딩 중...")
df_all = pd.read_csv(ALL_IMAGES_CSV)
print(f"    - 전체 이미지 개수: {len(df_all)}")

df_seed = pd.read_csv(SEED_LABEL_CSV)
print(f"    - Seed 라벨 개수: {len(df_seed)}")
print("    - Seed 라벨 분포:")
print(df_seed["label"].value_counts())

seed_paths_set = set(df_seed["image_path"].tolist())

# ==============================
# 2) 학습된 Binary 모델 로드
# ==============================
print("\n[2] Binary 모델 로드 중...")
if not os.path.isfile(BINARY_CKPT):
    raise FileNotFoundError(f"Binary checkpoint 를 찾을 수 없습니다: {BINARY_CKPT}")

model = build_binary_model().to(device)
state = torch.load(BINARY_CKPT, map_location=device)
model.load_state_dict(state)
model.eval()
print("    → Binary 모델 로드 완료.")

# ==============================
# 3) 전체 이미지에 대해 확률 추론
# ==============================
print("\n[3] 전체 이미지에 대해 p0/p2 추론 시작...")

df_all_for_ds = df_all.copy()
df_all_for_ds["label"] = 0
df_all_for_ds["weight"] = 1.0

ds_all = ShoeWearDataset(df_all_for_ds, transform=val_transform)
loader_all = DataLoader(ds_all, batch_size=64, shuffle=False,
                        num_workers=cfg.num_workers)

p0_list = []
p2_list = []

with torch.no_grad():
    for imgs, labels, weights in tqdm(loader_all, desc="[3] Inference", ncols=80):
        imgs = imgs.to(device)
        logits = model(imgs)
        probs = torch.softmax(logits, dim=1)  # [B, 2]
        p0_list.extend(probs[:, 0].cpu().numpy().tolist())
        p2_list.extend(probs[:, 1].cpu().numpy().tolist())

df_probs = df_all.copy()
df_probs["p0"] = p0_list
df_probs["p2"] = p2_list

df_probs.to_csv(PSEUDO_BINARY_CSV, index=False, encoding="utf-8-sig")
print("    → p0/p2 추론 완료 및 CSV 저장.")
print("    예시 5개:")
print(df_probs.head())

# ==============================
# 4) p0/p2 기반 pseudo 0/1/2 생성
# ==============================
print("\n[4] p0/p2 기반 pseudo-label 생성 중...")

P0_STRICT = 0.92      # p0 ≥ 0.92 
P2_STRICT = 0.92      # p2 ≥ 0.92 
MARGIN_STRICT = 0.20  # 서로 차이가 0.20 이상

# class 1 (중간)
MID_LOW = 0.45
MID_HIGH = 0.55
MAX_CONF_FOR_MID = 0.65  # p0,p2 중 큰 값이 0.65 미만이면 1로

print(f"    - 기준 설정:")
print(f"      * P0_STRICT = {P0_STRICT}, P2_STRICT = {P2_STRICT}, MARGIN_STRICT = {MARGIN_STRICT}")
print(f"      * MID_LOW = {MID_LOW}, MID_HIGH = {MID_HIGH}, MAX_CONF_FOR_MID = {MAX_CONF_FOR_MID}")

df_pseudo = df_probs.copy()
df_pseudo["pseudo_label"] = -1
df_pseudo["source"] = "pseudo_from_binary"
df_pseudo["weight"] = cfg.w_pseudo_mid  # 기본 pseudo weight

# --- 확실한 0 ---
cond_0 = (df_pseudo["p0"] >= P0_STRICT) & \
         ((df_pseudo["p0"] - df_pseudo["p2"]) >= MARGIN_STRICT)
df_pseudo.loc[cond_0, "pseudo_label"] = 0
df_pseudo.loc[cond_0, "weight"] = cfg.w_pseudo_core

# --- 확실한 2 ---
cond_2 = (df_pseudo["p2"] >= P2_STRICT) & \
         ((df_pseudo["p2"] - df_pseudo["p0"]) >= MARGIN_STRICT)
df_pseudo.loc[cond_2, "pseudo_label"] = 2
df_pseudo.loc[cond_2, "weight"] = cfg.w_pseudo_core

# --- 중간 1 후보 ---
cond_1 = (df_pseudo["pseudo_label"] == -1) & \
         (df_pseudo["p0"] >= MID_LOW) & (df_pseudo["p0"] <= MID_HIGH) & \
         (df_pseudo[["p0", "p2"]].max(axis=1) < MAX_CONF_FOR_MID)

df_pseudo.loc[cond_1, "pseudo_label"] = 1
df_pseudo.loc[cond_1, "weight"] = cfg.w_pseudo_mid

df_pseudo_labeled = df_pseudo[df_pseudo["pseudo_label"] != -1].copy()

print("    - Pseudo 라벨링 결과 (seed 포함 전):")
pseudo_counts_before = df_pseudo_labeled["pseudo_label"].value_counts()
print(pseudo_counts_before)
print("    - 비율(%):")
print((pseudo_counts_before / len(df_pseudo_labeled) * 100).round(2))

# ==============================
# 5) seed에 있는 이미지는 pseudo에서 제외
# ==============================
print("\n[5] seed 이미지 제외 중...")

mask_not_seed = ~df_pseudo_labeled["image_path"].isin(seed_paths_set)
df_pseudo_clean = df_pseudo_labeled[mask_not_seed].copy()

print("    - Seed 이미지 제외 후 pseudo 라벨 개수:")
pseudo_counts_after = df_pseudo_clean["pseudo_label"].value_counts()
print(pseudo_counts_after)
print("    - 비율(%):")
print((pseudo_counts_after / len(df_pseudo_clean) * 100).round(2))

df_pseudo_clean = df_pseudo_clean.rename(columns={"pseudo_label": "label"})
df_pseudo_clean = df_pseudo_clean[["image_path", "label", "weight", "source"]]

# ==============================
# 6) seed + pseudo 병합 → Stage2 학습용 CSV 생성
# ==============================
print("\n[6] seed + pseudo 병합 중...")

df_seed_stage = df_seed.copy()
df_seed_stage["weight"] = cfg.w_seed
df_seed_stage["source"] = "seed"
df_seed_stage = df_seed_stage[["image_path", "label", "weight", "source"]]

merged = pd.concat([df_seed_stage, df_pseudo_clean], ignore_index=True)

merged_counts = merged["label"].value_counts()
print("    - 병합 후 전체 학습 데이터 개수:", len(merged))
print("    - 병합 후 라벨 분포:")
print(merged_counts)
print("    - 병합 후 라벨 비율(%):")
print((merged_counts / len(merged) * 100).round(2))

merged.to_csv(MERGED_STAGE2_CSV, index=False, encoding="utf-8-sig")
print(f"\n✅ Stage2 학습용 merged CSV 저장 완료: {MERGED_STAGE2_CSV}")

# ==============================
# 7) 로그 기록 (CSV)
# ==============================
append_log_csv(
    LOG_CSV,
    {
        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "total_images": len(df_all),
        "seed_count": len(df_seed),

        "pseudo_0_before_seed": int(pseudo_counts_before.get(0, 0)),
        "pseudo_1_before_seed": int(pseudo_counts_before.get(1, 0)),
        "pseudo_2_before_seed": int(pseudo_counts_before.get(2, 0)),

        "pseudo_0_after_seed": int(pseudo_counts_after.get(0, 0)),
        "pseudo_1_after_seed": int(pseudo_counts_after.get(1, 0)),
        "pseudo_2_after_seed": int(pseudo_counts_after.get(2, 0)),

        "merged_total": int(len(merged)),
        "merged_0": int(merged_counts.get(0, 0)),
        "merged_1": int(merged_counts.get(1, 0)),
        "merged_2": int(merged_counts.get(2, 0)),

        "P0_STRICT": P0_STRICT,
        "P2_STRICT": P2_STRICT,
        "MARGIN_STRICT": MARGIN_STRICT,
        "MID_LOW": MID_LOW,
        "MID_HIGH": MID_HIGH,
        "MAX_CONF_FOR_MID": MAX_CONF_FOR_MID,
    },
    log_fields
)

print(f"\n📝 로그 기록 완료: {LOG_CSV}")
print("\n🎯 step05_generate_pseudo_binary 완료.")

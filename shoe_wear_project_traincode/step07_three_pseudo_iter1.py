##### step07_generate_pseudo_three.py (modified)

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

THREE_STAGE1_CKPT = cfg.three_ckpt_tmpl.format(1)

ALL_IMAGES_CSV = cfg.all_images_csv
SEED_LABEL_CSV = cfg.seed_label_csv

PSEUDO_THREE_CSV = os.path.join(cfg.annot_dir, "pseudo_three_stage1_probs.csv")
MERGED_STAGE3_CSV = os.path.join(cfg.annot_dir, "merged_train_stage3.csv")

LOG_CSV = os.path.join(cfg.log_dir, "pseudo_three_stage1_log.csv")
log_fields = [
    "time",
    "pseudo_0",
    "pseudo_1",
    "pseudo_2",
    "pseudo_0_after",
    "pseudo_1_after",
    "pseudo_2_after",
    "merged_total",
    "merged_0",
    "merged_1",
    "merged_2",
]


# ================================================================
# 2) 데이터 불러오기
# ================================================================
df_all = pd.read_csv(ALL_IMAGES_CSV)
df_seed = pd.read_csv(SEED_LABEL_CSV)

seed_paths_set = set(df_seed["image_path"].tolist())

print(f"\n[1] 전체 이미지: {len(df_all)}")
print(f"[1] Seed 이미지: {len(df_seed)}")


# ================================================================
# 3) 모델 로드
# ================================================================
if not os.path.isfile(THREE_STAGE1_CKPT):
    raise FileNotFoundError(f"3-Class Stage1 모델이 없습니다: {THREE_STAGE1_CKPT}")

model = build_three_class_model().to(device)
state = torch.load(THREE_STAGE1_CKPT, map_location=device)
model.load_state_dict(state)
model.eval()

print("[2] 3-Class Stage1 모델 로드 완료.")


# ================================================================
# 4) 전체 이미지 확률 추론 (p0, p1, p2)
# ================================================================
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

print("\n[3] 전체 이미지 p0/p1/p2 추론 중...")
with torch.no_grad():
    for imgs, labels, weights in tqdm(loader_all, desc="[3] Inference", ncols=80):
        imgs = imgs.to(device)
        logits = model(imgs)
        probs = torch.softmax(logits, dim=1)

        p0_list.extend(probs[:, 0].cpu().numpy().tolist())
        p1_list.extend(probs[:, 1].cpu().numpy().tolist())
        p2_list.extend(probs[:, 2].cpu().numpy().tolist())

df_probs = df_all.copy()
df_probs["p0"] = p0_list
df_probs["p1"] = p1_list
df_probs["p2"] = p2_list

df_probs.to_csv(PSEUDO_THREE_CSV, index=False, encoding="utf-8-sig")
print(f"[3] → 확률 저장 완료: {PSEUDO_THREE_CSV}")


# ================================================================
# 5) Pseudo 기준 적용
#    - 0/2: 꽤 확신 있는 애들
#    - 1: 애매한 애들 (이전보다 완화, p1이 1등 아니어도 허용)
# ================================================================

# --- 0/2 기준 (0은 더 깐깐하게, 2는 기존 수준 유지) --- #
P0_STRICT = 0.95      # 0 되려면 진짜 확신 있어야 함
P2_STRICT = 0.85      # 2는 기존 정도 유지
MARGIN_02 = 0.35      # 다른 두 클래스보다 0.20 이상 커야 0/2로 인정

# --- 1 기준 (애매한 애들까지 포함, p1이 꼭 max가 아니어도 됨) --- #
P1_MIN = 0.15              # p1 ≥ 0.15 이상이면 1 후보
P1_MAX_OTHERS = 0.90       # p0,p2 중 가장 큰 값이 0.90 이하면 OK
P1_MIN_OTHERS = 0.00       # p0,p2가 0이어도 허용 (더 완화)
P1_MAX_DIFF_02 = 0.55      # |p0 - p2| ≤ 0.55
P1_MAX_GAP_BELOW = 0.40    # max(p0,p2)보다 최대 0.20 작을 때까지 1 허용

print("\n[4] Pseudo 라벨 기준 하이퍼파라미터:")
print(f"    - P0_STRICT={P0_STRICT}, P2_STRICT={P2_STRICT}, MARGIN_02={MARGIN_02}")
print(f"    - P1_MIN={P1_MIN}")
print(f"    - P1_MAX_OTHERS={P1_MAX_OTHERS}, P1_MIN_OTHERS={P1_MIN_OTHERS}")
print(f"    - P1_MAX_DIFF_02={P1_MAX_DIFF_02}, P1_MAX_GAP_BELOW={P1_MAX_GAP_BELOW}")

df_pseudo = df_probs.copy()
df_pseudo["pseudo_label"] = -1
df_pseudo["weight"] = cfg.w_pseudo_mid
df_pseudo["source"] = "pseudo_from_three"

# ------ 확실한 0 ------ #
max_1_2 = df_pseudo[["p1", "p2"]].max(axis=1)
cond0 = (df_pseudo["p0"] >= P0_STRICT) & ((df_pseudo["p0"] - max_1_2) >= MARGIN_02)
df_pseudo.loc[cond0, "pseudo_label"] = 0
df_pseudo.loc[cond0, "weight"] = cfg.w_pseudo_core

# ------ 확실한 2 ------ #
max_0_1 = df_pseudo[["p0", "p1"]].max(axis=1)
cond2 = (df_pseudo["p2"] >= P2_STRICT) & ((df_pseudo["p2"] - max_0_1) >= MARGIN_02)
df_pseudo.loc[cond2, "pseudo_label"] = 2
df_pseudo.loc[cond2, "weight"] = cfg.w_pseudo_core

# ------ 중간 1 (완화된 새 기준: p1이 1등 아니어도 됨) ------ #
max_02 = df_pseudo[["p0", "p2"]].max(axis=1)
min_02 = df_pseudo[["p0", "p2"]].min(axis=1)
diff_02 = (df_pseudo["p0"] - df_pseudo["p2"]).abs()

cond1 = (
    (df_pseudo["pseudo_label"] == -1) &                # 아직 0/2로 안 뽑힌 애들
    (df_pseudo["p1"] >= P1_MIN) &                      # p1이 어느 정도 이상
    ((max_02 - df_pseudo["p1"]) <= P1_MAX_GAP_BELOW) & # ⭐ p1이 max(p0,p2)보다 너무 많이 작지 않으면 OK
    (max_02 <= P1_MAX_OTHERS) &                        # 0/2 쪽에서 너무 강한 확률이 없고
    (min_02 >= P1_MIN_OTHERS) &                        # 0/2 중 하나가 완전 0이어도 허용 (완화)
    (diff_02 <= P1_MAX_DIFF_02)                        # 0/2 간 차이가 너무 극단적이지 않음
)

df_pseudo.loc[cond1, "pseudo_label"] = 1
df_pseudo.loc[cond1, "weight"] = cfg.w_pseudo_mid

df_pseudo_labeled = df_pseudo[df_pseudo["pseudo_label"] != -1].copy()

print("\n[4] Pseudo 라벨링 결과 (Seed 제외 전):")
print(df_pseudo_labeled["pseudo_label"].value_counts())
before_counts = df_pseudo_labeled["pseudo_label"].value_counts()


# ================================================================
# 6) Seed 제외
# ================================================================
mask = ~df_pseudo_labeled["image_path"].isin(seed_paths_set)
df_pseudo_clean = df_pseudo_labeled[mask].copy()

print("\n[5] Seed 제외 후 Pseudo 라벨 분포:")
print(df_pseudo_clean["pseudo_label"].value_counts())
after_counts = df_pseudo_clean["pseudo_label"].value_counts()


# ================================================================
# 7) Seed + Pseudo 병합
# ================================================================
df_seed_stage = df_seed.copy()
df_seed_stage["weight"] = cfg.w_seed
df_seed_stage["source"] = "seed"

df_seed_stage = df_seed_stage[["image_path", "label", "weight", "source"]]
df_pseudo_final = df_pseudo_clean.rename(columns={"pseudo_label": "label"})
df_pseudo_final = df_pseudo_final[["image_path", "label", "weight", "source"]]

merged = pd.concat([df_seed_stage, df_pseudo_final], ignore_index=True)

print("\n[6] Seed + Pseudo 병합 후 라벨 분포:")
print(merged["label"].value_counts())

merged_counts = merged["label"].value_counts()

merged.to_csv(MERGED_STAGE3_CSV, index=False, encoding="utf-8-sig")
print(f"\n→ merged_train_stage3.csv 저장 완료: {MERGED_STAGE3_CSV}")


# ================================================================
# 8) 로그 기록
# ================================================================
append_log_csv(
    LOG_CSV,
    {
        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "pseudo_0": int(before_counts.get(0, 0)),
        "pseudo_1": int(before_counts.get(1, 0)),
        "pseudo_2": int(before_counts.get(2, 0)),
        "pseudo_0_after": int(after_counts.get(0, 0)),
        "pseudo_1_after": int(after_counts.get(1, 0)),
        "pseudo_2_after": int(after_counts.get(2, 0)),
        "merged_total": int(len(merged)),
        "merged_0": int(merged_counts.get(0, 0)),
        "merged_1": int(merged_counts.get(1, 0)),
        "merged_2": int(merged_counts.get(2, 0)),
    },
    log_fields
)

print("\n[7] 로그 기록 완료.")
print("✅ Step07 완료.")

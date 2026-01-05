##### step01_check_paths.py

import os
from glob import glob
import pandas as pd

# =========================
# 1) 프로젝트 루트 경로 설정
#    👉 네가 실제로 만든 폴더 경로로 바꿔줘
# =========================
PROJECT_ROOT = r"C:\Users\holyb\OneDrive\Desktop\shoe_wear_project"  # D:나 다른 경로면 여기만 수정

IMAGE_DIR = os.path.join(PROJECT_ROOT, "data", "images", "train")
ANNOT_DIR = os.path.join(PROJECT_ROOT, "data", "annotations")
SEED_LABEL_CSV = os.path.join(ANNOT_DIR, "labels_seed.csv")
OUTPUT_MODEL_DIR = os.path.join(PROJECT_ROOT, "outputs", "models")

# 폴더 없으면 생성 (models만)
os.makedirs(OUTPUT_MODEL_DIR, exist_ok=True)

print("PROJECT_ROOT:", PROJECT_ROOT)
print("IMAGE_DIR:", IMAGE_DIR)
print("ANNOT_DIR:", ANNOT_DIR)
print("OUTPUT_MODEL_DIR:", OUTPUT_MODEL_DIR)

# =========================
# 2) 이미지 개수 확인
# =========================
img_paths = sorted(
    glob(os.path.join(IMAGE_DIR, "*.jpg")) +
    glob(os.path.join(IMAGE_DIR, "*.jpeg")) +
    glob(os.path.join(IMAGE_DIR, "*.png"))
)

print(f"\n이미지 개수: {len(img_paths)}")
if len(img_paths) == 0:
    print("⚠ IMAGE_DIR 안에 이미지가 하나도 없습니다. 경로/위치를 다시 확인하세요.")

# =========================
# 3) labels_seed.csv 확인
# =========================
if not os.path.isfile(SEED_LABEL_CSV):
    print(f"\n❌ labels_seed.csv 파일이 없습니다: {SEED_LABEL_CSV}")
else:
    print(f"\n✅ labels_seed.csv 발견: {SEED_LABEL_CSV}")
    df_seed = pd.read_csv(SEED_LABEL_CSV)
    print("labels_seed.csv 헤더:", list(df_seed.columns))
    print("앞부분 3줄:")
    print(df_seed.head(3))

##### step02_prepare_csvs.py

import os
from glob import glob
import pandas as pd

# =========================
# 1) 프로젝트 루트 / 경로 설정
# =========================
PROJECT_ROOT = r"C:\Users\holyb\OneDrive\Desktop\shoe_wear_project"

IMAGE_DIR = os.path.join(PROJECT_ROOT, "data", "images", "train")
ANNOT_DIR = os.path.join(PROJECT_ROOT, "data", "annotations")

SEED_LABEL_CSV = os.path.join(ANNOT_DIR, "labels_seed.csv")
ALL_IMAGES_CSV = os.path.join(ANNOT_DIR, "all_images.csv")

print("=== 경로 확인 ===")
print("IMAGE_DIR:", IMAGE_DIR)
print("ANNOT_DIR:", ANNOT_DIR)
print("SEED_LABEL_CSV:", SEED_LABEL_CSV)
print("ALL_IMAGES_CSV:", ALL_IMAGES_CSV)

# ----------------------------------------------------
# 2) 전체 이미지 리스트 수집 → all_images.csv 저장
# ----------------------------------------------------
img_paths = sorted(
    glob(os.path.join(IMAGE_DIR, "*.jpg")) +
    glob(os.path.join(IMAGE_DIR, "*.jpeg")) +
    glob(os.path.join(IMAGE_DIR, "*.png"))
)

print(f"\n[1] IMAGE_DIR 내 이미지 개수: {len(img_paths)}")
if len(img_paths) == 0:
    raise RuntimeError("IMAGE_DIR 안에 이미지가 없습니다. 경로를 다시 확인하세요.")

df_all = pd.DataFrame({"image_path": img_paths})
df_all.to_csv(ALL_IMAGES_CSV, index=False, encoding="utf-8-sig")
print(f"→ all_images.csv 저장 완료: {ALL_IMAGES_CSV}")

# ----------------------------------------------------
# 3) labels_seed.csv 불러와서 경로 정리
#    - 예전 Colab 경로가 들어있어도 상관 없음
#    - 파일명만 떼어내서 현재 IMAGE_DIR 기준으로 다시 연결
# ----------------------------------------------------
if not os.path.isfile(SEED_LABEL_CSV):
    raise FileNotFoundError(f"labels_seed.csv 를 찾을 수 없습니다: {SEED_LABEL_CSV}")

df_seed = pd.read_csv(SEED_LABEL_CSV)
print("\n[2] labels_seed.csv 읽기 완료")
print("  컬럼:", list(df_seed.columns))
print("  상위 3개 행:")
print(df_seed.head(3))

# --- 컬럼 이름 정리 ---
# image_path, filename 둘 중 하나만 있어도 되게 처리
if "image_path" in df_seed.columns:
    df_seed["filename"] = df_seed["image_path"].apply(lambda x: os.path.basename(str(x)))
elif "filename" in df_seed.columns:
    df_seed["filename"] = df_seed["filename"].astype(str)
else:
    raise ValueError("labels_seed.csv 에 'image_path' 또는 'filename' 컬럼이 필요합니다.")

# label 컬럼 확인
if "label" not in df_seed.columns:
    raise ValueError("labels_seed.csv 에 'label' 컬럼이 필요합니다.")

# 현재 IMAGE_DIR 기준으로 새 image_path 구성
df_seed["image_path"] = df_seed["filename"].apply(lambda name: os.path.join(IMAGE_DIR, name))

# 실제 파일 존재 여부 체크
df_seed["exists"] = df_seed["image_path"].apply(os.path.isfile)
missing = df_seed[~df_seed["exists"]]

print(f"\n[3] seed 라벨 개수: {len(df_seed)}")
print("  라벨 분포:")
print(df_seed["label"].value_counts(dropna=False))

if len(missing) > 0:
    print(f"\n⚠ 경고: seed CSV에는 있는데 IMAGE_DIR 에 없는 이미지 {len(missing)}개 존재")
    print("  예시 몇 개:")
    print(missing[["filename", "image_path"]].head(5))
else:
    print("\n✅ 모든 seed 라벨 이미지가 IMAGE_DIR 에 존재합니다.")

# exists == True 인 것만 남겨서 깨끗하게 다시 저장
df_seed_clean = df_seed[df_seed["exists"]].copy()

# 보조 컬럼 제거 후 정리
df_seed_clean = df_seed_clean[["image_path", "label"]]

df_seed_clean.to_csv(SEED_LABEL_CSV, index=False, encoding="utf-8-sig")
print(f"\n[4] 경로 정리된 labels_seed.csv 저장 완료: {SEED_LABEL_CSV}")
print("  정리 후 라벨 분포:")
print(df_seed_clean["label"].value_counts(dropna=False))

print("\n✅ step02_prepare_csvs.py 완료")

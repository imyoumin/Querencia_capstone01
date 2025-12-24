# grounding_sam_pipeline.py

import numpy as np
import torch
from PIL import Image

from transformers import (
    AutoProcessor,
    AutoModelForZeroShotObjectDetection,
    SamProcessor,
    SamModel,
    infer_device,
)

# =========================
# 기본 설정
# =========================

DEVICE = infer_device()

# 👉 전시장 PC에서 완전 오프라인으로 쓰고 싶으면
#    아래 두 줄을 "models/grounding_dino_tiny", "models/sam_vit_base"
#    처럼 로컬 폴더 경로로 바꿔도 됨.
GROUNDING_DINO_ID = "IDEA-Research/grounding-dino-tiny"  # 또는 "models/grounding_dino_tiny"
SAM_ID = "facebook/sam-vit-base"                         # 또는 "models/sam_vit_base"

# 배경 회색(중간 회색)
BG_VALUE = 128  # 0=검정, 255=흰색, 128=중간 회색

# 모델을 전역으로 한 번만 로드하기 위한 캐시
_gd_processor = None
_gd_model = None
_sam_processor = None
_sam_model = None


# =========================
# 모델 로드 함수
# =========================

def _load_models():
    """Grounding DINO / SAM 모델을 전역으로 한 번만 로드."""
    global _gd_processor, _gd_model, _sam_processor, _sam_model

    if _gd_processor is None or _gd_model is None:
        print("[grounding_sam_pipeline] Loading Grounding DINO from:", GROUNDING_DINO_ID)
        _gd_processor = AutoProcessor.from_pretrained(GROUNDING_DINO_ID)
        _gd_model = AutoModelForZeroShotObjectDetection.from_pretrained(
            GROUNDING_DINO_ID
        ).to(DEVICE)

    if _sam_processor is None or _sam_model is None:
        print("[grounding_sam_pipeline] Loading SAM from:", SAM_ID)
        _sam_processor = SamProcessor.from_pretrained(SAM_ID)
        _sam_model = SamModel.from_pretrained(SAM_ID).to(DEVICE)


# =========================
# 메인 파이프라인 함수
# =========================

def run_grounding_sam(image: Image.Image):
    """
    1. Grounding DINO로 'a pair of shoes' 박스 검출
    2. 가장 score 높은 박스 1개를 선택
    3. 해당 박스를 SAM에 입력해 마스크 생성
    4. 마스크 바깥은 중간 회색(BG_VALUE)으로 채움
    5. 신발 영역만 crop (resize는 하지 않음)

    반환 형식 예:
    {
        "success": True/False,
        "reason": None 또는 실패 이유 문자열,
        "crop": PIL.Image (crop된 RGB 이미지, 배경은 회색),
        "boxes": [[x0, y0, x1, y1], ...],
        "scores": [float, ...],
        "labels": ["a pair of shoes", ...],
    }
    """
    _load_models()

    # 혹시 모를 grayscale 등을 대비해 항상 RGB로 변환
    if image.mode != "RGB":
        image = image.convert("RGB")

    np_image = np.array(image)  # (H, W, 3)
    height, width = np_image.shape[:2]

    # =========================
    # 1) Grounding DINO로 박스 검출
    # =========================
    text_labels = [["a pair of shoes"]]

    inputs = _gd_processor(
        images=image,
        text=text_labels,
        return_tensors="pt",
    ).to(DEVICE)

    with torch.no_grad():
        outputs = _gd_model(**inputs)

    results = _gd_processor.post_process_grounded_object_detection(
        outputs=outputs,
        input_ids=inputs.input_ids,
        threshold=0.35,
        text_threshold=0.25,
        target_sizes=[(height, width)],  # (H, W)
    )

    result = results[0]
    boxes = result["boxes"]   # (N, 4)
    scores = result["scores"] # (N,)
    labels = result["labels"] # (N,)

    if len(boxes) == 0:
        return {
            "success": False,
            "reason": "no_shoe_detected",
            "crop": None,
            "boxes": [],
            "scores": [],
            "labels": [],
        }

    # 가장 score가 높은 박스 선택
    best_idx = int(torch.argmax(scores).item())
    best_box = boxes[best_idx]
    best_score = float(scores[best_idx].item())
    best_label = labels[best_idx]

    # =========================
    # 2) SAM 입력 준비
    # =========================
    box_list = best_box.tolist()  # [x0, y0, x1, y1]

    sam_inputs = _sam_processor(
        image,
        input_boxes=[[box_list]],  # [[box]] 형태
        return_tensors="pt",
    ).to(DEVICE)

    with torch.no_grad():
        sam_outputs = _sam_model(**sam_inputs)

    masks = _sam_processor.post_process_masks(
        sam_outputs.pred_masks,
        sam_inputs["original_sizes"],
        sam_inputs["reshaped_input_sizes"],
    )[0]  # (num_boxes, 1, H, W) 또는 (num_boxes, H, W) 형태

    # 하나의 박스만 넣었으니 0번째 사용
    masks_box = masks[0]  # 보통 (1, H, W) 또는 (H, W)

    if masks_box.ndim == 3:
        # (1, H, W) -> (H, W)
        masks_box = masks_box[0]

    # 여러 마스크 채널이 있을 경우 max로 합치기 (안전용)
    if masks_box.ndim == 3:
        mask_2d = masks_box.max(dim=0).values
    else:
        mask_2d = masks_box

    mask_bin = (mask_2d > 0.5).cpu().numpy()  # (H, W), bool

    ys, xs = np.where(mask_bin)
    if len(ys) == 0 or len(xs) == 0:
        return {
            "success": False,
            "reason": "empty_mask",
            "crop": None,
            "boxes": [best_box.tolist()],
            "scores": [best_score],
            "labels": [best_label],
        }

    # =========================
    # 3) 회색 배경으로 합성 + crop
    # =========================
    y_min, y_max = ys.min(), ys.max()
    x_min, x_max = xs.min(), xs.max()

    # 회색 배경 (H, W, 3)
    bg = np.full_like(np_image, fill_value=BG_VALUE, dtype=np.uint8)

    # 마스크를 3채널로 확장
    mask_3 = mask_bin.astype(np.uint8)[..., None]  # (H, W, 1)

    # foreground(신발)만 원본 유지, 나머지는 회색으로 채우기
    composited = np.where(mask_3 == 1, np_image, bg)  # (H, W, 3)

    # 신발 영역만 crop
    crop_rgb = composited[y_min:y_max + 1, x_min:x_max + 1, :]  # (h, w, 3)
    crop_img = Image.fromarray(crop_rgb, mode="RGB")

    # =========================
    # 4) 결과 반환 (resize 안 함)
    # =========================
    return {
        "success": True,
        "reason": None,
        "crop": crop_img,                  # 신발만 남긴 crop 이미지 (배경 회색, 사이즈는 가변)
        "boxes": [best_box.tolist()],      # 선택된 박스 1개
        "scores": [best_score],
        "labels": [best_label],
    }

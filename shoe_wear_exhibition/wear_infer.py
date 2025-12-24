# wear_infer.py
#
# Grounding+SAM에서 잘라낸 신발 crop 이미지를 받아서
# 최종 신발 마모도 모델(WearNet)으로 0/1/2 각 클래스 확률을 계산하는 모듈.
#
# 반환값:
#   - pred_label: int   (0, 1, 2 중 하나)
#   - pred_score: float (pred_label에 해당하는 확률 값, 0.0 ~ 1.0)
#   - probs_list: [p0, p1, p2]  (각 클래스별 확률)
#
# 사용 예:
#   from wear_infer import predict_wear
#   label_idx, score, probs = predict_wear(crop_img)

import os
import torch
import torch.nn.functional as F
from PIL import Image

from step03_dataset_and_model import (
    build_wear_model,
    val_transform,
)


# ---------------------------------
# 1) 디바이스 및 모델 로드
# ---------------------------------

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# weight 파일 위치
# → wearnet_final.pt를 "models/" 폴더 안에 두는 경우
WEIGHT_PATH = os.path.join("models", "wearnet_final.pt")

# 최종 신발 마모도 모델 (ShoeWearNet, 3-class)
_model = build_wear_model().to(DEVICE)

# 학습된 가중치 로드
state = torch.load(WEIGHT_PATH, map_location=DEVICE)
_model.load_state_dict(state)
_model.eval()


# ---------------------------------
# 2) 추론 함수
# ---------------------------------

def predict_wear(img: Image.Image, temperature: float = 2.5):
    """
    Grounding+SAM에서 crop된 신발 이미지를 받아서
    (pred_label, pred_score, probs_list)를 반환한다.

    temperature > 1 으로 주면 확률이 덜 극단적으로 퍼진다.
    """
    if img.mode != "RGB":
        img = img.convert("RGB")

    x = val_transform(img)
    x = x.unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = _model(x)            # (1, 3)

        # 🔴 여기: temperature softmax
        if temperature is not None and temperature > 0:
            scaled_logits = logits / temperature
        else:
            scaled_logits = logits

        probs = F.softmax(scaled_logits, dim=1)[0]  # (3,)

    pred_score, pred_label = torch.max(probs, dim=0)
    probs_list = probs.cpu().tolist()

    return int(pred_label.item()), float(pred_score.item()), probs_list

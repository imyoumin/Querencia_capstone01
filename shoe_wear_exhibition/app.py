# app.py

import os
import io
from datetime import datetime

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from PIL import Image

import numpy as np
from rembg import remove  # 배경 제거용

from grounding_sam_pipeline import run_grounding_sam
from wear_infer import predict_wear  # ← WearNet 추론 함수


# =========================
# 기본 설정
# =========================

app = FastAPI()

# static 폴더를 /static 경로에 마운트 (index.html, js, css 등)
app.mount("/static", StaticFiles(directory="static"), name="static")

# 기록용 이미지 저장 폴더
SAVE_DIR = "captures"
os.makedirs(SAVE_DIR, exist_ok=True)

# ✅ 전역 상태 변수
AR_UNLOCKED = False        # AR 가림막 해제 여부 (False=가림, True=보임)
CURRENT_SHOE = "Red"     # 현재 AR Wrapper가 보여줘야 할 신발 (기본값: Red)

# ✅ 신발별 대표 색상 (RGB) 테이블 (6가지 색상 정의)
# identify_shoe 호출 시 이 값들과 비교하여 가장 가까운 신발을 찾습니다.
SHOE_COLOR_TABLE = {
    "Red":       np.array([200, 40, 40]),    # 빨강
    "Green":     np.array([40, 180, 40]),    # 초록
    "Blue":      np.array([40, 40, 200]),    # 파랑 (밝은 파랑)
    "Navy":      np.array([20, 20, 100]),    # 네이비 (어두운 파랑)
    "Aluminium": np.array([200, 200, 200]),  # 알루미늄 (밝은 회색/은색)
    "Black":     np.array([30, 30, 30]),     # 검정 (완전 0은 아니게)
}

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# =========================
# 헬퍼 함수들 (색상 추출 / 매칭)
# =========================

def extract_foreground_mean_rgb(pil_rgba: Image.Image):
    """
    RGBA 이미지에서 alpha>0인 전경(신발) 픽셀만 골라 평균 RGB를 계산.
    """
    arr = np.array(pil_rgba)

    if arr.ndim == 3 and arr.shape[2] == 4:
        rgb = arr[..., :3]
        alpha = arr[..., 3]
        mask = alpha > 0

        if not np.any(mask):
            rgb_pixels = rgb.reshape(-1, 3)
        else:
            rgb_pixels = rgb[mask]
    else:
        rgb_pixels = arr.reshape(-1, 3)

    mean_rgb = rgb_pixels.mean(axis=0)
    return mean_rgb

def find_closest_shoe(mean_rgb: np.ndarray):
    """
    SHOE_COLOR_TABLE에서 가장 가까운 dist 키를 반환.
    """
    best_key = None
    best_dist = float("inf")

    for key, ref_rgb in SHOE_COLOR_TABLE.items():
        d = np.linalg.norm(mean_rgb - ref_rgb)
        if d < best_dist:
            best_dist = d
            best_key = key

    return best_key, best_dist


# =========================
# 1) HTML 페이지 라우팅
# =========================

@app.get("/", response_class=HTMLResponse)
def read_root():
    with open("static/intro.html", "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())

@app.get("/analysis", response_class=HTMLResponse)
def read_analysis():
    with open("static/analysis.html", "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())

@app.get("/result", response_class=HTMLResponse)
def read_result():
    with open("static/result.html", "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())

@app.get("/recommend", response_class=HTMLResponse)
def read_recommend():
    with open("static/recommend.html", "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())


# =========================
# 2) AR 상태 관리 & 리모컨 기능
# =========================

@app.post("/toggle_ar")
def toggle_ar():
    """
    [웹] 'AR 신어보기' 버튼 클릭 시 호출
    - 잠금 상태를 토글 (보통 False -> True로 켬)
    """
    global AR_UNLOCKED
    AR_UNLOCKED = not AR_UNLOCKED
    print(f"[app] AR_UNLOCKED toggled -> {AR_UNLOCKED}")
    return {"success": True, "unlocked": AR_UNLOCKED}

@app.post("/lock_ar")
def lock_ar():
    """
    [웹] '다시 측정하기' 등 이탈 시 호출
    - 강제로 잠금 상태(False)로 변경하여 가림막을 내림
    """
    global AR_UNLOCKED
    AR_UNLOCKED = False
    print(f"[app] AR_UNLOCKED locked -> {AR_UNLOCKED}")
    return {"success": True, "unlocked": AR_UNLOCKED}

@app.get("/ar_status")
def ar_status():
    """
    [Wrapper] 0.5초마다 호출하여 잠금 상태 확인
    """
    return {"unlocked": AR_UNLOCKED}

@app.post("/set_shoe/{shoe_key}")
def set_shoe(shoe_key: str):
    """
    [웹] 결과 화면에서 화살표로 신발을 바꿀 때 호출
    - 현재 보여줄 신발 키(Red, Green 등)를 업데이트
    """
    global CURRENT_SHOE
    CURRENT_SHOE = shoe_key
    print(f"[app] Shoe changed to -> {CURRENT_SHOE}")
    return {"success": True, "current": CURRENT_SHOE}

@app.get("/get_shoe")
def get_shoe():
    """
    [Wrapper] 0.5초마다 호출하여 현재 보여줘야 할 신발 확인
    """
    return {"shoe": CURRENT_SHOE}

@app.post("/reset_ar")
def reset_ar():
    """
    [웹] 처음으로 돌아가거나 완전 초기화 시 호출
    - 잠금 설정 + 신발 초기값(Red)으로 리셋
    """
    global AR_UNLOCKED, CURRENT_SHOE
    AR_UNLOCKED = False
    CURRENT_SHOE = "Red"  # 초기값 수정됨
    print("[app] AR System Reset")
    return {"success": True}


# =========================
# 3) 예측 및 분석 엔드포인트
# =========================

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    [웹] 마모도 분석 요청
    """
    # 0) 이미지 읽기
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")

    # 1) 원본 저장
    orig_filename = f"{timestamp}_original.jpg"
    orig_path = os.path.join(SAVE_DIR, orig_filename)
    image.save(orig_path)
    print(f"[app] saved original image: {orig_path}")

    # 2) 전처리
    result = run_grounding_sam(image)

    if not result["success"]:
        reason = result.get("reason", "unknown")
        print(f"[app] preprocessing failed: {reason}")
        return {
            "success": False,
            "wear_level": "pending",
            "score": 0.0,
            "probs": None,
            "message": f"전처리 실패: {reason}",
            "original_path": orig_filename,
            "crop_path": None,
        }

    # 3) Crop 저장
    crop_img = result["crop"]
    crop_filename = f"{timestamp}_crop.jpg"
    crop_path = os.path.join(SAVE_DIR, crop_filename)
    crop_img.save(crop_path)
    print(f"[app] saved crop image: {crop_path}")

    # 4) WearNet 추론
    label_idx, score, probs = predict_wear(crop_img)

    wear_map = {0: "new", 1: "moderate", 2: "heavy"}
    wear_level = wear_map.get(label_idx, "unknown")

    return {
        "success": True,
        "wear_level": wear_level,
        "score": score,
        "probs": {
            "p0_new": probs[0],
            "p1_moderate": probs[1],
            "p2_heavy": probs[2],
        },
        "message": "분석 성공",
        "original_path": orig_filename,
        "crop_path": crop_filename,
    }


@app.post("/identify_shoe")
async def identify_shoe(file: UploadFile = File(...)):
    """
    [Wrapper/웹] 이미지의 색상을 분석하여 가장 가까운 신발(dist) 찾기
    """
    try:
        image_bytes = await file.read()
        input_image = Image.open(io.BytesIO(image_bytes)).convert("RGBA")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"이미지 로딩 실패: {e}")

    try:
        fg_image = remove(input_image)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"rembg 처리 실패: {e}")

    mean_rgb = extract_foreground_mean_rgb(fg_image)
    mean_rgb_list = [float(x) for x in mean_rgb]

    dist_key, distance = find_closest_shoe(mean_rgb)

    if dist_key is None:
        return {"success": False, "message": "식별 실패"}

    return {
        "success": True,
        "dist_key": dist_key,
        "mean_rgb": mean_rgb_list,
        "distance": float(distance),
        "message": "색상 식별 완료",
    }


# =========================
# 4) 서버 실행
# =========================

if __name__ == "__main__":
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=False)
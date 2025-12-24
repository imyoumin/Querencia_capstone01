// static/js/app.js

const video = document.getElementById("video");
const canvas = document.getElementById("canvas");
const captureBtn = document.getElementById("captureBtn");
const resultDiv = document.getElementById("result");
const capturedImage = document.getElementById("capturedImage");
const goToARBtn = document.getElementById("goToARBtn"); // 결과 후에 보여줄 버튼

console.log("goToARBtn:", goToARBtn);  // 콘솔에서 버튼 존재 확인용

// 현재 상태: 실시간 촬영 중인지, 캡처 화면인지
let isCaptured = false;

// 1) 웹캠 시작
// 1) 웹캠 시작  
async function startCamera() {
  try {
    const secondCamId = "f937ce0a1ba8300d9ebba23adc470a65839c6d62a0a5a7e6396cd88390819911";

    const stream = await navigator.mediaDevices.getUserMedia({
      video: { deviceId: { exact: secondCamId } }
    });

    video.srcObject = stream;

    // 초기 상태: 실시간 화면만 보이게
    video.style.display = "block";
    if (capturedImage) {
      capturedImage.style.display = "none";
    }
  } catch (err) {
    console.error("웹캠 사용 불가:", err);
    resultDiv.innerText = "웹캠 권한을 허용해주세요.";
  }
}


// 2) 버튼 클릭 시: (1) 캡처 & 예측 or (2) 다시 촬영 모드로 전환
captureBtn.addEventListener("click", async () => {
  // 영상 로딩 안됐으면 막기
  if (!video.videoWidth || !video.videoHeight) {
    resultDiv.innerText = "영상 로딩 중입니다. 잠시 후 다시 시도해주세요.";
    return;
  }

  // ----------------------------
  // 상태 1: 아직 캡처 안한 상태 → 촬영하기
  // ----------------------------
  if (!isCaptured) {
    // 새 분석 시작할 때는 AR 버튼 숨기기
    if (goToARBtn) {
      goToARBtn.style.display = "none";
      goToARBtn.textContent = "신발을 직접 신어보시겠습니까?";
    }

    // 캔버스 크기를 비디오 크기에 맞춤
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;

    const ctx = canvas.getContext("2d");
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

    // 원본 이미지 dataURL → 동일 카드 안의 <img>에 표시
    const dataURL = canvas.toDataURL("image/jpeg");
    if (capturedImage) {
      capturedImage.src = dataURL;
      capturedImage.style.display = "block";
    }

    // 실시간 영상은 숨김
    video.style.display = "none";

    // 버튼 문구 변경
    captureBtn.textContent = "다시 촬영하기";
    isCaptured = true;

    // 서버로 보낼 blob 생성
    canvas.toBlob(async (blob) => {
      if (!blob) {
        console.error("이미지 blob 변환 실패");
        resultDiv.innerText = "이미지 처리 중 오류가 발생했습니다.";
        return;
      }

      const formData = new FormData();
      formData.append("file", blob, "capture.jpg");

      resultDiv.innerText = "분석 중입니다...";

      try {
        const response = await fetch("/predict", {
          method: "POST",
          body: formData,
        });

        const data = await response.json();
        console.log("predict response:", data);

        if (!data.success) {
          // 전처리 실패 등
          resultDiv.innerText =
            `분석 실패\n` +
            (data.message ? `message: ${data.message}` : "");
          if (goToARBtn) {
            goToARBtn.style.display = "none";
          }
          return;
        }

        // -------------------------
        // 🎨 확률 가로 막대 표시 준비
        // -------------------------
        let p0 = null, p1 = null, p2 = null;
        if (data.probs) {
          p0 = (data.probs.p0_new * 100).toFixed(1);
          p1 = (data.probs.p1_moderate * 100).toFixed(1);
          p2 = (data.probs.p2_heavy * 100).toFixed(1);
        }

        const scorePercent = (data.score * 100).toFixed(1);

        // -------------------------
        // 🎨 HTML 기반 그래프 렌더링
        // -------------------------
        if (p0 !== null) {
          resultDiv.innerHTML = `
            <div class="result-summary">
              <div class="result-main">
                마모도 상태(최종 예측):
                <span class="result-tag">${data.wear_level}</span>
              </div>
              <div class="result-score">
                예측 신뢰도(score): ${scorePercent}%
              </div>
            </div>

            <div class="prob-bars">
              <div class="prob-row">
                <div class="prob-label">0 (새 신발)</div>
                <div class="prob-bar-wrap">
                  <div class="prob-bar prob-bar-0" style="width: ${p0}%"></div>
                </div>
                <div class="prob-value">${p0}%</div>
              </div>

              <div class="prob-row">
                <div class="prob-label">1 (보통 마모)</div>
                <div class="prob-bar-wrap">
                  <div class="prob-bar prob-bar-1" style="width: ${p1}%"></div>
                </div>
                <div class="prob-value">${p1}%</div>
              </div>

              <div class="prob-row">
                <div class="prob-label">2 (심한 마모)</div>
                <div class="prob-bar-wrap">
                  <div class="prob-bar prob-bar-2" style="width: ${p2}%"></div>
                </div>
                <div class="prob-value">${p2}%</div>
              </div>
            </div>

            <div class="result-message">
              ${data.message ? data.message : ""}
            </div>
          `;
        } else {
          // fallback: 확률 없음 → 텍스트 표시
          resultDiv.innerText = `
            마모도 상태(최종 예측): ${data.wear_level}
            예측 신뢰도(score): ${scorePercent}%
            ${data.message ? data.message : ""}
          `;
        }

        // 결과 후 AR 버튼 보이기
        if (goToARBtn) {
          goToARBtn.style.display = "inline-block";
        }
      } catch (err) {
        console.error("서버 요청 실패:", err);
        resultDiv.innerText = "분석 요청 중 오류가 발생했습니다.";
        if (goToARBtn) {
          goToARBtn.style.display = "none";
        }
      }
    }, "image/jpeg");

    return;
  }

  // ----------------------------
  // 상태 2: 이미 캡처된 상태 → 다시 촬영 모드로 전환
  // ----------------------------
  if (isCaptured) {
    video.style.display = "block";

    if (capturedImage) {
      capturedImage.style.display = "none";
    }

    captureBtn.textContent = "촬영하기";
    isCaptured = false;

    return;
  }
});

// 3) AR 버튼 클릭 시: 서버에 토글 신호 보내기
if (goToARBtn) {
  goToARBtn.addEventListener("click", async () => {
    try {
      const res = await fetch("/toggle_ar", {
        method: "POST",
      });
      const data = await res.json();
      console.log("toggle_ar:", data);

      if (data.unlocked) {
        goToARBtn.textContent = "AR 화면 다시 가리기";
      } else {
        goToARBtn.textContent = "신발을 직접 신어보시겠습니까?";
      }
    } catch (err) {
      console.error("toggle_ar 요청 실패:", err);
      alert("AR 상태를 바꾸는 중 오류가 발생했습니다.");j
    }
  });
}

// 초기 실행
startCamera();

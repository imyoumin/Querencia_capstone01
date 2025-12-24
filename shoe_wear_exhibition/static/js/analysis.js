/* static/js/analysis.js */

const video = document.getElementById("video");
const canvas = document.getElementById("canvas");
const captureBtn = document.getElementById("captureBtn");
const loadingOverlay = document.getElementById("loadingOverlay");
const loadingText = document.getElementById("loadingText");
const flash = document.getElementById("flash");

// ✨ 추가된 모달 요소들
const errorModal = document.getElementById("errorModal");
const errorDesc = document.getElementById("errorDesc");
const closeErrorBtn = document.getElementById("closeErrorBtn");

// 1. 카메라 시작
async function startCamera() {
    const CamId = "e268805c923c421d35163520a52b6d53cc5e39d3a4a94dd6e5491054c983c4f9";

    try {
        const stream = await navigator.mediaDevices.getUserMedia({
            video: {
                deviceId: { exact: CamId }   // 특정 웹캠 강제 사용
            }
        });

        video.srcObject = stream;
    } catch (err) {
        console.error(err);
        showErrorModal("특정 카메라에 접근할 수 없습니다.<br>연결 상태를 확인해주세요.");
    }
}




// 2. 로딩 멘트 애니메이션
function animateLoadingText() {
    const messages = [
        "이미지 업로드 중...",
        "배경 제거 및 노이즈 처리...",
        "마모도 패턴 정밀 분석...",
        "결과 보고서 작성 중..."
    ];
    let i = 0;
    loadingText.innerText = messages[0];
    
    return setInterval(() => {
        i = (i + 1) % messages.length;
        loadingText.innerText = messages[i];
    }, 1500); // 1.5초 간격으로 천천히 변경
}

// 3. ✨ 에러 모달 띄우기 함수
function showErrorModal(message) {
    loadingOverlay.style.display = "none"; // 로딩창 끄기
    if (message) errorDesc.innerHTML = message; // 에러 메시지 주입
    errorModal.style.display = "flex"; // 모달 보이기
}

// 4. 에러 모달 닫기 버튼 (다시 시도)
closeErrorBtn.addEventListener("click", () => {
    errorModal.style.display = "none";
    // 필요하다면 여기서 카메라 재설정 등을 할 수 있음
});

// 5. 촬영 버튼 클릭 핸들러
captureBtn.addEventListener("click", () => {
    // 영상이 로드되지 않았으면 무시
    if (!video.videoWidth) return;

    // (1) 플래시 효과
    flash.classList.add("trigger");
    setTimeout(() => flash.classList.remove("trigger"), 200);

    // (2) 캔버스에 현재 프레임 캡처
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    const ctx = canvas.getContext("2d");
    ctx.drawImage(video, 0, 0);

    // (3) 로딩 UI 시작
    loadingOverlay.style.display = "flex";
    const textInterval = animateLoadingText(); // 멘트 롤링 시작

    // (4) 서버 전송
    canvas.toBlob(async (blob) => {
        const formData = new FormData();
        formData.append("file", blob, "capture.jpg");

        try {
            const res = await fetch("/predict", { method: "POST", body: formData });
            const data = await res.json();

            clearInterval(textInterval); // 로딩 멘트 정지

            if (data.success) {
                // 성공: 데이터 저장 후 결과 페이지 이동
                localStorage.setItem("wearResult", JSON.stringify(data));
                localStorage.setItem("capturedImage", canvas.toDataURL()); 
                
                loadingText.innerText = "분석 완료!";
                setTimeout(() => {
                    window.location.href = "/result";
                }, 500);
            } else {
                // ❌ 분석 실패 시: 모달 띄우기
                showErrorModal("분석에 실패했습니다.<br>" + (data.message || "신발이 잘 보이게 다시 찍어주세요."));
            }
        } catch (err) {
            clearInterval(textInterval);
            console.error(err);
            // ❌ 네트워크 에러 시: 모달 띄우기
            showErrorModal("서버와 연결할 수 없습니다.<br>잠시 후 다시 시도해주세요.");
        }
    }, "image/jpeg", 0.9);
});



// 앱 시작 시 카메라 실행
startCamera();
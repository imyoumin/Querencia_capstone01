document.addEventListener("DOMContentLoaded", () => {
    // ============================================
    // 1. DOM 요소 가져오기
    // ============================================
    const donutChart = document.getElementById("donutChart");
    const scoreValue = document.getElementById("scoreValue");
    const statusArea = document.getElementById("statusArea");
    
    // AR 제안 섹션
    const arSection = document.getElementById("arSection");
    const arIntroView = document.getElementById("arIntroView");
    const arLoadingView = document.getElementById("arLoadingView");
    const suggestionText = document.querySelector(".suggestion-text");
    const arBtn = document.getElementById("arBtn");
    
    const retryBtn = document.getElementById("retryBtn");

    // ============================================
    // 2. 데이터 유효성 검사 및 파싱
    // ============================================
    const storedData = localStorage.getItem("wearResult");
    if (!storedData) {
        // 데이터 없으면 홈으로 (테스트 시 주석 처리 가능)
        window.location.href = "/";
        return;
    }
    const data = JSON.parse(storedData);

    // ============================================
    // 3. 마모도 점수 계산
    // ============================================
    let p1 = data.probs.p1_moderate || 0;
    let p2 = data.probs.p2_heavy || 0;

    let calculatedScore = ((p1 * 65) + (p2 * 100))*2;

    // ✔ 모델이 계산한 원래 점수 (0~100)
    let baseScore = Math.min(Math.round(calculatedScore), 100);

    // ✔ 0~50 난수 생성
    let randomNoise = Math.floor(Math.random() * 31); // 0~50

    // ✔ 난수 더한 최종 점수 (100 넘지 않도록 제한)
    let finalScore = Math.min(baseScore + randomNoise, 99);

    // ===========================
    // 디버깅용 콘솔 출력
    // ===========================
    console.log(
        `%c[Wear Score Debug]\n` +
        `기존 점수: ${baseScore}\n` +
        `추가된 난수: ${randomNoise}\n` +
        `최종 점수: ${finalScore}`,
        "color:#4CAF50; font-weight:bold;"
    );

// 최종 점수
console.log(`Wear Score (Final): ${finalScore}`);


    // 상태 메시지 정의
    let statusInfo = {};
    if (finalScore <= 20) {
        statusInfo = { emoji: "✨", title: "아주 깨끗해요!", desc: "새 신발 컨디션입니다.<br>걱정 없이 신으셔도 됩니다.", color: "#2E7D32" };
    } else if (finalScore <= 40) {
        statusInfo = { emoji: "👟", title: "상태 양호", desc: "일상적인 사용감만 조금 있어요.<br>아직 구조적으로 매우 안정적입니다.", color: "#66BB6A" };
    } else if (finalScore <= 60) {
        statusInfo = { emoji: "🤔", title: "주의가 필요해요", desc: "겉감 흐트러짐, 주름, 변색 등이 눈에 띄기 시작합니다.<br>신었을 때 편안함이 조금씩 달라질 수 있어요.", color: "#FFCA28" };
    } else if (finalScore <= 80) {
        statusInfo = { emoji: "⚠️", title: "교체를 권장해요", desc: "신발의 형태가 무너지고 소재 피로가 누적된 상태예요.<br>착용감 저하와 외형 손상이 뚜렷합니다.", color: "#F57C00" };
    } else {
        statusInfo = { emoji: "🚨", title: "위험 상태!", desc: "신발이 제 기능을 유지하기 어려운 단계입니다.<br>외형 손상, 변형, 내구성 저하가 명확하여 교체가 필요합니다.", color: "#D32F2F" };
    }

    // ============================================
    // 4. 애니메이션 실행
    // ============================================
    function animateResult() {
        let currentScore = 0;
        const duration = 2000;
        const intervalTime = 20;
        const step = Math.max(0.5, finalScore / (duration / intervalTime));

        const timer = setInterval(() => {
            currentScore += step;
            if (currentScore >= finalScore) {
                currentScore = finalScore;
                clearInterval(timer);
                
                setSuggestionText(finalScore);
                setTimeout(() => { 
                    arSection.classList.add("show"); 
                }, 500);
            }
            
            scoreValue.innerText = Math.floor(currentScore);
            donutChart.style.background = `conic-gradient(${statusInfo.color} 0% ${currentScore}%, #f0f0f0 ${currentScore}% 100%)`;
        }, intervalTime);

        statusArea.innerHTML = `
            <span class="status-emoji">${statusInfo.emoji}</span>
            <h2 class="status-title" style="color:${statusInfo.color}">${statusInfo.title}</h2>
            <p class="status-desc">${statusInfo.desc}</p>
        `;
    }

    function setSuggestionText(score) {
        let htmlContent = "";
        if (score <= 20) { htmlContent = `"신발 상태가 완벽하네요! ✨<br>그래도 <span class='suggestion-highlight'>최신 유행 신발</span>은 궁금하지 않으세요?"`; } 
        else if (score <= 60) { htmlContent = `"아직 튼튼하지만...👟<br><span class='suggestion-highlight'>다른 스타일</span>로 기분 전환 해보실래요?"`; } 
        else { htmlContent = `"이 신발은 이제 쉬게 해주세요..😢<br><span class='suggestion-highlight'>AI 추천 새 신발</span>을 신어보시겠어요?"`; }
        suggestionText.innerHTML = htmlContent;
    }

    setTimeout(animateResult, 300);

    // ============================================
    // 5. [중요] 버튼 클릭 -> 신호 전송 -> 페이지 이동
    // ============================================
    arBtn.addEventListener("click", async () => {
        // 1. 버튼 중복 클릭 방지
        arBtn.disabled = true;

        // 2. 로딩 화면 표시 (UX)
        arIntroView.style.display = "none";
        arLoadingView.style.display = "flex";

        try {
            // 3. 서버에 가림막 OPEN 신호 전송
            console.log("Sending AR Toggle Signal...");
            await fetch("/toggle_ar", { method: "POST" });
            
            // 4. 신호 전송 성공 시, 잠시 대기 후 페이지 이동 (가림막 열리는 시간 고려)
            // 즉시 이동하려면 setTimeout 제거하고 바로 이동해도 됩니다.
            setTimeout(() => {
                window.location.href = "/recommend";
            }, 1000); // 1초 정도 로딩 보여줌

        } catch (err) {
            console.error("AR Signal Failed:", err);
            // 에러가 나도 일단 이동은 시킴 (혹은 alert 표시)
            alert("AR 장비 연결 확인이 필요합니다. 화면만 이동합니다.");
            window.location.href = "/recommend";
        }
    });

    // [처음으로 돌아가기]
    retryBtn.addEventListener("click", () => {
        window.location.href = "/";
    });
});


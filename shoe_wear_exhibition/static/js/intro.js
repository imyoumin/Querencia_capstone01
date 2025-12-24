// static/js/intro.js
let currentStep = 1;
const totalSteps = 3;

const nextBtn = document.getElementById('nextBtn');
const startBtn = document.getElementById('startBtn');

nextBtn.addEventListener('click', () => {
    // 현재 슬라이드 숨김
    document.getElementById(`slide${currentStep}`).classList.remove('active');
    
    currentStep++;
    
    if (currentStep <= totalSteps) {
        // 다음 슬라이드 표시
        document.getElementById(`slide${currentStep}`).classList.add('active');
        
        // 마지막 단계 도달 시 버튼 교체
        if (currentStep === totalSteps) {
            nextBtn.style.display = 'none';   // '다음' 숨김
            startBtn.style.display = 'flex';  // '시작하기' 표시 (flex로 해야 css .btn 속성 유지)
        }
    }
});



function goToAnalysis() {
    window.location.href = '/analysis';
}
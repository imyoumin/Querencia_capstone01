/* static/js/recommend.js */

document.addEventListener("DOMContentLoaded", () => {
    // ============================================
    // 1. DOM 요소 가져오기
    // ============================================
    const shoeImageDisplay = document.getElementById("shoeImageDisplay");
    const shoeNameDisplay = document.getElementById("shoeNameDisplay");
    
    // 버튼들
    const prevBtn = document.getElementById("prevBtn");
    const nextBtn = document.getElementById("nextBtn");
    const buyBtn = document.getElementById("buyBtn");
    const retryBtn = document.getElementById("retryBtn");
    
    // 모달 및 QR 이미지 요소
    const qrModal = document.getElementById("qrModal");
    const closeQrBtn = document.getElementById("closeQrBtn");
    const qrImage = document.getElementById("qrImage");

    // ============================================
    // 2. 신발 데이터 정의
    // ============================================
    const shoes = [
        { 
            key: 'Red', 
            name: 'Rapid Red',     
            img: '/static/shoe_image/Red.png',
            qr: '/static/images/qr_red.png'   
        },
        { 
            key: 'Green', 
            name: 'Forest Green', 
            img: '/static/shoe_image/Green.png',
            qr: '/static/images/qr_green.png' 
        },
        { 
            key: 'Blue', 
            name: 'Ocean Blue',    
            img: '/static/shoe_image/Blue.png',
            qr: '/static/images/qr_blue.png'  
        },
        { 
            key: 'Navy', 
            name: 'Deep Navy',     
            img: '/static/shoe_image/Navy.png',
            qr: '/static/images/qr_navy.png'  
        },
        { 
            key: 'Aluminium', 
            name: 'Sleek Silver',  
            img: '/static/shoe_image/Aluminium.png',
            qr: '/static/images/qr_alu.png'   
        },
        { 
            key: 'Black', 
            name: 'Classic Black', 
            img: '/static/shoe_image/Black.png',
            qr: '/static/images/qr_black.png' 
        }
    ];

    let currentIndex = 0; 

    // ============================================
    // 3. [핵심 수정] 초기화 함수 (서버 상태 동기화)
    // ============================================
    async function init() {
        try {
            // 1. 서버에 "지금 무슨 신발이야?" 하고 물어봄
            const res = await fetch('/get_shoe');
            const data = await res.json();
            
            // 2. 서버가 알려준 신발(data.shoe)이 우리 리스트의 몇 번째인지 찾음
            if (data.shoe) {
                const serverIndex = shoes.findIndex(s => s.key === data.shoe);
                
                if (serverIndex !== -1) {
                    currentIndex = serverIndex; // 찾았으면 그 인덱스로 설정
                    console.log(`[Init] Server says current shoe is: ${data.shoe} (Index: ${currentIndex})`);
                }
            }
        } catch (err) {
            console.error("Failed to fetch initial shoe state:", err);
            // 에러 나면 그냥 0번(Red) 유지
        }

        // 3. 결정된 인덱스로 UI 업데이트 (서버 전송은 안 함, 이미 서버랑 같은 상태니까)
        updateShoeUI(currentIndex);
    }

    // ============================================
    // 4. UI 업데이트 및 서버 전송 로직
    // ============================================
    
    function updateShoeUI(index) {
        const shoe = shoes[index];
        
        // 텍스트 변경
        shoeNameDisplay.innerText = shoe.name;
        
        // 이미지 변경 (+애니메이션 효과)
        if (shoeImageDisplay) {
            shoeImageDisplay.classList.add("change");
            setTimeout(() => {
                shoeImageDisplay.src = shoe.img;
                shoeImageDisplay.classList.remove("change");
            }, 300); 
        }
    }

    // 사용자가 '화살표'를 눌렀을 때만 서버로 신호를 보냄
    async function sendShoeSignal(shoeKey) {
        try {
            console.log(`Sending signal for: ${shoeKey}`);
            await fetch(`/set_shoe/${shoeKey}`, { method: "POST" });
        } catch (err) {
            console.error("Projector signal failed:", err);
        }
    }

    function changeShoe(direction) {
        currentIndex += direction;

        if (currentIndex < 0) {
            currentIndex = shoes.length - 1;
        } else if (currentIndex >= shoes.length) {
            currentIndex = 0;
        }

        // UI 업데이트
        updateShoeUI(currentIndex);
        // [중요] 사용자가 직접 바꿀 때만 서버에 신호 전송
        sendShoeSignal(shoes[currentIndex].key);
    }

    // ============================================
    // 5. 이벤트 리스너 등록
    // ============================================
    
    prevBtn.addEventListener("click", () => changeShoe(-1));
    nextBtn.addEventListener("click", () => changeShoe(1));

    buyBtn.addEventListener("click", () => {
        const currentShoe = shoes[currentIndex];
        if(qrImage) {
            qrImage.src = currentShoe.qr;
        }
        qrModal.style.display = "flex";
    });

    closeQrBtn.addEventListener("click", () => {
        qrModal.style.display = "none";
    });

    qrModal.addEventListener("click", (e) => {
        if (e.target === qrModal) {
            qrModal.style.display = "none";
        }
    });

    retryBtn.addEventListener("click", async () => {
        try {
            await fetch("/reset_ar", { method: "POST" });
        } catch (err) { console.error(err); }
        window.location.href = "/";
    });

    

    // 실행
    init();
});


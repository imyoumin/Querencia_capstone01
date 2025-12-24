<div align="center">

# 👟 Wear Lab — 신발 마모도 체험 + AR 착화 전시
**Capstone Design 1 | Team Kerencia (케렌시아)**

[![Made with](https://img.shields.io/badge/Made%20with-AR%20%26%20AI-black)](#)
[![Platform](https://img.shields.io/badge/Platform-Web%20%2F%20Exhibition-blue)](#)
[![Status](https://img.shields.io/badge/Status-Active%20Developing-green)](#)

🎬 **소개 영상(YouTube)**: https://youtu.be/G521t-9eQb0  
📄 **전시 매뉴얼(PDF)**: `Project Proposal.pdf`  
🗂️ **회의록 아카이빙(Notion)**: https://www.notion.so/26702f343e7e8025a504df4d94ab0ac5  

<br/>

<img width="200" alt="캐릭터" src="https://github.com/user-attachments/assets/0a31fb96-8301-4a84-a6b5-b03b1f42286d" />

</div>

---

## ✨ 프로젝트 소개
**Wear Lab**은 관람객이 터치스크린으로 체험을 시작한 뒤,  
**신발 마모도를 AI로 측정**하고 그 결과에 따라 **AR로 추천 신발을 착화 체험**할 수 있는 인터랙티브 전시입니다.  
마지막에는 QR을 통해 **구매 페이지로 연결**되어 실제 행동(구매/탐색)까지 이어지도록 설계했습니다.

---

## 🧭 체험 흐름 (Experience Flow)
전시는 아래 순서로 진행됩니다.

- **STEP 1 — 터치스크린으로 체험 시작**  
- **STEP 2 — 신발 마모도 측정 (AI 분석)**  
- **STEP 3 — AR 착화 추천**  
- **STEP 4 — 색상 비교 (G컬러)**  
- **STEP 5 — 선택 및 구매 연결 (QR)**  

> 위 흐름은 팀 케렌시아 매뉴얼의 Step 구성과 동일합니다.

---

## 🗂️ 회의록 아카이빙 (Meeting Archive)
프로젝트의 회의 기록 및 진행 아카이빙은 아래 Notion 페이지에서 확인할 수 있습니다.  
👉 https://www.notion.so/26702f343e7e8025a504df4d94ab0ac5

---

## 🎥 프로토타입 시연 영상 (Prototype Demo Videos)
<div align="center">

<details>
  <summary><b>Prototype v1 — 3-1프로토타입ver1 시연영상.mp4 (Click to open)</b></summary>
  <br/>
  <video controls width="720" src="image/3-1프로토타입ver1%20시연영상.mp4"></video>
  <p><a href="image/3-1프로토타입ver1%20시연영상.mp4">영상 파일로 열기</a></p>
</details>

<br/>

<details>
  <summary><b>Prototype v2 — 3-2프로토타입ver2시연영상.mp4 (Click to open)</b></summary>
  <br/>
  <video controls width="720" src="image/3-2프로토타입ver2시연영상.mp4"></video>
  <p><a href="image/3-2프로토타입ver2시연영상.mp4">영상 파일로 열기</a></p>
</details>

</div>

<blockquote>
※ GitHub README에서는 환경에 따라 <code>&lt;video&gt;</code> 인라인 재생이 제한될 수 있습니다.  
재생이 안 될 경우 위의 <b>“영상 파일로 열기”</b> 링크로 확인해주세요.
</blockquote>

---

## 🖼️ 스크린샷 / 전시 사진
<div align="center">
  <img src="image/IMG_1664.jpg" style="width:170px; height:210px; object-fit:cover;" />
  <img src="image/IMG_1657.jpg" style="width:170px; height:210px; object-fit:cover;" />
  <img src="image/IMG_1662.jpg" style="width:170px; height:210px; object-fit:cover;" />
  <img src="image/IMG_1713.jpg" style="width:210px; height:210px; object-fit:cover;" />
</div>

<br/>

<div align="center">
  <img src="image/ssass.png" width="500" />
</div>

---

## 🧱 시스템 구성 (Architecture)
> AR 신발 구현

AR 신발 착화 기능은 **Snap Lens Studio**에서 제공하는 **Foot Tracking 모델**을 기반으로 구현하였습니다. 본 작품에서는 별도의 신체 인식 모델을 직접 학습하기보다는, 실제 전시 환경에서의 안정성과 실시간 반응성을 우선적으로 고려하여 검증된 트래킹 모델을 활용하였습니다.

<div align="center">
  <img src="image/스크린샷 2025-12-18 185631.png" width="530" />
  <img src="image/스크린샷 2025-12-18 192152.png" width="410" />
</div>

**Lens Studio의 Foot Tracking 기능**은 카메라 입력을 통해 사용자의 발 위치와 방향을 실시간으로 추적하며, 이를 기준으로 3D 신발 모델을 화면 상의 발에 자연스럽게 정렬합니다. 이를 통해 관람자는 별도의 마커나 추가 장비 없이도 AR 신발 착화 체험을 즉각적으로 수행할 수 있습니다. 본 작품에서는 이 트래킹 결과에 **아디다스 삼바 3D 모델**을 연동하여, 발 움직임에 따라 신발이 함께 반응하도록 구성하였습니다.

<div align="center">
  <img src="image/스크린샷 2025-12-18 183315.png" width="430" />
  <img src="image/스크린샷 2025-12-18 183742.png" width="430" />
</div>

또한 본 시스템은 PC 환경에서의 전시 운영을 고려하여 Snap Camera Kit Web SDK 기반 환경에 최적화되었으며, 웹캠을 활용한 실시간 트래킹이 안정적으로 이루어지도록 구성하였습니다. 이를 통해 전시 공간에서도 반복적인 체험과 장시간 운영이 가능한 AR 착화 환경을 구현하였습니다.

---

<h2>📁 디렉터리 구조 (Directory Structure)</h2>

<pre><code>Querencia_capstone01/
├─ ar_wrapper/                 # AR 착화 웹 (Adidas Samba 6종 모델 착화)
├─ shoe_wear_exhibition/       # 마모도 측정 FastAPI 서버 (GroundingDINO + SAM 전처리 + 마모도 모델 추론)
├─ image/                      # README용 이미지/스크린샷/프로토타입 영상
├─ Project Proposal.pdf        # 프로젝트 제안서
├─ Project Manual.pdf          # 프로젝트 매뉴얼
├─ Final Project Report.pdf    # 프로젝트 최종 보고서
└─ README.md</code></pre>

---

<h2>⚙️ 실행 환경 및 실행 방법 (Environment & How to Run)</h2>

<p>본 프로젝트는 전시 운영을 위해 <b>두 개의 로컬 웹 서버</b>를 동시에 실행합니다.</p>

<ul>
  <li>
    <b>(A) shoe_wear_exhibition/ (FastAPI 서버)</b><br/>
    촬영된 신발 사진을 입력으로 받아 <b>GroundingDINO + SAM 기반 전처리</b>를 수행한 뒤,<br/>
    학습된 <b>신발 마모도 측정 모델</b>로 추론하여 결과를 웹에서 보여줍니다.
  </li>
  <li>
    <b>(B) ar_wrapper/ (AR 착화 웹)</b><br/>
    Snap Camera Kit Web SDK 기반 AR 착화 페이지로,<br/>
    <b>Adidas Samba 6가지 3D 모델</b>을 착화 체험할 수 있습니다.
  </li>
</ul>

<h3>✅ 실행 환경 (Prerequisites)</h3>
<ul>
  <li>Python 3.x</li>
  <li>Node.js (npx 사용 가능)</li>
</ul>

<h3>1) FastAPI 서버 실행 (shoe_wear_exhibition/)</h3>

<p><b>(1) 의존성 설치</b></p>
<pre><code>pip install fastapi uvicorn[standard] transformers torch Pillow numpy safetensors accelerate python-multipart rembg</code></pre>

<p><b>(2) 서버 실행</b></p>
<pre><code>cd shoe_wear_exhibition
python app.py</code></pre>

<p><i>만약 app.py가 uvicorn 실행을 포함하지 않는 구조라면 아래 명령으로 실행하세요.</i></p>
<pre><code>uvicorn app:app --host 0.0.0.0 --port 8000 --reload</code></pre>

<h3>2) AR 웹 실행 (ar_wrapper/)</h3>
<pre><code>cd ar_wrapper
npx serve .</code></pre>

<p>터미널에 출력되는 로컬 주소(예: <code>http://localhost:3000</code>)로 접속하면 AR 착화 페이지가 열립니다.</p>

<h3>3) 전시 운영 (두 서버 동시 실행)</h3>
<p>전시 환경에서는 <b>두 터미널을 열어 각각 실행</b>합니다.</p>

<p><b>Terminal A (FastAPI)</b></p>
<pre><code>cd shoe_wear_exhibition
python app.py</code></pre>

<p><b>Terminal B (AR Web)</b></p>
<pre><code>cd ar_wrapper
npx serve .</code></pre>

<p>두 서버가 같은 PC의 로컬 환경에서 동시에 동작하며, <b>AR 착화 체험</b>과 <b>마모도 측정/결과 표시</b>가 전시 흐름에 맞게 연동됩니다.</p>

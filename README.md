

## 🛠 Tech Stack
* **Language:** Python
* **Package Management:** uv
* **Medical Imaging:** SimpleITK
* **3D Processing & Visualization:** PyVista, Open3D
* **Computer Vision & AI:** MediaPipe (Tasks API), OpenCV, NumPy

## 📂 Project Structure
```text
.
├── README.md                 # 프로젝트 설명서
├── pyproject.toml / uv.lock  # 의존성 및 패키지 관리 파일 (uv 환경)
├── data/                     # DICOM, NIfTI 원본 데이터 및 결과 OBJ 저장 폴더
├── face_landmarker.task      # MediaPipe Face Landmark 추론을 위한 가중치 모델
│
# 1. Data Processing & Extraction
├── check_metadata.py         # DICOM/NIfTI 영상의 메타데이터(Spacing 등) 검증
├── skull_extractor.py        # DICOM 기반 두개골(Bone) 메쉬 추출 모듈
├── skin_extractor.py         # DICOM 기반 피부(Face) 메쉬 추출 모듈
├── tumor_extractor.py        # NIfTI 기반 종양(Target) 메쉬 추출 모듈
├── total_extractor.py        # 전체 3D 메쉬 동시 추출 및 절대 좌표 통합 스크립트
│
# 2. Registration & Simulation
├── landmark_picker.py        # 3D 메쉬 상에서 Ground Truth(GT) 랜드마크 수동 추출기
└── xr_simulation.py          # 가상 카메라 렌더링, MediaPipe 추론, 오차 복원 및 3D 정합 (Main)
```

## 🚀 Getting Started

### 1. Environment Setup
이 프로젝트는 빠르고 가벼운 파이썬 패키지 매니저인 `uv`를 사용합니다.

```bash
# uv 설치 (미설치 시)
pip install uv

# 의존성 설치 및 가상환경 동기화
uv sync

# 가상환경 활성화
source .venv/bin/activate  # Mac/Linux
.venv\Scripts\activate     # Windows
```

### 2. Pipeline Execution

**Step 1: 3D Mesh Extraction**
환자의 DICOM 데이터와 분할(Segmentation)된 NIfTI 마스크를 기반으로 3D 메쉬(.obj)를 추출합니다.
Skull과 Skin은 Dicom 데이터에서 임계값 기준으로 추출합니다. Tumor는 라벨링이 되어있다는 가정하에 NIfTI 데이터를 불러와 추출합니다. Taubin Smoothing을 적용하여 표면을 부드럽게 처리하며, 물리적 스케일(Spacing)이 적용된 절대 좌표로 저장됩니다.

```bash
uv run total_extractor.py
```

**Step 2: Ground Truth Landmark Selection**
추출된 3D 피부 메쉬에서 정합의 기준이 될 단단한 지점(Rigid Point) 4곳(코끝, 양쪽 눈꼬리, 인중)을 선택하여 절대 좌표를 획득합니다.

```bash
uv run landmark_picker.py
```

**Step 3: XR Registration Simulation**
추출된 모델과 랜드마크를 기반으로 HMD 환경을 시뮬레이션합니다.

1. 환자의 오차(회전/이동)를 가정한 가상 카메라 촬영
2. MediaPipe를 통한 2D 랜드마크 추출
3. 양 눈꼬리 거리를 기반으로 물리적 스케일(mm) 증폭 및 원점 정렬
4. Open3D Point-to-Point(SVD) 정합 수행 및 RMSE 오차 계산
5. 정합 전/후 비교 및 Mesh Overlay 시각화

```bash
uv run xr_simulation.py
```

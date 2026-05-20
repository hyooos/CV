# 🚗 교통사고 과실비율 자동 분석 시스템

### 블랙박스 영상과 사용자 진술 기반 Hybrid 사고 분석 프레임워크

> 블랙박스 영상과 사용자 진술을 기반으로 사고 상황을 분석하고 과실비율 및 한국어 자연어 보고서를 자동 생성하는 시스템

---

## Overview

본 프로젝트는 블랙박스 사고 영상과 사용자 진술을 입력받아 사고 유형, 차량 진행 정보, 위반사항 등을 자동 분석하는 Hybrid AI 기반 사고 분석 시스템입니다.

영상 이해(Video Understanding), 객체 탐지(Object Detection), 차량 추적(Tracking), trajectory evidence 분석, Rule-based reasoning, LLM 기반 자연어 처리(NLP)를 결합하여 실제 교통사고 분석 프로세스를 자동화했습니다.

또한 Rule-based 과실 추론과 Residual Learning 기반 보정 구조를 함께 사용하여 보다 현실적인 과실비율 산출이 가능하도록 설계했습니다.

---

## Key Features

- 블랙박스 영상 기반 사고 유형 자동 분류
- 객체 탐지 및 IoU Tracker 기반 차량 추적
- 진술문 기반 위반사항 자동 추출 (LLM)
- trajectory evidence 기반 과실 보정
- Rule-based + Residual Learning 결합 구조
- 한국어 자연어 사고 분석 보고서 자동 생성
- Gradio 기반 웹 데모 제공

---

## Tech Stack

#### Language

<p>
  <img src="https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white" height="28"/>
</p>

---

#### Deep Learning & Computer Vision

<p>
  <img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=flat-square&logo=pytorch&logoColor=white" height="28"/>
  <img src="https://img.shields.io/badge/Torchvision-5C6BC0?style=flat-square" height="28"/>
  <img src="https://img.shields.io/badge/Faster R--CNN-1E88E5?style=flat-square" height="28"/>
  <img src="https://img.shields.io/badge/R2Plus1D-00897B?style=flat-square" height="28"/>
</p>

---

#### NLP & LLM

<p>
  <img src="https://img.shields.io/badge/LLaMA--3.1--8B--Instant-8E24AA?style=flat-square" height="28"/>
  <img src="https://img.shields.io/badge/Groq-000000?style=flat-square" height="28"/>
</p>

---

#### Machine Learning & Modeling

<p>
  <img src="https://img.shields.io/badge/Scikit Learn-F7931E?style=flat-square&logo=scikitlearn&logoColor=white" height="28"/>
  <img src="https://img.shields.io/badge/RidgeCV-43A047?style=flat-square" height="28"/>
  <img src="https://img.shields.io/badge/Residual Learning-6D4C41?style=flat-square" height="28"/>
</p>

---

#### Visualization & Interface

<p>
  <img src="https://img.shields.io/badge/Gradio-FF9800?style=flat-square" height="28"/>
  <img src="https://img.shields.io/badge/OpenCV-27338E?style=flat-square&logo=opencv&logoColor=white" height="28"/>
</p>

---

## Pipeline

<img width="4021" height="5263" alt="architecture" src="https://github.com/user-attachments/assets/2d6b7365-9ab2-4cc1-931c-9f0cbdcf3d0f" />

---

## System Architecture

| 단계 | 모듈 | 모델 / 기법 |
|---|---|---|
| Scene / Case Anchor | 영상 분류 | R2Plus1D |
| Statement Parsing | 위반사항 추출 | LLaMA-3.1-8B-Instant (Groq) |
| Perception | 객체 탐지 / 추적 | Faster R-CNN + IoU Tracker |
| Trajectory Reasoning | evidence 생성 | bbox track 기반 행동 분석 |
| Rule + Score Engine | 기준 비율 + 보정 | Lookup Table + RidgeCV |
| Report Generation | 자연어 보고서 생성 | LLaMA-3.1-8B-Instant (Groq) |

---

## Results

#### Main Findings

- 블랙박스 영상 기반 사고 유형 자동 분류 가능
- trajectory evidence 기반 과실 보정 성능 확인
- 규칙 기반 시스템 대비 상황 적응력 향상
- 자연어 보고서 자동 생성으로 해석 가능성 강화
- Video Understanding + Rule-based Reasoning 결합 구조 검증

---

## Directory Structure

```bash
.
├── app.py
├── train_video_classifier.py
├── detect/
│   ├── train_detector.py
│   └── detection_outputs/
├── classification/
│   └── video_classification_outputs/
├── scripts/
│   ├── train_adjustment.py
│   ├── run_case.py
│   ├── build_adjustment_csv.py
│   ├── build_adjustment_csv_from_labels.py
│   └── prepare_input.py
├── src/accident_liability/
│   ├── scene/
│   ├── perception/
│   ├── trajectory/
│   ├── rules/
│   ├── llm/
│   ├── report/
│   ├── pipeline/
│   └── schemas.py
├── data/
│   ├── lookup/
│   └── adjustment_input.csv
├── outputs/adjustment/
├── requirements.txt
└── README.md
```

---

## Run

### Installation

```bash
git clone https://github.com/your-repository.git

cd your-repository

pip install -r requirements.txt
pip install python-dotenv
```

---

### Additional Dependency

`ffmpeg` 시스템 패키지가 필요합니다.

```bash
# macOS
brew install ffmpeg

# Ubuntu / RunPod
apt-get update && apt-get install -y ffmpeg
```

---

### Groq API Key Setup

자연어 보고서 생성에는 Groq API Key가 필요합니다. (무료)

```bash
echo "GROQ_API_KEY=your_api_key" > .env
```

---

### Gradio Demo

```bash
# CPU
python app.py

# GPU
python app.py --device cuda --share
```

- Local URL: `http://localhost:7860`
- `--share` 옵션 사용 시 외부 공개 URL 생성

---

## Model Training

### 1) Video Classifier

R2Plus1D 기반 multi-head classification 모델 학습

```bash
python train_video_classifier.py \
  --train_json classification/video_data/processed/train.json \
  --val_json classification/video_data/processed/val.json \
  --output_dir classification/video_classification_outputs \
  --epochs 100 \
  --batch_size 2 \
  --frame_size 224 \
  --lr 1e-4 \
  --device cuda:0
```

---

### 2) Object Detector

Faster R-CNN 기반 객체 탐지 모델 학습

```bash
python detect/train_detector.py
```

학습된 가중치:

```bash
detect/detection_outputs/checkpoints/faster_rcnn_baseline/best.pth
```

---

### 3) Adjustment Model

evidence 기반 residual learning 모델 학습

```bash
python scripts/train_adjustment.py \
  --input_csv data/adjustment_input.csv \
  --output_dir outputs/adjustment
```

산출물:

```bash
outputs/adjustment/adjustment_model.joblib
learned_adjustment_table.csv
```

---

## Inference (CLI)

GUI 없이 단일 사고 영상을 추론할 때 사용합니다.

```bash
python scripts/run_case.py \
  --video_path path/to/accident.mp4 \
  --statement "상대 차량이 신호를 무시하고 진입했습니다." \
  --accident_place "사거리교차로(신호등 있음)" \
  --base_ratio_csv data/lookup/base_ratio_table.csv \
  --adjustment_model outputs/adjustment/adjustment_model.joblib \
  --classifier_weights classification/video_classification_outputs/best.pth \
  --device cuda \
  --use_llm
```

### Options

- `--classifier_weights`
  - 영상 기반 anchor 정보 자동 예측
- 미지정 시
  - 사고 장소 및 진행 정보 수동 입력 필요
- `--use_llm`
  - 진술문 기반 위반사항 자동 추출 활성화

---

## Conclusion

본 프로젝트는 영상 이해와 규칙 기반 추론, 그리고 LLM 기반 자연어 처리를 통합하여 교통사고 과실비율 분석 과정을 자동화한 시스템입니다.
특히 trajectory evidence와 residual learning 기반 보정 구조를 통해 기존 규칙 기반 방식의 한계를 보완하고 상황 적응력을 향상시켰습니다.
또한 블랙박스 영상과 사용자 진술을 함께 활용함으로써 실제 사고 분석 프로세스와 유사한 end-to-end 파이프라인을 구현했다는 점에서 의의가 있습니다.

# Tacticalist AI 모델 학습 및 검증 저장소

본 저장소는 **TIPS 과제(AX 혁신 기업 창의기술개발 사업)**의 일환으로 수행 중인 객체 탐지 알고리즘의 학습, 검증을 위한 소스코드를 포함하고 있습니다.

## 1. 프로젝트 개요
* **목적**: 전술 환경에서의 객체(총기, 군복, 장비 등 9종) 실시간 탐지 및 분류 알고리즘 개발
* **핵심 모델**: YOLO v11m (Medium 모델 기반 최적화)
* **주요 성과**: **mAP50 기준 91.97%** 달성

## 2. 주요 기능 및 코드 구성

### 2.1 모델 학습 (`train.py`)
전술 환경 특유의 복잡한 시점 변화에 대응하기 위해 기하학적 데이터 증강(Data Augmentation)을 강화하여 학습을 수행합니다.
* **기하학적 증강**: Perspective(0.001), Degrees(15.0), Shear(10.0) 등 트리거캠 및 측면 시점 대응 로직 적용
* **학습 기법**: Mosaic(1.0), Mixup(0.1) 적용 및 에포크 후반부 안정화를 위한 `close_mosaic` 설정
* **하이퍼파라미터**: Imgsz 640, Epochs 200, Batch 32 환경에서 최적 가중치 도출

### 2.2 성능 평가 (`accuracy.py`)
학습된 최적 가중치(`best.pt`)를 활용하여 정량적 지표를 산출합니다.
* **검증 지표**: mAP50, mAP50-95, 각 클래스별 AP 수치 자동 계산
* **데이터 분할**: Test Split을 활용한 엄격한 모델 성능 검증 수행

### 2.3 시각화 검증 (`validation.py`)
수치적 성능을 넘어, 실제 추론 결과의 신뢰성을 시각적으로 검전하기 위한 커스텀 스크립트입니다.
* **GT vs Pred 비교**: 정답 레이블(Ground Truth)과 모델 예측값(Prediction)을 동일 이미지상에 시각화
* **배치 처리**: 대규모 테스트셋에 대한 일괄 추론 및 검증 이미지 자동 저장 기능

## 3. 개발 및 시험 환경
* **OS**: Ubuntu 22.04 LTS
* **GPU**: NVIDIA L4 (AWS G6 Instance)
* **Framework**: PyTorch 2.1.0, Ultralytics 8.4.24, Python 3.10.19
* **Library**: OpenCV, PyYAML, Glob

## 4. 실행 방법

### 4.1 모델 학습
```bash
python train.py
```

### 4.2 성능 측정 (mAP)
```bash
python accuracy.py
```

### 4.3 추론 결과 시각화
```bash
python validation.py
```

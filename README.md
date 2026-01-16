# 🛰️ GSD-Guided Oriented Object Detection

> Improving tiny object localization in satellite imagery using Ground Sample Distance metadata

---

## 🎯 What is this?

A research project that uses **GSD (Ground Sample Distance)** metadata from satellite images to improve detection accuracy for tiny oriented objects.

**In simple terms:**  
Satellite images come with metadata telling you how many meters each pixel represents. We use this information to help the AI better locate small objects like cars, ships, and storage tanks.

---

## 💡 The Core Idea

```
Pixel size (GSD) × Bounding box size (pixels) = Real-world object size (meters)
```

When the AI detects a "small vehicle," we can check:
- Does this 50×50 pixel box actually match a car's real size (~4m)?
- Or did the model make a localization error?

This simple check helps refine bounding box accuracy—critical for tiny objects where 2-pixel errors matter a lot.

---

## 🔍 Why it Matters

**The Problem:**  
Tiny objects in aerial images are hard to localize precisely. Current detectors suffer ~37% performance drop when requiring higher accuracy (IoU 0.5 → 0.75).

**Our Approach:**  
Leverage GSD metadata (already present in satellite imagery) as semantic reasoning to guide localization—without extra sensors or computational cost.

---

## 📊 Dataset

**DOTA v2.0** - Large-scale oriented object detection benchmark for aerial images
- 11,268 satellite/aerial images
- 1.79M annotated instances (18 object categories)
- Includes GSD metadata for physical size reasoning

---

# 🛰️ GSD 기반 회전 객체 탐지

> 위성 영상의 지상 샘플 거리(GSD) 메타데이터를 활용한 초소형 객체 위치 정확도 개선

---

## 🎯 무엇을 하는 프로젝트인가?

위성 영상에 포함된 **GSD(Ground Sample Distance)** 메타데이터를 활용하여 초소형 회전 객체의 탐지 정확도를 향상시키는 연구 프로젝트입니다.

**쉽게 말하면:**  
위성 영상에는 각 픽셀이 실제로 몇 미터를 나타내는지 알려주는 정보(GSD)가 있습니다. 이 정보를 활용해서 자동차, 선박, 저장 탱크 같은 작은 물체들을 AI가 더 정확하게 찾도록 돕습니다.

---

## 💡 핵심 아이디어

```
픽셀 크기 (GSD) × 박스 크기 (픽셀) = 실제 물체 크기 (미터)
```

AI가 "소형 차량"을 탐지했을 때, 우리는 확인할 수 있습니다:
- 이 50×50 픽셀 박스가 실제 자동차 크기(~4m)와 맞는가?
- 아니면 모델이 위치를 잘못 예측한 건가?

이런 간단한 검증으로 바운딩 박스 정확도를 개선할 수 있습니다—특히 2픽셀 오차도 크게 영향을 미치는 초소형 객체에서 중요합니다.

---

## 🔍 왜 중요한가?

**문제점:**  
항공 영상 속 초소형 객체들은 정밀한 위치 추정이 어렵습니다. 현재 탐지기들은 더 높은 정확도를 요구할 때(IoU 0.5 → 0.75) 약 37%의 성능 하락을 겪습니다.

**우리의 접근:**  
위성 영상에 이미 존재하는 GSD 메타데이터를 의미론적 추론(semantic reasoning)에 활용하여 위치 추정을 가이드합니다—추가 센서나 계산 비용 없이.

---

## 📊 데이터셋

**DOTA v2.0** - 항공 영상 회전 객체 탐지를 위한 대규모 벤치마크
- 11,268개의 위성/항공 영상
- 1.79M개의 주석된 인스턴스 (18개 객체 카테고리)
- 물리적 크기 추론을 위한 GSD 메타데이터 포함

---

</div>
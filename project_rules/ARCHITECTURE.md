# Yoga AI Trainer - System Architecture

## Project Overview

This project is a modular and research-oriented AI-based yoga pose detection and correction system.

The system compares:
- Multiple pose estimation frameworks
- Multiple machine learning models
- Real-time performance metrics
- Pose-specific error detection accuracy

The system is designed for:
- scalability
- modularity
- reusable ML pipelines
- comparative analysis
- real-time inference

---

# Core Objectives

The project focuses on:

1. Real-time yoga pose detection
2. Pose-specific mistake classification
3. Real-time correction feedback
4. Comparative evaluation of:
   - MediaPipe vs MoveNet
   - Random Forest vs SVM vs KNN vs Logistic Regression
5. Lightweight and efficient inference

---

# High-Level System Flow

```text
Camera / Video Input
        ↓
Landmark Extraction
        ↓
Feature Engineering
        ↓
Pose-Specific Classification
        ↓
Error Detection
        ↓
Correction Feedback
        ↓
Evaluation & Comparison
```

---

# System Modules

## 1. Landmark Extraction

Responsible for:
- extracting body landmarks/keypoints
- supporting multiple extractors

Implemented Extractors:
- MediaPipe
- MoveNet

Folder:
```text
src/landmark_extraction/
```

Rules:
- all extractors must follow a common interface
- extraction logic must remain independent from ML logic

---

## 2. Feature Engineering

Responsible for:
- angle calculations
- landmark normalization
- feature scaling
- distance calculations
- feature vector generation

Folder:
```text
src/feature_engineering/
```

Goals:
- avoid duplicated feature logic
- support reusable feature pipelines
- support multiple extractors

---

## 3. Dataset Management

Responsible for:
- dataset loading
- validation
- preprocessing
- train/test splitting

Folder:
```text
src/datasets/
```

Dataset Structure:
```text
data/raw/<pose_name>/<error_class>/
```

Example:
```text
data/raw/tree_pose/bent_knee/
```

---

## 4. Machine Learning Models

Responsible for:
- training
- inference
- prediction
- model management

Folder:
```text
src/models/
```

Supported Models:
- Random Forest
- SVM
- KNN
- Logistic Regression
- and others as well

Architecture Rule:
- training pipeline must remain generic
- evaluation pipeline must remain generic
- models must be interchangeable

---

## 5. Evaluation

Responsible for:
- accuracy analysis
- confusion matrices
- precision/recall/F1-score
- FPS analysis
- robustness testing

Folder:
```text
src/evaluation/
```

Goals:
- support comparative experiments
- generate reusable evaluation reports

---

## 6. Comparison Module

Responsible for:
- model comparison
- extractor comparison
- benchmark generation
- experiment reporting

Folder:
```text
src/comparison/
```

Research Focus:
- performance comparison
- real-time efficiency
- accuracy tradeoffs

---

## 7. Correction System

Responsible for:
- pose-specific correction rules
- feedback generation
- angle threshold validation

Folder:
```text
src/correction/
```

Architecture Rule:
- correction logic should remain pose-specific
- correction logic must remain isolated from training logic

---

## 8. Visualization

Responsible for:
- plotting metrics
- visualization overlays
- graph generation

Folder:
```text
src/visualization/
```

---

## 9. Demo System

Responsible for:
- webcam demo
- real-time testing
- demo videos

Folder:
```text
src/demo/
```

---

# Design Principles

## Modularity

Each module must:
- have a single responsibility
- remain reusable
- avoid hidden dependencies

---

## Reusability

The system must:
- avoid duplicated logic
- use generic training/evaluation pipelines
- support new poses with minimal code changes

---

## Scalability

Adding:
- a new pose
- a new model
- a new extractor

should require minimal architectural changes.

---

## Separation of Concerns

The following responsibilities must remain isolated:

| Responsibility | Module |
|---|---|
| Extraction | landmark_extraction |
| Features | feature_engineering |
| Training | models |
| Evaluation | evaluation |
| Comparison | comparison |
| Feedback | correction |

---

# Configuration-Based Design

Pose-specific behavior should be controlled using:
```text
configs/poses/
```

NOT hardcoded logic.

Example:
```yaml
pose_name: Plank Pose

classes:
  - correct
  - high_back
  - low_back
```

---

# Model Storage Structure

```text
saved_models/
├── mediapipe/
├── movenet/
├── poseNet/
```

Example:
```text
saved_models/mediapipe/Plank Pose/random_forest.pkl
```

---

# Outputs Structure

```text
outputs/
├── plots/
├── reports/
├── logs/
├── predictions/
```

---

# Engineering Goals

The repository should:
- remain clean and navigable
- support research experiments
- support future expansion
- remain understandable for collaborators

The project should avoid:
- monolithic scripts
- duplicated logic
- tightly coupled modules
- hardcoded paths
- unnecessary complexity

---

# Future Expansion Possibilities

Potential future improvements:
- deep learning classification
- web deployment
- mobile integration
- personalized recommendations
- advanced correction systems
- temporal pose analysis
- sequence-based models
This project is a modular research-oriented AI-based yoga pose detection and correction system.

The project compares:
- MediaPipe vs MoveNet vs poseNet
- Random Forest vs SVM vs KNN and other ML algorithms

The project focuses on:
- reusable architecture
- modular code
- feature engineering
- comparative evaluation
- real-time performance

# Architecture Rules

- Use modular reusable code
- Avoid duplicated logic
- Separate extraction, training, evaluation, and visualization
- Keep files focused on one responsibility
- Use config-driven pipelines
- Use pathlib instead of hardcoded paths
- Keep reusable functions inside shared utilities
- Avoid writing large monolithic scripts

# Folder Rules

- Landmark extraction logic goes inside src/landmark_extraction
- Shared utilities go inside src/utils
- Feature engineering goes inside src/feature_engineering
- Evaluation logic goes inside src/evaluation
- Comparison logic goes inside src/comparison
- Do not place experimental notebooks inside src

# Coding Standards

- Use type hints
- Use docstrings
- Use descriptive naming
- Avoid magic numbers
- Keep functions small and reusable
- Avoid duplicate code
- Use logging instead of print where appropriate

# ML Pipeline Rules

- Training logic must remain generic
- Evaluation logic must remain generic
- Pose-specific logic should stay inside configs or correction modules
- Models must be interchangeable
- Landmark extractors must follow a common interface

# Naming Rules

GOOD:
- compare_models.py
- metrics.py
- feature_pipeline.py

BAD:
- final2.py
- latest_working.py
- temp.py

# Research Goals

The system should support:
- model comparison
- landmark extractor comparison
- FPS benchmarking
- confusion matrix generation
- robustness evaluation

# Important Design Principle

Project should eventually allow this:

```bash
python train.py \
    --pose plank_pose \
    --extractor mediapipe \
    --model random_forest
```
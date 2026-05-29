# YogAI — Comparative Yoga Pose Detection & Correction

A modular, research-oriented ML system for yoga pose detection, mistake classification,
real-time correction feedback, and comparative evaluation of multiple pose estimation
frameworks and ML algorithms.

---

## Project Goals

| Axis | Options Compared |
|---|---|
| Pose Estimator | MediaPipe · MoveNet · PoseNet |
| ML Classifier | Random Forest · SVM · KNN · Logistic Regression |
| Evaluation | Accuracy · F1 · FPS · Robustness |

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Prepare dataset (extract landmarks + engineer features)
python scripts/prepare_dataset.py --pose tree_pose --extractor mediapipe

# 3. Train models for a pose
python scripts/run_training_pipeline.py --pose tree_pose --extractor mediapipe

# 4. Evaluate all models
python scripts/run_full_evaluation.py --pose tree_pose --extractor mediapipe

# 5. Run cross-extractor comparison
python scripts/run_comparison.py --pose tree_pose

# 6. Launch webcam demo
python src/demo/webcam_demo.py --pose tree_pose --extractor mediapipe --model random_forest
```

---

## Repository Structure

```
yogai/
├── configs/          # YAML configs for poses, models, extractors, training
├── data/             # Raw images, processed CSVs, annotations
├── notebooks/        # Jupyter experiments (never production code)
├── src/              # All source modules (extraction → features → models → eval)
├── saved_models/     # Serialised .pkl models organised by extractor/pose
├── outputs/          # Plots, reports, logs, predictions
├── tests/            # Unit + integration tests
├── scripts/          # Entry-point pipeline runners
└── docs/             # Architecture docs, diagrams, research notes
```

See [docs/architecture/](docs/architecture/) for detailed design docs.

---

## Design Principles

- **Modular** — each file owns one responsibility
- **Config-driven** — pose/model/extractor behaviour lives in `configs/`, not code
- **Generic pipelines** — training and evaluation are reusable across all poses/models
- **No hardcoded paths** — all paths derived from `configs/paths.py` via `pathlib`
- **Interchangeable components** — extractors and models follow common interfaces

---

## Adding a New Pose

1. Create `data/raw/<new_pose>/` with class sub-folders
2. Add `configs/poses/<new_pose>.yaml`
3. Run `scripts/prepare_dataset.py --pose <new_pose>`
4. Run `scripts/run_training_pipeline.py --pose <new_pose>`

No source code changes required.

---

## Adding a New ML Model

1. Add `src/models/classical_ml/<new_model>.py` implementing `BaseModel`
2. Register it in `src/models/model_registry.py`

---

## Adding a New Pose Estimator

1. Add `src/landmark_extraction/<new_extractor>_extractor.py` implementing `BaseExtractor`
2. Register it in `src/landmark_extraction/extractor_factory.py`
3. Add `configs/extractors/<new_extractor>.yaml`

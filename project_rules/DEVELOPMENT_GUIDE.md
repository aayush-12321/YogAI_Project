# Development Guide

This document defines the engineering standards, coding practices, and development workflow for the Yoga AI Trainer project.

The goal is to maintain:
- clean architecture
- modular code
- reusable components
- scalable ML pipelines
- consistent project structure

---

# Core Development Principles

## 1. Single Responsibility Principle

Each file/module should focus on one responsibility only.

GOOD:
```text
metrics.py
angle_features.py
trainer.py
```

BAD:
```text
all_in_one.py
final_working.py
```

---

## 2. Avoid Duplicate Logic

Reusable logic must be centralized.

Example:
- angle calculations → feature_engineering/
- scaling → feature_engineering/
- plotting → visualization/

Never copy-paste the same logic across files.

---

## 3. Keep Pipelines Generic

Training and evaluation pipelines must remain reusable.

GOOD:
```python
train_model(pose="tree_pose", model="svm")
```

BAD:
```python
train_tree_pose_svm_final.py
```

---

# Folder Usage Rules

| Folder | Purpose |
|---|---|
| src/landmark_extraction | pose extraction logic |
| src/feature_engineering | feature generation |
| src/models | training + inference |
| src/evaluation | metrics and evaluation |
| src/comparison | benchmarking |
| src/correction | correction rules |
| src/utils | reusable utilities |
| notebooks/ | temporary experiments |
| outputs/ | generated outputs |

---

# Coding Standards

## Naming Conventions

### Files
Use:
```text
snake_case.py
```

GOOD:
```text
compare_models.py
plot_metrics.py
```

BAD:
```text
FinalVersion.py
latest2.py
```

---

## Variables

GOOD:
```python
joint_angle
feature_vector
prediction_result
```

BAD:
```python
x
temp
data2
```

---

# Function Design Rules

Functions should:
- be small
- be reusable
- do one thing only

GOOD:
```python
calculate_angle()
normalize_landmarks()
generate_feature_vector()
```

BAD:
```python
process_everything()
```

---

# Type Hints

Use type hints whenever possible.

GOOD:
```python
def calculate_angle(
    point_a: tuple,
    point_b: tuple,
    point_c: tuple
) -> float:
```

---

# Docstrings

Use concise docstrings.

GOOD:
```python
def normalize_landmarks():
    """
    Normalize pose landmarks relative to body center.
    """
```

---

# Path Handling

NEVER hardcode paths.

BAD:
```python
"C:/Users/name/Desktop/project"
```

GOOD:
```python
from pathlib import Path
```

Use centralized path configs.

---

# Logging

Use logging for important events.

Avoid excessive print statements.

GOOD:
```python
logger.info("Training completed")
```

---

# Configuration Rules

Configurations must remain separate from code.

Use:
```text
configs/
```

Examples:
- model configs
- training configs
- pose configs

Avoid hardcoded thresholds whenever possible.

---

# Experimentation Rules

Temporary experiments belong in:
```text
notebooks/
```

Core logic must NOT remain inside notebooks.

Reusable logic should be moved into:
```text
src/
```

---

# Model Development Workflow

## Step 1
Collect dataset

## Step 2
Extract landmarks

## Step 3
Generate features

## Step 4
Train models

## Step 5
Evaluate models

## Step 6
Compare results

## Step 7
Deploy best model in demo system

---

# Evaluation Standards

Every experiment should ideally report:
- accuracy
- precision
- recall
- F1-score
- confusion matrix
- inference speed
- FPS (if applicable)

---

# Comparison Rules

Comparisons should remain reproducible.

When comparing models:
- use same dataset split
- use same features
- use same evaluation metrics

When comparing extractors:
- use same ML model
- use same test conditions

---

# Git Practices

## Commit Naming

GOOD:
```text
add movenet extractor
refactor training pipeline
implement evaluation metrics
```

BAD:
```text
update
changes
fix
```

---

# File Size Rules

Avoid excessively large files.

Recommended:
- split files when they exceed ~300–500 lines
- separate reusable utilities properly

---

# Architecture Constraints

The project should avoid:
- tightly coupled modules
- duplicated pipelines
- giant scripts
- hidden dependencies
- unnecessary overengineering

---

# AI Agent Guidelines

When generating code using AI agents:
- prioritize modularity
- avoid duplicate code
- follow existing folder structure
- maintain reusable abstractions
- preserve architecture consistency
- avoid unnecessary complexity

---

# Project Philosophy

This repository is designed to:
- resemble a professional ML research repository
- remain understandable and maintainable
- support future expansion
- support reproducible experiments

The project prioritizes:
- architecture quality
- experimentation quality
- modularity
- research-oriented structure

over unnecessary complexity or excessive features.
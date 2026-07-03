<div align="center">
  <img src="https://img.icons8.com/color/96/000000/yoga.png" alt="YogAI Logo">
  <h1>YogAI</h1>
  <p><b>An AI-powered Yoga Pose Classification & Machine Learning Pipeline</b></p>

  <p>
    <img src="https://img.shields.io/badge/Python-3.12-blue?logo=python&logoColor=white" alt="Python">
    <img src="https://img.shields.io/badge/scikit--learn-1.5.2-orange?logo=scikit-learn&logoColor=white" alt="scikit-learn">
    <img src="https://img.shields.io/badge/XGBoost-3.2.0-blue?logo=xgboost&logoColor=white" alt="XGBoost">
    <img src="https://img.shields.io/badge/MediaPipe-0.10.35-blueviolet" alt="MediaPipe">
    <img src="https://img.shields.io/badge/OpenCV-4.9.0-green?logo=opencv&logoColor=white" alt="OpenCV">
  </p>

  > **🌐 Looking for the Web App?** 
  > Visit the [YogAI Web Application Repository](https://github.com/aayush-12321/yogai-web) to explore the full-stack web app. It supports live yoga sessions and video uploads, providing real-time analysis with both audio and textual feedback!
</div>

---

## 🧘‍♂️ Overview

**YogAI** is a comprehensive machine learning pipeline designed to classify various Yoga poses. Built with modularity and real-world deployment in mind, YogAI systematically parses video/image data, extracts human pose landmarks, calculates complex geometric features, and trains robust machine learning models (like Logistic Regression, SVC, KNN, and XGBoost) to classify and assess posture accuracy.

## 🧘‍♀️ Supported Poses & Classification Classes

YogAI currently supports analyzing and correcting posture for three foundational yoga poses, as well as a master model capable of classifying the pose itself:

- **Plank Pose:** Classifies alignment into `c` (correct), `h` (hips too high), and `l` (hips too low).
- **Warrior II Pose:** Identifies forms such as `correct`, `back_leg_bent`, `drooping_arms`, `knee_caved`, `leaning_torso`, and `shallow_stance`.
- **Mountain Pose:** Detects forms like `correct`, `arms_not_dropped`, `bent_forward`, `tilted_head`, and `wide_stance`.
- **Master Model:** A unified model that determines which pose the user is attempting (`mountain`, `plank`, or `warrior2`) before handing off to a pose-specific classifier.

## ✨ Key Features

- **Robust Feature Engineering:** Computes precise joint angles and distances using pose landmarks tailored specifically for each yoga pose.
- **Experimental Tracking & ML Notebooks:** Structured experimental workflows inside Jupyter Notebooks for immediate visualization of model performance and comparisons.
- **Latency Profiling:** Optimizes ML pipelines for rapid execution and calculates metrics to ensure they run smoothly in real-time edge environments.
- **Live Demo & Video Inference:** Built-in demo notebooks and scripts that apply trained models directly on live webcam feeds or pre-recorded videos.
- **Modular Architecture:** A clean, segmented architecture where data processing, feature engineering, and modeling are strictly decoupled to allow easy scaling and expansion.

## 📂 Project Structure & Workflow Detail

The repository is modularized into distinct domains, allowing for easy navigation depending on the task:

### 1. `src/feature_engineering/` - Custom Feature Extraction
This module is responsible for calculating domain-specific geometric features (angles, distances) from raw landmarks. It contains specialized scripts designed for specific poses:
- `plank_specific_fe.py`: Logic specific to plank alignment (e.g., shoulder-hip-ankle angles).
- `warrior2_specific_fe.py`: Logic specific to Warrior II stance features.
- `master_model_fe.py`: Generalized feature extraction combining logic for a unified multi-pose classifier.

### 2. `src/models/` - Experiments, Training, & Saved Models
Navigate here to run experiments, view benchmark results, and access exported pipelines.
- `training/`: Contains dedicated Jupyter notebooks (e.g., `plank_pose_experiment.ipynb`, `master_model.ipynb`) used for training classifiers, optimizing hyperparameters, and evaluating performance metrics (F1-score, accuracy, confusion matrices).
- `saved_files/`: This is where the fully fitted models, scikit-learn pipelines (`.joblib`), and metadata JSON files are exported after a successful notebook run.
- `models.py`: Factory functions returning unfitted sklearn Pipelines and estimators for consistent experimentation.

### 3. `src/demo/` - Real-Time Inference & Demos
Once models are trained and saved, navigate here to test them in the real world.
- Contains interactive Jupyter notebooks (e.g., `test_video_master.ipynb`, `test_video_mountain.ipynb`) designed to process `.mp4` video files or live webcam feeds.
- Interacts with the MediaPipe `pose_landmarker_lite.task` to parse skeletons and overlay predictions using OpenCV.

### 4. `src/landmark_extraction/` - Raw Coordinate Extraction
- Scripts and utilities handling raw coordinate extraction directly from MediaPipe APIs.

##  Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/aayush-12321/YogAI_Project.git
cd YogAI_Project
```

### 2. Create a Virtual Environment

It is recommended to use a virtual environment to manage dependencies:

```bash
python -m venv venv_yogai
# On Windows
venv_yogai\Scripts\activate
# On macOS/Linux
source venv_yogai/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

## Running ML Experiments & Training

Currently, the project uses a notebook-driven approach for experimentation and model building. To train a model or explore metrics:

1. Launch Jupyter Notebook / Lab in your environment.
2. Navigate to `src/models/training/`.
3. Open a pose-specific notebook (e.g., `plank_pose_experiment.ipynb`) or the multi-pose notebook (`master_model.ipynb`).
4. Run all cells. The notebook will automatically execute the data pipeline, train various models, display benchmark tables, and export the best-performing models to `src/models/saved_files/`.

## 🎥 Running the Live Video Demos

To see the trained models in action:

1. Navigate to `src/demo/`.
2. Open a test notebook such as `test_video_master.ipynb`.
3. Point it to a test video file located in `src/demo/test_videos/` 
4. Execute the cells to process the video frame-by-frame, extract real-time landmarks, and render the ML predictions overlaid on the video stream.

## 🛠️ Adding New Components

### Adding a New Model Architecture
All model training and experimentation is done directly inside the Jupyter Notebooks located in the `src/models/training/` folder.
1. Open the relevant training notebook (e.g., `plank_pose_experiment.ipynb`).
2. Add your new sklearn-compatible estimator directly in the notebook's model evaluation loop.
3. Re-run the notebook to benchmark the new model against the existing ones and save the newly exported pipelines.

### Adding a New Pose
1. Implement custom feature extraction logic inside `src/feature_engineering/` (e.g., `newpose_specific_fe.py`) or existing code can be used.
2. Add the corresponding feature keys to `configs/models/models_config.yaml`.
3. Create a new experimental notebook in `src/models/training/` to iterate and train on the new pose data.

## License

Distributed under the MIT License. See `LICENSE` for more information.

---

> **🌐 Looking for the Web App?** 
> Visit the [YogAI Web Application Repository](https://github.com/aayush-12321/yogai-web) to explore the full-stack web app. It supports live yoga sessions and video uploads, providing real-time analysis with both audio and textual feedback!

<!-- <div align="center">
  <i>Built with ❤️ for AI, Health, and Open Source.</i>
</div> -->

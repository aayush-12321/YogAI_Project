## 🐍 Environment Setup (Python 3.11)

Follow these steps to set up the Python environment for this project.

---

### 1️⃣ Install Python 3.11

Download and install Python 3.11 from the official website:  
👉 [https://www.python.org/downloads/release/python-3110/](https://www.python.org/downloads/release/python-3110/)

> ⚠️ During installation, make sure to **check the box** that says  
> “Add Python 3.11 to PATH”.

---

### 2️⃣ Verify Installation

After installation, open **Command Prompt** and verify:
```bash
python --version
```

### 3️⃣ Create a Virtual Environment

Navigate to your project folder.
Then create a virtual environment using Python 3.11:
```bash
"C:\Users\Acer\AppData\Local\Programs\Python\Python311\python.exe" -m venv env
```
This will create a new folder named `env` containing the isolated Python environment.

### 4️⃣ Activate the Virtual Environment

For Windows (Command Prompt):
```bash
env\Scripts\activate
```

### 5️⃣ Install Required Packages

Once the environment is activated, install the dependencies:
```bash
pip install -r requirements.txt
```


### 1. Folder structure

```
plank_model
│   1.data.ipynb - process collected videos
|   2.sklearn.ipynb - train models using Sklearn ML algo
│   3.deep_leaning.ipynb - train models using Deep Learning
│   4.evaluation.ipynb - evaluate trained models
│   5.detection.ipynb - detection on test videos
|   kaggle.csv - data gathered from Kaggle dataset
|   train.csv - train dataset after converted from videos
|   test.csv - test dataset after converted from videos
|   evaluation.csv - models' evaluation results
│
└───model/ - folder contains best trained models and input scaler
│   │
```

### 2. Important landmarks

There are 3 popular errors of basic plank that will be targeted:

-   High lower back: while performing the exercise, instead of keeping the lower back straight, it is raised too high.
-   Low lower back: while performing the exercise, instead of keeping the lower back straight, it is brought down too low.

**_the important MediaPipe Pose landmarks_** for this exercise are: nose, left shoulder, right shoulder, right elbow, left elbow, right wrist, left wrist, right hip, left hip, right knee, left knee, right ankle, left ankle, right heel, left heel, right foot index and left foot index

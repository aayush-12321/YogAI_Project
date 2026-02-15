#!/usr/bin/env python
# coding: utf-8

# In[1]:


from common_utils import vector_from_points, angle_between_vectors


# In[2]:


from common_utils import save_dataframe_to_csv


# In[3]:


import pandas as pd
import numpy as np


# In[6]:


# Elbow angle (both arms straight)
def compute_elbow_angle(row, side="left"):
    shoulder = row[[f"{side}_shoulder_x", f"{side}_shoulder_y", f"{side}_shoulder_z"]].values
    elbow = row[[f"{side}_elbow_x", f"{side}_elbow_y", f"{side}_elbow_z"]].values
    wrist = row[[f"{side}_index_x", f"{side}_index_y", f"{side}_index_z"]].values

    vec1 = vector_from_points(shoulder, elbow)
    vec2 = vector_from_points(wrist, elbow)

    return angle_between_vectors(vec1, vec2)

# Shoulder line angle (arms in one straight line)

# Angle at left shoulder formed by:
# RIGHT_SHOULDER – LEFT_SHOULDER – LEFT_ELBOW

def compute_shoulder_line_angle(row, side="left"):
    if side == "left":
        opp_shoulder = row[["right_shoulder_x","right_shoulder_y","right_shoulder_z"]].values
    else:
        opp_shoulder = row[["left_shoulder_x","left_shoulder_y","left_shoulder_z"]].values

    shoulder = row[[f"{side}_shoulder_x", f"{side}_shoulder_y", f"{side}_shoulder_z"]].values
    elbow = row[[f"{side}_elbow_x", f"{side}_elbow_y", f"{side}_elbow_z"]].values

    vec1 = vector_from_points(opp_shoulder, shoulder)
    vec2 = vector_from_points(elbow, shoulder)

    return angle_between_vectors(vec1, vec2)

# Knee angle (front & back leg)
def compute_knee_angle(row, side="left"):
    hip = row[[f"{side}_hip_x", f"{side}_hip_y", f"{side}_hip_z"]].values
    knee = row[[f"{side}_knee_x", f"{side}_knee_y", f"{side}_knee_z"]].values
    heel = row[[f"{side}_heel_x", f"{side}_heel_y", f"{side}_heel_z"]].values

    vec1 = vector_from_points(hip, knee)
    vec2 = vector_from_points(heel, knee)

    return angle_between_vectors(vec1, vec2)

# Hip angle (torso vs front thigh)
def compute_hip_angle(row, side="left"):
    shoulder = row[[f"{side}_shoulder_x", f"{side}_shoulder_y", f"{side}_shoulder_z"]].values
    hip = row[[f"{side}_hip_x", f"{side}_hip_y", f"{side}_hip_z"]].values
    knee = row[[f"{side}_knee_x", f"{side}_knee_y", f"{side}_knee_z"]].values

    vec1 = vector_from_points(shoulder, hip)
    vec2 = vector_from_points(knee, hip)

    return angle_between_vectors(vec1, vec2)

# Torso vertical angle

# Angle between:
# HIP – SHOULDER – NOSE
def compute_torso_angle(row, side="left"):
    hip = row[[f"{side}_hip_x", f"{side}_hip_y", f"{side}_hip_z"]].values
    shoulder = row[[f"{side}_shoulder_x", f"{side}_shoulder_y", f"{side}_shoulder_z"]].values
    nose = row[["nose_x","nose_y","nose_z"]].values

    vec1 = vector_from_points(hip, shoulder)
    vec2 = vector_from_points(nose, shoulder)

    return angle_between_vectors(vec1, vec2)


# Detect Front vs Back Leg Automatically
def detect_front_leg(row):
    left_knee = compute_knee_angle(row, "left")
    right_knee = compute_knee_angle(row, "right")

    if left_knee < right_knee:
        return "left", "right"
    else:
        return "right", "left"






# In[7]:


def add_warrior2_angles(df):
    angle_columns = []

    df["left_elbow_angle"] = df.apply(lambda r: compute_elbow_angle(r, "left"), axis=1)
    df["right_elbow_angle"] = df.apply(lambda r: compute_elbow_angle(r, "right"), axis=1)

    df["left_shoulder_line_angle"] = df.apply(lambda r: compute_shoulder_line_angle(r, "left"), axis=1)
    df["right_shoulder_line_angle"] = df.apply(lambda r: compute_shoulder_line_angle(r, "right"), axis=1)

    df["left_knee_angle"] = df.apply(lambda r: compute_knee_angle(r, "left"), axis=1)
    df["right_knee_angle"] = df.apply(lambda r: compute_knee_angle(r, "right"), axis=1)

    df["left_hip_angle"] = df.apply(lambda r: compute_hip_angle(r, "left"), axis=1)
    df["right_hip_angle"] = df.apply(lambda r: compute_hip_angle(r, "right"), axis=1)

    df["left_torso_angle"] = df.apply(lambda r: compute_torso_angle(r, "left"), axis=1)
    df["right_torso_angle"] = df.apply(lambda r: compute_torso_angle(r, "right"), axis=1)

    angle_columns = [
        "left_elbow_angle", "right_elbow_angle",
        "left_shoulder_line_angle", "right_shoulder_line_angle",
        "left_knee_angle", "right_knee_angle",
        "left_hip_angle", "right_hip_angle",
        "left_torso_angle", "right_torso_angle"
    ]

    return df, angle_columns


# In[8]:
if __name__ == "__main__":

    df = pd.read_csv("train_augmented_warrior_2.csv")  # Dataset with selective landmarks (all lowercase)
    df, angle_cols = add_warrior2_angles(df)
    print(df.head())
    print("New angle columns:", angle_cols)


    # In[9]:


    # Save dataset using common function
    save_dataframe_to_csv(df, "warrior2_with_angles.csv")


    # In[ ]:





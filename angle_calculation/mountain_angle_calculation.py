#!/usr/bin/env python
# coding: utf-8

# In[1]:


from common_utils import vector_from_points, angle_between_vectors


# In[2]:


from common_utils import save_dataframe_to_csv


# In[3]:


import pandas as pd
import numpy as np


# In[ ]:


def compute_mountain_shoulder_angle(row, side="left"):
    """
    Angle at shoulder formed by: hip - shoulder - elbow
    """
    hip = row[[f"{side}_hip_x", f"{side}_hip_y", f"{side}_hip_z"]].values
    shoulder = row[[f"{side}_shoulder_x", f"{side}_shoulder_y", f"{side}_shoulder_z"]].values
    elbow = row[[f"{side}_elbow_x", f"{side}_elbow_y", f"{side}_elbow_z"]].values
    vec1 = vector_from_points(hip, shoulder)
    vec2 = vector_from_points(elbow, shoulder)
    return angle_between_vectors(vec1, vec2)

def compute_mountain_elbow_angle(row, side="left"):
    """
    Angle at elbow formed by: shoulder - elbow - wrist
    """
    shoulder = row[[f"{side}_shoulder_x", f"{side}_shoulder_y", f"{side}_shoulder_z"]].values
    elbow = row[[f"{side}_elbow_x", f"{side}_elbow_y", f"{side}_elbow_z"]].values
    wrist = row[[f"{side}_wrist_x", f"{side}_wrist_y", f"{side}_wrist_z"]].values
    vec1 = vector_from_points(shoulder, elbow)
    vec2 = vector_from_points(wrist, elbow)
    return angle_between_vectors(vec1, vec2)

def compute_mountain_hip_angle(row, side="left"):
    """
    Angle at hip formed by: shoulder - hip - knee
    """
    shoulder = row[[f"{side}_shoulder_x", f"{side}_shoulder_y", f"{side}_shoulder_z"]].values
    hip = row[[f"{side}_hip_x", f"{side}_hip_y", f"{side}_hip_z"]].values
    knee = row[[f"{side}_knee_x", f"{side}_knee_y", f"{side}_knee_z"]].values
    vec1 = vector_from_points(shoulder, hip)
    vec2 = vector_from_points(knee, hip)
    return angle_between_vectors(vec1, vec2)

def compute_mountain_knee_angle(row, side="left"):
    """
    Angle at knee formed by: hip - knee - ankle
    """
    hip = row[[f"{side}_hip_x", f"{side}_hip_y", f"{side}_hip_z"]].values
    knee = row[[f"{side}_knee_x", f"{side}_knee_y", f"{side}_knee_z"]].values
    ankle = row[[f"{side}_ankle_x", f"{side}_ankle_y", f"{side}_ankle_z"]].values
    vec1 = vector_from_points(hip, knee)
    vec2 = vector_from_points(ankle, knee)
    return angle_between_vectors(vec1, vec2)


# In[ ]:


def add_mountain_angles(df):
    """
    Adds Mountain pose-specific angles to the dataframe.
    Angles added per side: shoulder_angle, elbow_angle, hip_angle, knee_angle
    df: pandas DataFrame with selective landmarks
    Returns df with new angle columns and a list of angle column names
    """
    angle_columns = []
    for side in ["left", "right"]:
        # Shoulder angle: hip - shoulder - elbow
        df[f"{side}_shoulder_angle"] = df.apply(lambda row: compute_mountain_shoulder_angle(row, side), axis=1)
        # Elbow angle: shoulder - elbow - wrist
        df[f"{side}_elbow_angle"] = df.apply(lambda row: compute_mountain_elbow_angle(row, side), axis=1)
        # Hip angle: shoulder - hip - knee
        df[f"{side}_hip_angle"] = df.apply(lambda row: compute_mountain_hip_angle(row, side), axis=1)
        # Knee angle: hip - knee - ankle
        df[f"{side}_knee_angle"] = df.apply(lambda row: compute_mountain_knee_angle(row, side), axis=1)

        angle_columns += [
            f"{side}_shoulder_angle", f"{side}_elbow_angle",
            f"{side}_hip_angle", f"{side}_knee_angle"
        ]

    return df, angle_columns


# In[ ]:
if __name__ == "__main__":

    df = pd.read_csv("train_augmented_mountain.csv")  # Dataset with selective landmarks (all lowercase)
    df, angle_cols = add_mountain_angles(df)
    print(df.head())
    print("New angle columns:", angle_cols)


    # In[ ]:


    # Save dataset using common function
    save_dataframe_to_csv(df, "mountain_with_angles.csv")


    # In[ ]:





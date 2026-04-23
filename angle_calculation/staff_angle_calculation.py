from common_utils import vector_from_points, angle_between_vectors
from common_utils import save_dataframe_to_csv
import pandas as pd
import numpy as np


def compute_staff_shoulder_angle(row, side="left"):
    """Angle at shoulder: hip - shoulder - elbow"""
    hip      = row[[f"{side}_hip_x",      f"{side}_hip_y",      f"{side}_hip_z"]].values
    shoulder = row[[f"{side}_shoulder_x", f"{side}_shoulder_y", f"{side}_shoulder_z"]].values
    elbow    = row[[f"{side}_elbow_x",    f"{side}_elbow_y",    f"{side}_elbow_z"]].values
    return angle_between_vectors(vector_from_points(hip, shoulder), vector_from_points(elbow, shoulder))

def compute_staff_elbow_angle(row, side="left"):
    """Angle at elbow: shoulder - elbow - wrist"""
    shoulder = row[[f"{side}_shoulder_x", f"{side}_shoulder_y", f"{side}_shoulder_z"]].values
    elbow    = row[[f"{side}_elbow_x",    f"{side}_elbow_y",    f"{side}_elbow_z"]].values
    wrist    = row[[f"{side}_wrist_x",    f"{side}_wrist_y",    f"{side}_wrist_z"]].values
    return angle_between_vectors(vector_from_points(shoulder, elbow), vector_from_points(wrist, elbow))

def compute_staff_hip_angle(row, side="left"):
    """Angle at hip: shoulder - hip - knee"""
    shoulder = row[[f"{side}_shoulder_x", f"{side}_shoulder_y", f"{side}_shoulder_z"]].values
    hip      = row[[f"{side}_hip_x",      f"{side}_hip_y",      f"{side}_hip_z"]].values
    knee     = row[[f"{side}_knee_x",     f"{side}_knee_y",     f"{side}_knee_z"]].values
    return angle_between_vectors(vector_from_points(shoulder, hip), vector_from_points(knee, hip))

def compute_staff_knee_angle(row, side="left"):
    """Angle at knee: hip - knee - ankle"""
    hip   = row[[f"{side}_hip_x",   f"{side}_hip_y",   f"{side}_hip_z"]].values
    knee  = row[[f"{side}_knee_x",  f"{side}_knee_y",  f"{side}_knee_z"]].values
    ankle = row[[f"{side}_ankle_x", f"{side}_ankle_y", f"{side}_ankle_z"]].values
    return angle_between_vectors(vector_from_points(hip, knee), vector_from_points(ankle, knee))


def add_staff_angles(df):
    """
    Adds staff pose-specific angles to the dataframe.
    Angles added per side: shoulder_angle, elbow_angle, hip_angle, knee_angle
    df: pandas DataFrame with selective landmarks
    Returns df with new angle columns and a list of angle column names
    """
    angle_columns = []
    for side in ["left", "right"]:
        df[f"{side}_shoulder_angle"] = df.apply(lambda row: compute_staff_shoulder_angle(row, side), axis=1)
        df[f"{side}_elbow_angle"]    = df.apply(lambda row: compute_staff_elbow_angle(row, side),    axis=1)
        df[f"{side}_hip_angle"]      = df.apply(lambda row: compute_staff_hip_angle(row, side),      axis=1)
        df[f"{side}_knee_angle"]     = df.apply(lambda row: compute_staff_knee_angle(row, side),     axis=1)

        angle_columns += [
            f"{side}_shoulder_angle",
            f"{side}_elbow_angle",
            f"{side}_hip_angle",
            f"{side}_knee_angle",
        ]

    return df, angle_columns


if __name__ == "__main__":
    df = pd.read_csv("train.csv")
    df, angle_cols = add_staff_angles(df)
    print(df.head())
    print("New angle columns:", angle_cols)

    save_dataframe_to_csv(df, "staff_with_angles.csv")
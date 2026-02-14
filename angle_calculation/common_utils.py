import pandas as pd
import numpy as np

# =========================
# Common Utility Functions
# =========================

def vector_from_points(a, b):
    """
    Create vector from point a to point b
    a, b: lists or arrays of shape (3,) → x, y, z
    """
    return np.array(b) - np.array(a)

def angle_between_vectors(v1, v2):
    """
    Compute angle (in degrees) between two vectors
    """
    v1 = np.array(v1)
    v2 = np.array(v2)
    cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    # Clip to handle numerical errors
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    return np.degrees(np.arccos(cos_angle))

def save_dataframe_to_csv(df, save_path, include_index=False):
    """
    Save a pandas DataFrame to a CSV file.
    
    Parameters:
    -----------
    df : pandas DataFrame
        The DataFrame to save.
    save_path : str
        The file path where the CSV will be saved.
    include_index : bool, optional
        Whether to include the DataFrame index as a column in the CSV. Default is False.
        
    Returns:
    --------
    None
        The function saves the DataFrame to the specified CSV file.
    
    Example:
    --------
    df = pd.DataFrame(data)
    save_dataframe_to_csv(df, "my_dataset.csv")
    """
    df.to_csv(save_path, index=include_index)
    print(f"DataFrame successfully saved to: {save_path}")

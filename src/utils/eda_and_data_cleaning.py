import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Constants across all datasets
METADATA_COLUMNS = ['source_id', 'label', 'frame_number']

def _extract_biomechanical_features(df, metadata_cols=METADATA_COLUMNS):
    """
    Internal helper to dynamically extract only the physical/geometric 
    features (angles and distances) from the dataframe.
    """
    return [col for col in df.columns if col not in metadata_cols]


# 
# 1. RESEARCH & DATA PROFILING MODULE
# 

def summarize_dataset_structure(file_path):
    """
    Performs a deep statistical overview of the raw data before any modification.
    Perfect for documenting baseline dataset characteristics in a research report.
    """
    print(f"==================================================================")
    print(f"       RESEARCH COMPREHENSIVE PROFILE: {file_path}               ")
    print(f"==================================================================")
    
    df = pd.read_csv(file_path)
    biomechanical_features = _extract_biomechanical_features(df)
    
    # Structural Info
    total_rows, total_cols = df.shape
    print(f" DATASET DIMENSIONS:")
    print(f"   - Total Recorded Frames (Rows): {total_rows}")
    print(f"   - Total Columns: {total_cols}")
    print(f"   - Biomechanical Features Detected ({len(biomechanical_features)}): {biomechanical_features}\n")
    
    # High-Level Summary Statistics
    print(f" FEATURE DESCRIPTIVE STATISTICS (ANGLES/DISTANCES):")
    display(df[biomechanical_features].describe().T[['mean', 'std', 'min', 'max']])
    print("\n")
    
    # Class & Source Distributions
    print(f" LABEL/CLASS DISTRIBUTION (Target Balance):")
    label_counts = df['label'].value_counts()
    label_pct = df['label'].value_counts(normalize=True) * 100
    for idx in label_counts.index:
        print(f"   - Class '{idx}': {label_counts[idx]} frames ({label_pct[idx]:.2f}%)")
    print("\n")
        
    print(f"📹 VIDEO SOURCE DISTRIBUTION (Data Diversity):")
    source_counts = df['source_id'].value_counts()
    print(f"   - Total unique video sources: {df['source_id'].nunique()}")
    print(f"   - Frames per top sources:\n{source_counts.head(5).to_string()}\n")
    
    # Missing Value Integrity Check
    print(f"🔍 LANDMARK DROP/MISSING VALUE PROFILE:")
    missing_series = df.isnull().sum()
    missing_features = missing_series[missing_series > 0]
    if missing_features.empty:
        print("   - Phenomenal! Zero missing values detected in the raw dataset.")
    else:
        for col, count in missing_features.items():
            pct = (count / total_rows) * 100
            print(f"   - Feature '{col}': {count} missing frames ({pct:.2f}%)")
    print(f"==================================================================\n")
    
    return df


# 
# 2. DATA CLEANING MODULE
# 

def drop_missing_landmark_frames(df):
    """
    Removes frames where MediaPipe lost tracking of key body joints.
    """
    initial_rows = df.shape[0]
    df_cleaned = df.dropna().copy()
    dropped_rows = initial_rows - df_cleaned.shape[0]
    
    print(f"[CLEANING] Dropped {dropped_rows} frames due to incomplete landmark tracking.")
    print(f"[CLEANING] Remaining stable frames: {df_cleaned.shape[0]}")
    return df_cleaned


# 
# 3. VISUAL EXPLORATORY DATA ANALYSIS (EDA)
# 

def plot_feature_variance_by_class(df, pose_name="Pose"):
    """
    Generates boxplots of geometric features grouped by label class. 
    Separates angular metrics and distance metrics into separate subplots for scaling clarity.
    """
    biomechanical_features = _extract_biomechanical_features(df)
    
    # Split features by type based on column names
    angle_features = [col for col in biomechanical_features if 'angle' in col.lower()]
    distance_features = [col for col in biomechanical_features if 'angle' not in col.lower()]
    
    # Create subplots only for the feature types that actually exist in the dataset
    plots_to_make = []
    if angle_features:
        plots_to_make.append(("Angular Features (Degrees)", angle_features))
    if distance_features:
        plots_to_make.append(("Distance / Alignment Features (Normalized)", distance_features))
        
    fig, axes = plt.subplots(len(plots_to_make), 1, figsize=(14, max(6, len(biomechanical_features) * 0.7)), squeeze=False)
    
    for idx, (title, features) in enumerate(plots_to_make):
        melted_df = df.melt(id_vars=['label'], value_vars=features, var_name='Feature', value_name='Value')
        ax = axes[idx, 0]
        
        sns.boxplot(data=melted_df, x='Value', y='Feature', hue='label', palette='Set2', ax=ax)
        ax.set_title(f"{title} — {pose_name}", fontsize=12, fontweight='bold')
        ax.set_xlabel("Measurement Value")
        ax.set_ylabel("Extracted Feature")
        ax.grid(axis='x', linestyle='--', alpha=0.5)
        if idx == 0:
            ax.legend(title='label', bbox_to_anchor=(1.02, 1), loc='upper left')
        else:
            ax.get_legend().remove() # Prevent repeating the legend block
            
    plt.tight_layout()
    plt.show()

def plot_feature_collinearity_matrix(df, pose_name="Pose"):
    """
    Plots a Pearson correlation heatmap to diagnose multi-collinearity.
    Helps decide if any overlapping angles can be dropped to simplify the model.
    """
    biomechanical_features = _extract_biomechanical_features(df)
    correlation_matrix = df[biomechanical_features].corr()
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", 
                linewidths=0.5, cbar_kws={'label': 'Correlation Coefficient'})
    plt.title(f"Feature Collinearity Matrix — {pose_name}", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()

def plot_signal_stability_over_time(df, target_source_id=None, pose_name="Pose"):
    """
    Plots feature paths frame-by-frame for a single video sequence.
    Allows you to detect tracking jitter and verify biomechanical signal continuity.
    """
    biomechanical_features = _extract_biomechanical_features(df)
    
    if target_source_id is None:
        target_source_id = df['source_id'].iloc[0]
        
    sequence_df = df[df['source_id'] == target_source_id].sort_values(by='frame_number')
    
    if sequence_df.empty:
        print(f"[ WARNING] No sequential data matches source ID: {target_source_id}")
        return
        
    plt.figure(figsize=(15, 6))
    for feature in biomechanical_features:
        plt.plot(sequence_df['frame_number'], sequence_df[feature], label=feature, alpha=0.8, linewidth=1.5)
        
    plt.title(f"Biomechanical Signal Stability Over Time (Source: {target_source_id}) — {pose_name}", fontsize=14, fontweight='bold')
    plt.xlabel("Sequential Frame Number")
    plt.ylabel("Calculated Metrics")
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.show()
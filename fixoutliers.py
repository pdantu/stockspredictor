import os
import pandas as pd
import numpy as np

# Set path to metrics folder
metrics_folder = os.path.join(os.getcwd(), 'metrics')

# Get all metric CSVs
metric_files = [f for f in os.listdir(metrics_folder) if f.endswith('-metrics.csv')]

# Loop through each file
for file in metric_files:
    path = os.path.join(metrics_folder, file)
    df = pd.read_csv(path)

    # Identify numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns

    # Loop through each numeric column to find and fix outliers
    for col in numeric_cols:
        if col not in df.columns:
            continue
        series = df[col]
        mean_val = series.mean(skipna=True)
        std_val = series.std(skipna=True)

        if std_val == 0 or np.isnan(std_val):
            continue

        z_scores = (series - mean_val) / std_val
        outliers = np.abs(z_scores) > 3

        if outliers.any():
            df.loc[outliers, col] = mean_val
            print(f"✅ Fixed outliers in '{col}' of {file}")

    # Save the cleaned file
    df.to_csv(path, index=False)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pickle
import os

# Paths
SRC_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SRC_DIR.parent
DATA_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
IMAGES_DIR = PROJECT_ROOT / "images"

# Create directories if they don't exist
IMAGES_DIR.mkdir(parents=True, exist_ok=True)

def load_data():
    # Loads processed data, cluster labels, and scaler.
    print("Loading data for profiling")
    try:
        df_customers = pd.read_csv(DATA_DIR / 'customers_final.csv')
        df_clusters = pd.read_csv(DATA_DIR / 'clusters_FINAL.csv')
        with open(MODELS_DIR / 'scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        print("Data loaded successfully.")
        return df_customers, df_clusters, scaler
    except FileNotFoundError as e:
        print(f"Error loading data: {e}")
        return None, None, None

def inverse_transform_data(df, scaler):
    # Inverse transforms the scaled data to original values.
    print("Inverse transforming data")
    
    # Identify feature columns (exclude CustomerID if present)
    feature_cols = [col for col in df.columns if col != 'CustomerID']
    
    # Check if column count matches scaler
    if len(feature_cols) != scaler.n_features_in_:
        print(f"Error: Feature count mismatch. Data has {len(feature_cols)}, scaler expects {scaler.n_features_in_}.")
        # Try to match columns if possible, or fail
        return None

    X_scaled = df[feature_cols].values
    X_original = scaler.inverse_transform(X_scaled)
    
    df_original = pd.DataFrame(X_original, columns=feature_cols)
    if 'CustomerID' in df.columns:
        df_original['CustomerID'] = df['CustomerID']
        
    return df_original

def run_profiling():
    # Runs the customer profiling pipeline and replicates logic from 05_Profiling.ipynb
    print("Starting profiling")
    
    df_scaled, df_clusters, scaler = load_data()
    if df_scaled is None:
        return

    # Inverse Transform to get original values
    df_original = inverse_transform_data(df_scaled, scaler)
    if df_original is None:
        return

    # Merge with Cluster Labels and ensure CustomerID is key
    df_merged = df_original.merge(df_clusters[['CustomerID', 'Cluster']], on='CustomerID', how='inner')
    print(f"Merged data shape: {df_merged.shape}")
    
    # Save merged data
    output_merged_path = DATA_DIR / 'customers_with_clusters.csv'
    df_merged.to_csv(output_merged_path, index=False)
    print(f"Saved customers with clusters to {output_merged_path}")

    # Cluster Statistics (Profiles)
    # I calculate mean for key behavioral features and I focus on the core behavioral features for the summary
    behavior_cols = [
        'Recency', 'Frequency', 'Monetary', 
        'AvgBasketValue', 'BasketValueVariance', 'DistinctProducts'
    ]
    
    # Ensure these columns exist
    available_cols = [col for col in behavior_cols if col in df_merged.columns]
    
    cluster_profiles = df_merged.groupby('Cluster')[available_cols].mean().reset_index()
    cluster_counts = df_merged['Cluster'].value_counts().reset_index()
    cluster_counts.columns = ['Cluster', 'Count']
    
    cluster_profiles = cluster_profiles.merge(cluster_counts, on='Cluster')
    
    print("Cluster Profiles (Mean Values):")
    print(cluster_profiles)
    
    output_profile_path = DATA_DIR / 'cluster_profiles.csv'
    cluster_profiles.to_csv(output_profile_path, index=False)
    print(f"Saved cluster profiles to {output_profile_path}")

    # Visualizations
    # Box Plots for Key Features
    print("Generating box plots")
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    
    for i, col in enumerate(available_cols):
        if i < len(axes):
            sns.boxplot(x='Cluster', y=col, data=df_merged, hue='Cluster', ax=axes[i], palette="tab10", legend=False)
            axes[i].set_title(f'Distribution of {col} by Cluster')
            axes[i].set_ylabel(col)
            axes[i].set_xlabel('Cluster')
            
    plt.tight_layout()
    boxplot_path = IMAGES_DIR / 'cluster_boxplots.png'
    plt.savefig(boxplot_path)
    plt.close()
    print(f"Saved box plots to {boxplot_path}")

    # Relative Importance Heatmap
    print("Generating heatmap")
    # Calculate relative importance
    
    # Using the method from notebook: Relative deviation from population mean
    population_means = df_merged[available_cols].mean()
    relative_imp = (df_merged.groupby('Cluster')[available_cols].mean() - population_means) / population_means
    
    plt.figure(figsize=(10, 6))
    sns.heatmap(relative_imp, annot=True, fmt='.2f', cmap='RdBu_r', center=0)
    plt.title('Relative Importance of Attributes by Cluster\n(Deviation from Population Mean)')
    plt.tight_layout()
    
    heatmap_path = IMAGES_DIR / 'cluster_heatmap.png'
    plt.savefig(heatmap_path)
    plt.close()
    print(f"Saved heatmap to {heatmap_path}")

    print("Profiling completed successfully")

if __name__ == "__main__":
    run_profiling()

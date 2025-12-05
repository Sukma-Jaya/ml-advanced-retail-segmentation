import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from umap import UMAP
from pathlib import Path
import pickle
import os

# Paths
SRC_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SRC_DIR.parent
DATA_DIR = PROJECT_ROOT / "data" / "processed"
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
MODELS_DIR = PROJECT_ROOT / "models"
IMAGES_DIR = PROJECT_ROOT / "images"

# Create directories if they don't exist
IMAGES_DIR.mkdir(parents=True, exist_ok=True)

def load_data():
    # Loads the processed customer data and scaler.
    print("Loading data")
    try:
        df = pd.read_csv(DATA_DIR / 'customers_final.csv')
        with open(MODELS_DIR / 'scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        print(f"Data loaded. Shape: {df.shape}")
        return df, scaler
    except FileNotFoundError as e:
        print(f"Error loading data: {e}")
        return None, None

def run_preprocessing():
    # Runs the dimensionality reduction pipeline (UMAP) and replicates logic from 03_DimensionalityReduction.ipynb.
    print("Starting preprocessing (Dimensionality Reduction)")
    
    df, scaler = load_data()
    if df is None:
        return

    # Prepare data for UMAP    
    # Drop CustomerID to get the feature matrix X
    if 'CustomerID' in df.columns:
        customer_ids = df['CustomerID']
        X = df.drop(columns=['CustomerID'])
    else:
        print("Error: CustomerID column not found.")
        return

    print(f"Feature matrix shape: {X.shape}")

    # UMAP 10D for Clustering
    print("Running UMAP 10D")
    # Parameters from notebook
    umap_10d = UMAP(
        n_neighbors=30,
        min_dist=0.1,
        n_components=10,
        random_state=42,
        n_jobs=1
    )
    
    X_umap_10d = umap_10d.fit_transform(X)
    print(f"UMAP 10D embedding shape: {X_umap_10d.shape}")

    # Save UMAP 10D results
    df_umap_10d = pd.DataFrame(X_umap_10d, columns=[f'UMAP{i+1}' for i in range(X_umap_10d.shape[1])])
    df_umap_10d['CustomerID'] = customer_ids.values
    output_path_10d = DATA_DIR / 'umap_10d.csv'
    df_umap_10d.to_csv(output_path_10d, index=False)
    print(f"Saved UMAP 10D embeddings to {output_path_10d}")

    # UMAP 2D for Visualization
    print("Running UMAP 2D")
    # Parameters from notebook
    umap_2d = UMAP(
        n_neighbors=30,
        min_dist=0.1,
        n_components=2,
        random_state=42,
        n_jobs=1
    )
    
    X_umap_2d = umap_2d.fit_transform(X)
    print(f"UMAP 2D embedding shape: {X_umap_2d.shape}")

    # Save UMAP 2D results
    df_umap_2d = pd.DataFrame(X_umap_2d, columns=['UMAP1', 'UMAP2'])
    df_umap_2d['CustomerID'] = customer_ids.values
    output_path_2d = DATA_DIR / 'umap_2d.csv'
    df_umap_2d.to_csv(output_path_2d, index=False)
    print(f"Saved UMAP 2D embeddings to {output_path_2d}")

    # Generate and Save Visualization
    print("Generating UMAP 2D plot")
    plt.figure(figsize=(10, 8))
    plt.scatter(X_umap_2d[:, 0], X_umap_2d[:, 1], s=1, alpha=0.5)
    plt.title('UMAP Projection (2D)')
    plt.xlabel('UMAP1')
    plt.ylabel('UMAP2')
    plt.tight_layout()
    
    plot_path = IMAGES_DIR / 'umap_2d_projection.png'
    plt.savefig(plot_path)
    plt.close()
    print(f"Saved UMAP 2D plot to {plot_path}")

    print("Preprocessing completed successfully")

if __name__ == "__main__":
    run_preprocessing()

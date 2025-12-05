import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from pathlib import Path
import os

# Paths
SRC_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SRC_DIR.parent
DATA_DIR = PROJECT_ROOT / "data" / "processed"
IMAGES_DIR = PROJECT_ROOT / "images"

# Create directories if they don't exist
IMAGES_DIR.mkdir(parents=True, exist_ok=True)

def load_data():
    # Loads the UMAP embeddings.
    print("Loading UMAP data")
    try:
        df_10d = pd.read_csv(DATA_DIR / 'umap_10d.csv')
        df_2d = pd.read_csv(DATA_DIR / 'umap_2d.csv')
        print(f"Data loaded. 10D shape: {df_10d.shape}, 2D shape: {df_2d.shape}")
        return df_10d, df_2d
    except FileNotFoundError as e:
        print(f"Error loading data: {e}")
        return None, None

def run_clustering():

    print("Starting clustering")
    
    df_10d, df_2d = load_data()
    if df_10d is None or df_2d is None:
        return

    # Prepare data for clustering
    # Drop CustomerID to get the feature matrix X
    if 'CustomerID' in df_10d.columns:
        customer_ids = df_10d['CustomerID']
        X = df_10d.drop(columns=['CustomerID'])
    else:
        print("Error: CustomerID column not found in 10D data.")
        return

    # K-Means Parameters from notebook cell 56
    best_k = 4
    best_init = 'k-means++'
    best_ninit = 10
    best_maxiter = 300
    random_state = 42

    print(f"Training K-Means with k={best_k}")
    kmeans = KMeans(
        n_clusters=best_k,
        init=best_init,
        n_init=best_ninit,
        max_iter=best_maxiter,
        random_state=random_state
    )
    
    labels = kmeans.fit_predict(X)
    print("Clustering completed")

    # Save results
    output_df = pd.DataFrame({
        'CustomerID': customer_ids,
        'Cluster': labels,
        'Model': "K-Means"
    })
    
    output_path = DATA_DIR / 'clusters_FINAL.csv'
    output_df.to_csv(output_path, index=False)
    print(f"Cluster labels exported to {output_path}")

    # Visualization
    print("Generating cluster visualization")
    
    # Add labels to 2D dataframe for plotting
    df_plot = df_2d.copy()
    df_plot['Cluster'] = labels
    
    plt.figure(figsize=(8, 6))
    sns.scatterplot(
        data=df_plot,
        x="UMAP1", y="UMAP2",
        hue="Cluster",
        palette="tab10",
        s=25
    )
    plt.title(
        f"Optimized K-Means Clustering on UMAP 2D\n"
        f"(k={best_k}, init={best_init}, n_init={best_ninit}, max_iter={best_maxiter})"
    )
    plt.xlabel("UMAP1")
    plt.ylabel("UMAP2")
    plt.legend(title="Cluster")
    plt.tight_layout()
    
    plot_path = IMAGES_DIR / 'kmeans_final_optimized.png'
    plt.savefig(plot_path)
    plt.close()
    print(f"Saved cluster plot to {plot_path}")

    print("Clustering pipeline completed successfully")

if __name__ == "__main__":
    run_clustering()

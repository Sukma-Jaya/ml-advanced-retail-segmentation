import sys
from pathlib import Path
import time

# Add src to python path to allow imports
sys.path.append(str(Path(__file__).parent / "src"))

from src import data_prep
from src import preprocessing
from src import clustering
from src import profiling

def main():
    start_time = time.time()
    print("Starting Customer Segmentation Pipeline")

    # Data Preparation
    print("\n[Step 1/4] Data Preparation")
    try:
        data_prep.run_data_prep()
    except Exception as e:
        print(f"Error in Data Preparation: {e}")
        return

    # Preprocessing (Dimensionality Reduction)
    print("\n[Step 2/4] Preprocessing (Dimensionality Reduction)")
    try:
        preprocessing.run_preprocessing()
    except Exception as e:
        print(f"Error in Preprocessing: {e}")
        return

    # Clustering
    print("\n[Step 3/4] Clustering (K-Means)")
    try:
        clustering.run_clustering()
    except Exception as e:
        print(f"Error in Clustering: {e}")
        return

    # Profiling & Analysis
    print("\n[Step 4/4] Profiling & Analysis")
    try:
        profiling.run_profiling()
    except Exception as e:
        print(f"Error in Profiling: {e}")
        return

    end_time = time.time()
    duration = end_time - start_time
    print(f"Pipeline Completed Successfully in {duration:.2f} seconds")

if __name__ == "__main__":
    main()

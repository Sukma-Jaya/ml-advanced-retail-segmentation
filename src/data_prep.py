import pandas as pd
from pathlib import Path
import pickle
from sklearn.preprocessing import StandardScaler
import os

# Paths
SRC_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SRC_DIR.parent
DATA_DIR = PROJECT_ROOT / "data" / "processed"
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
MODELS_DIR = PROJECT_ROOT / "models"

# Create directories if they don't exist
DATA_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

def load_data(data_path):
    # Loads raw data and performs initial renaming and type conversion.
    print(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path)
    df = df.rename(columns={
        'Price': 'UnitPrice',
        'Customer ID': 'CustomerID'
    })
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
    df['TotalPrice'] = df['Quantity'] * df['UnitPrice']
    print(f"Raw shape: {df.shape}")
    return df

def clean_data(df):
    # Applies irreversible data cleaning rules.
    print("Cleaning data")
    
    # Remove rows with missing CustomerID
    before = len(df)
    df = df.dropna(subset=['CustomerID'])
    print(f"Removed {before - len(df)} rows without CustomerID")
    
    # Remove cancelled invoices
    before = len(df)
    cancel_mask = df['Invoice'].astype(str).str.startswith('C', na=False)
    df = df.loc[~cancel_mask].copy()
    print(f"Removed {before - len(df)} cancelled transactions")
    
    # Remove negative quantities or prices
    before = len(df)
    valid_mask = (df['Quantity'] > 0) & (df['UnitPrice'] > 0)
    df = df.loc[valid_mask].copy()
    print(f"Removed {before - len(df)} rows with non-positive values")
    
    # Drop exact duplicates
    before = len(df)
    df = df.drop_duplicates().copy()
    print(f"Removed {before - len(df)} duplicate rows")
    
    return df

def engineer_features(df):
    # Aggregates transactions to customer-level features
    print("Engineering features")
    
    # RFM Metrics
    snapshot_date = df['InvoiceDate'].max() + pd.Timedelta(days=1)
    rfm = (
        df.groupby('CustomerID')
        .agg({
            'InvoiceDate': 'max',
            'Invoice': 'nunique',
            'TotalPrice': 'sum'
        })
        .rename(columns={
            'InvoiceDate': 'LastPurchaseDate',
            'Invoice': 'Frequency',
            'TotalPrice': 'Monetary'
        })
    )
    rfm['Recency'] = (snapshot_date - rfm['LastPurchaseDate']).dt.days
    rfm = rfm.drop(columns=['LastPurchaseDate'])
    customer_features = rfm.copy()
    
    # Country One-Hot Encoding
    def most_frequent(series: pd.Series) -> str:
        mode = series.mode()
        return mode.iloc[0] if not mode.empty else series.iloc[0]

    customer_country = df.groupby('CustomerID')['Country'].agg(most_frequent)
    country_dummies = pd.get_dummies(customer_country, prefix='Country')
    customer_features = customer_features.join(country_dummies, how='left')
    
    # Basket Statistics
    invoice_summary = (
        df.groupby(['CustomerID', 'Invoice'])
        .agg(InvoiceRevenue=('TotalPrice', 'sum'), InvoiceQuantity=('Quantity', 'sum'))
        .reset_index()
    )
    avg_basket = invoice_summary.groupby('CustomerID')['InvoiceRevenue'].mean().rename('AvgBasketValue')
    var_basket = invoice_summary.groupby('CustomerID')['InvoiceRevenue'].var(ddof=0).rename('BasketValueVariance')
    distinct_products = df.groupby('CustomerID')['StockCode'].nunique().rename('DistinctProducts')
    
    basket_features = pd.concat([avg_basket, var_basket, distinct_products], axis=1)
    basket_features = basket_features.fillna({'BasketValueVariance': 0, 'AvgBasketValue': 0, 'DistinctProducts': 0})
    
    customer_features = customer_features.join(basket_features, how='left')
    
    # Fill any remaining NaNs in basket stats
    cols_to_fill = ['AvgBasketValue', 'BasketValueVariance', 'DistinctProducts']
    customer_features[cols_to_fill] = customer_features[cols_to_fill].fillna(0)
    
    customer_features = customer_features.sort_index()
    return customer_features

def scale_features(customer_features):
    # Standardizes features and saves the scaler.
    print("Scaling features")
    
    scaler = StandardScaler()
    scaled_matrix = scaler.fit_transform(customer_features)
    
    # Save scaler
    scaler_path = MODELS_DIR / 'scaler.pkl'
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    print(f"Saved scaler to {scaler_path}")
    
    # Create DataFrame with scaled values
    df_scaled = pd.DataFrame(
        scaled_matrix,
        index=customer_features.index,
        columns=customer_features.columns
    )
    
    # Reset index to make CustomerID a column
    df_final = df_scaled.reset_index().rename(columns={'index': 'CustomerID'})
    
    return df_final

def run_data_prep():    
    data_path = RAW_DATA_DIR / 'online_retail_II.csv'
    
    # Pipeline
    df = load_data(data_path)
    df_clean = clean_data(df)
    customer_features = engineer_features(df_clean)
    df_final = scale_features(customer_features)
    
    # Export
    output_path = DATA_DIR / 'customers_final.csv'
    df_final.to_csv(output_path, index=False)
    print(f"Saved processed data to {output_path}")
    
    return df_final

if __name__ == "__main__":
    run_data_prep()

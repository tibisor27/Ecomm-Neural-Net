import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import mlflow
import mlflow.sklearn
import json
from datetime import datetime
import hashlib

def log_data_profile(df: pd.DataFrame, stage: str):
    """Log comprehensive data profiling metrics to MLflow"""
    
    # Basic data info
    mlflow.log_param(f"{stage}_n_samples", len(df))
    mlflow.log_param(f"{stage}_n_features", len(df.columns))
    mlflow.log_param(f"{stage}_columns", list(df.columns))
    
    # Data quality metrics
    missing_counts = df.isnull().sum().to_dict()
    mlflow.log_metrics({f"{stage}_missing_{col}": count for col, count in missing_counts.items()})
    
    # Statistical summaries for numerical columns
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    for col in numerical_cols:
        if col in df.columns:
            mlflow.log_metric(f"{stage}_{col}_mean", df[col].mean())
            mlflow.log_metric(f"{stage}_{col}_std", df[col].std())
            mlflow.log_metric(f"{stage}_{col}_min", df[col].min())
            mlflow.log_metric(f"{stage}_{col}_max", df[col].max())
    
    # Data fingerprint for versioning
    data_hash = hashlib.md5(pd.util.hash_pandas_object(df, index=True).values).hexdigest()
    mlflow.log_param(f"{stage}_data_hash", data_hash)
    
    print(f"✅ Logged {stage} data profile: {len(df)} samples, {len(df.columns)} features")
    return data_hash

def clean_raw_data(file_path: str) -> pd.DataFrame:
    """Reads and cleans raw data, returning a cleaned DataFrame."""
    raw_data = pd.read_csv(file_path)
    
    # Fix FutureWarnings by avoiding inplace operations
    raw_data['time_spent'] = raw_data['time_spent'].fillna(raw_data['time_spent'].median())
    raw_data['pages_viewed'] = raw_data['pages_viewed'].fillna(raw_data['pages_viewed'].mean())
    raw_data['basket_value'] = raw_data['basket_value'].fillna(0)
    raw_data['device_type'] = raw_data['device_type'].fillna('Unknown')
    raw_data['customer_type'] = raw_data['customer_type'].fillna('New')

    # Ensure proper data types
    raw_data['customer_id'] = raw_data['customer_id'].astype(int)
    raw_data['time_spent'] = raw_data['time_spent'].astype(float)
    raw_data['pages_viewed'] = raw_data['pages_viewed'].astype(int)
    raw_data['basket_value'] = raw_data['basket_value'].astype(float)

    return raw_data

def prepare_model_features(file_path: str) -> pd.DataFrame: 
    """Prepares the dataset for the model: scales numerical features and encodes categorical features."""

    model_data = pd.read_csv(file_path)

    numerical_features = ['time_spent', 'pages_viewed', 'basket_value']
    scaler = MinMaxScaler()
    model_data[numerical_features] = scaler.fit_transform(model_data[numerical_features])

    categorical_features = ['device_type', 'customer_type']
    # Fix: Add dtype=int to get_dummies to ensure numeric output
    encoded = pd.get_dummies(model_data[categorical_features], prefix=categorical_features, dtype=int)

    model_feature_set = pd.concat([model_data.drop(columns=categorical_features), encoded], axis=1)
    return model_feature_set
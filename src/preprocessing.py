import pandas as pd
from sklearn.preprocessing import MinMaxScaler

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

    encoded = pd.get_dummies(model_data[categorical_features], prefix=categorical_features, dtype=int)

    model_feature_set = pd.concat([model_data.drop(columns=categorical_features), encoded], axis=1)
    return model_feature_set
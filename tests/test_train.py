import pytest
import pandas as pd
from src.train import train_model

class TestTrainModel:
    @pytest.fixture
    def train_data(self):
        return pd.DataFrame({
            'customer_id': [1, 2, 3],
            'time_spent': [10.0, 20.0, 30.0],
            'pages_viewed': [5, 10, 15],
            'basket_value': [100.0, 200.0, 300.0],
            'purchase': [1, 0, 1]
        })

    @pytest.fixture
    def val_data(self):
        return pd.DataFrame({
            'customer_id': [4, 5],
            'time_spent': [15.0, 25.0],
            'pages_viewed': [7, 12],
            'basket_value': [150.0, 250.0]
        })

    def test_train_model(self, train_data, val_data):
        train_csv = "train.csv"
        val_csv = "val.csv"
        train_data.to_csv(train_csv, index=False)
        val_data.to_csv(val_csv, index=False)

        model, predictions = train_model(train_csv, val_csv)

        assert model is not None
        assert isinstance(predictions, pd.DataFrame)
        assert 'customer_id' in predictions.columns
        assert 'purchase' in predictions.columns

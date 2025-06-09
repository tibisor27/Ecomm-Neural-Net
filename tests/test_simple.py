"""
Simple tests with CLASSES and FIXTURES
"""
import pytest
import pandas as pd
import torch
import os

# ==================== FIXTURES (Common data for tests) ====================

@pytest.fixture
def file_path():
    """Fixture that provides the path to the data file"""
    return "data/raw_customer_data.csv"

@pytest.fixture
def sample_data():
    """Fixture with simple test data"""
    return pd.DataFrame({
        'customer_id': [1, 2, 3, 4],
        'time_spent': [10.0, 20.0, 30.0, 40.0],
        'pages_viewed': [1, 5, 10, 2],
        'basket_value': [0.0, 100.0, 200.0, 50.0],
        'device_type': ['mobile', 'desktop', 'mobile', 'tablet'],
        'customer_type': ['new', 'returning', 'new', 'returning'],
        'purchase': [1, 0, 1, 0]
    })

@pytest.fixture
def model():
    """Fixture that provides a model for testing"""
    from src.model import PurchaseModel
    return PurchaseModel(input_size=5)

# ==================== CLASS 1: Data Tests ====================

class TestData:
    """All tests related to data"""
    
    def test_data_file_exists(self, file_path):
        """Check that the data file exists"""
        assert os.path.exists(file_path), f"Cannot find file {file_path}"
        print(f"File {file_path} exists")
    
    def test_data_can_be_loaded(self, file_path):
        """Check that we can read the data"""
        data = pd.read_csv(file_path)
        assert len(data) > 0, "File is empty"
        print(f"Successfully read {len(data)} rows of data")
    
    def test_data_has_important_columns(self, file_path):
        """Check that we have the important columns"""
        data = pd.read_csv(file_path)
        
        important_columns = ['customer_id', 'purchase', 'time_spent', 'basket_value']
        
        for column in important_columns:
            assert column in data.columns, f"Missing column {column}"
        
        print("All important columns are present")
    
    def test_no_missing_values_in_important_columns(self, file_path):
        """Check that we have no missing values in important columns"""
        data = pd.read_csv(file_path)
        
        important_columns = ['customer_id', 'purchase']
        
        for column in important_columns:
            if column in data.columns:
                missing_values = data[column].isnull().sum()
                assert missing_values == 0, f"Column {column} has {missing_values} missing values"
        
        print("No missing values in important columns")
    
    def test_purchase_values_are_valid(self, file_path):
        """Check that purchase values are 0 or 1"""
        data = pd.read_csv(file_path)
        
        if 'purchase' in data.columns:
            unique_values = data['purchase'].unique()
            for value in unique_values:
                assert value in [0, 1], f"Purchase value {value} is not valid. Must be 0 or 1"
        
        print("All purchase values are correct (0 or 1)")

# ==================== CLASS 2: Model Tests ====================

class TestModel:
    """All tests related to model"""
    
    def test_model_can_be_created(self, model):
        """Check that the model can be created"""
        assert model is not None, "Model was not created"
        print("Model can be created successfully")
    
    def test_model_can_predict(self, model):
        """Check that the model can make predictions"""
        # Test data
        test_data = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0]], dtype=torch.float32)
        
        # Try to make a prediction
        prediction = model(test_data)
        
        # Check that prediction looks good
        assert prediction.shape == (1, 1), "Prediction shape is wrong"
        assert 0 <= prediction.item() <= 1, "Prediction must be between 0 and 1"
        
        print(f"Model predicted: {prediction.item():.3f}")
    
    def test_model_with_multiple_samples(self, model):
        """Check that model can predict for multiple examples at once"""
        # 3 test examples
        test_data = torch.tensor([
            [1.0, 2.0, 3.0, 4.0, 5.0],
            [2.0, 3.0, 4.0, 5.0, 6.0], 
            [3.0, 4.0, 5.0, 6.0, 7.0]
        ], dtype=torch.float32)
        
        predictions = model(test_data)
        
        # Check shape and values
        assert predictions.shape == (3, 1), "Predictions shape is wrong for multiple examples"
        
        for i, prediction in enumerate(predictions):
            assert 0 <= prediction.item() <= 1, f"Prediction {i} is not between 0 and 1"
        
        print(f"Model predicted for 3 examples: {[f'{p.item():.3f}' for p in predictions]}")

# ==================== CLASS 3: Preprocessing Tests ====================

class TestPreprocessing:
    """All tests related to preprocessing"""
    
    def test_preprocessing_works(self, file_path):
        """Check that preprocessing works"""
        from src.preprocessing import clean_raw_data
        
        # Try to clean the data
        clean_data = clean_raw_data(file_path)
        
        # Check that something came out
        assert clean_data is not None, "Cleaning returned nothing"
        assert len(clean_data) > 0, "Clean data is empty"
        
        print(f"Preprocessing cleaned {len(clean_data)} rows")
    
    def test_sample_data_preprocessing(self, sample_data):
        """Test with sample data from fixture"""
        # Check that sample_data looks good
        assert len(sample_data) == 4, "Sample data must have 4 rows"
        assert 'purchase' in sample_data.columns, "Sample data must have purchase column"
        
        # Check that values are OK
        assert sample_data['purchase'].isin([0, 1]).all(), "Purchase must be 0 or 1"
        
        print("Sample data from fixture is correct")

# ==================== CLASS 4: Simple Integration Tests ====================

class TestIntegration:
    """Simple tests that check everything works together"""
    
    def test_full_pipeline_runs(self, file_path, model):
        """Complete test: read data -> preprocessing -> prediction"""
        from src.preprocessing import clean_raw_data
        
        # Step 1: Read and clean data
        clean_data = clean_raw_data(file_path)
        assert clean_data is not None, "Preprocessing failed"
        
        # Step 2: Take first data for test
        if len(clean_data) > 0:
            # Simulate that we prepared data for model (5 features)
            first_example = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0]], dtype=torch.float32)
            
            # Step 3: Make prediction
            prediction = model(first_example)
            
            # Check that everything went well
            assert prediction is not None, "Prediction failed"
            assert 0 <= prediction.item() <= 1, "Prediction is not valid"
            
            print(f"Complete pipeline works! Prediction: {prediction.item():.3f}")

# ==================== Function for quick execution ====================

def run_simple_tests():
    """Simple function to run tests without pytest"""
    print("Running simple tests...\n")
    
    # Run only a few basic tests
    try:
        # Simple test for file
        assert os.path.exists("data/raw_customer_data.csv"), "File does not exist"
        print("File exists")
        
        # Simple test for model
        from src.model import PurchaseModel
        model = PurchaseModel(input_size=5)
        test_data = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0]], dtype=torch.float32)
        prediction = model(test_data)
        assert 0 <= prediction.item() <= 1, "Prediction is not valid"
        print(f"Model works: {prediction.item():.3f}")
        
        print("\nBasic tests passed!")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    run_simple_tests() 
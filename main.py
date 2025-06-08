from src.preprocessing import clean_raw_data, prepare_model_features
from src.train import train_model
from src.experiment_tracker import create_tracker
import torch
import pandas as pd

def main():
    tracker = create_tracker(log_file="logs/experiments_log.csv")
    
    # Task 1 - Clean raw data
    clean_data = clean_raw_data("data/raw_customer_data.csv")
    clean_data.to_csv("data/clean_customer_data.csv", index=False)
    print("Task 1 - Data cleaning completed.")

    # Task 2 - Prepare features for the model
    model_features = prepare_model_features("data/clean_customer_data.csv")
    model_features.to_csv("data/prepared_model_features.csv", index=False)
    print("Task 2 - Feature preparation completed.")

    # Task 3 - Train model with tracking
    experiment_name = f"baseline_experiment_{pd.Timestamp.now().strftime('%Y-%m-%d_%H-%M')}"
    purchase_model, results = train_model(
        data_file="data/prepared_model_features.csv",
        experiment_name=experiment_name,
        tracker=tracker
    )
    results.to_csv("outputs/test_predictions.csv", index=False)
    
    # Save the trained model for future predictions
    torch.save(purchase_model.state_dict(), "models/final/trained_model.pth")
    print("Task 3 - Model trained and tested successfully!")
    print("Model saved as 'models/final/trained_model.pth'")
    
    # View all experiments
    print("\n" + "="*50)
    tracker.view_all_experiments()

if __name__ == "__main__":
    main()

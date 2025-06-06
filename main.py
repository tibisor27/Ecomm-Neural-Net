from src.preprocessing import clean_raw_data, prepare_model_features
from src.train import train_model
import torch

def main():
    # Task 1 - Clean raw data
    clean_data = clean_raw_data("data/raw_customer_data.csv")
    clean_data.to_csv("data/clean_customer_data.csv", index=False)
    print("Task 1 - Data cleaning completed.")

    # Task 2 - Prepare features for the model
    model_features = prepare_model_features("data/clean_customer_data.csv")
    model_features.to_csv("data/prepared_model_features.csv", index=False)
    print("Task 2 - Feature preparation completed.")

    # Task 3 - Train model with internal 80/20 split
    purchase_model, results = train_model("data/prepared_model_features.csv")
    results.to_csv("outputs/test_predictions.csv", index=False)
    
    # Save the trained model for future predictions
    torch.save(purchase_model.state_dict(), "models/final/trained_model.pth")
    print("Task 3 - Model trained and tested successfully!")
    print("Model saved as 'models/final/trained_model.pth'")

if __name__ == "__main__":
    main()

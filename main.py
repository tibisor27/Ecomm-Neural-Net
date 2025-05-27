from src.preprocessing import clean_raw_data, prepare_model_features
from src.train import train_model

def main():
    # Task 1 - Clean raw data
    clean_data = clean_raw_data("data/raw_customer_data.csv")
    clean_data.to_csv("data/clean_customer_data.csv", index=False)
    print("Task 1 - Data cleaning completed.")

    # Task 2 - Prepare features for the model
    model_features = prepare_model_features("data/model_data.csv")
    model_features.to_csv("data/prepared_model_features.csv", index=False)
    print("Task 2 - Feature preparation completed.")

    # Task 3 - Train model and make validation predictions
    purchase_model, validation_preds = train_model("data/input_model_features.csv", "data/validation_features.csv")
    validation_preds.to_csv("data/validation_predictions.csv", index=False)
    print("Task 3 - Model trained and predictions saved.")

if __name__ == "__main__":
    main()

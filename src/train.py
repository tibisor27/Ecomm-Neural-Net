import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from .model import PurchaseModel

def train_model(data_file):
    """Trains model on single dataset with internal 80/20 train/test split"""
    # Load the complete dataset
    data = pd.read_csv(data_file)
    features = data.drop(columns=['customer_id', 'purchase'])
    targets = data['purchase']
    customer_ids = data['customer_id']
    
    # Split: 80% train, 20% test
    X_train, X_test, y_train, y_test, ids_train, ids_test = train_test_split(
        features, targets, customer_ids, test_size=0.20, random_state=42, stratify=targets
    )
    
    # Initialize model
    input_size = X_train.shape[1]
    model = PurchaseModel(input_size)
    
    # Convert to tensors
    X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).unsqueeze(1)
    X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32)
    
    # Training setup
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    
    print(f"Training on {len(X_train)} samples (80%), testing on {len(X_test)} samples (20%)")
    
    # Training loop
    for epoch in range(100):
        model.train()
        train_loss = 0
        for features, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        # Print progress every 20 epochs
        if (epoch + 1) % 20 == 0:
            avg_loss = train_loss / len(train_loader)
            print(f"Epoch {epoch+1}/100 - Average Loss: {avg_loss:.4f}")
    
    # Final test evaluation
    model.eval()
    with torch.no_grad():
        test_predictions = model(X_test_tensor).squeeze().numpy()
        test_binary_preds = (test_predictions > 0.5).astype(int)
    
    # Calculate test metrics
    test_accuracy = accuracy_score(y_test, test_binary_preds)
    test_precision = precision_score(y_test, test_binary_preds)
    test_recall = recall_score(y_test, test_binary_preds)
    test_f1 = f1_score(y_test, test_binary_preds)
    
    print(f"\n=== FINAL TEST RESULTS ===")
    print(f"Test Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.1f}%)")
    print(f"Test Precision: {test_precision:.4f}")
    print(f"Test Recall: {test_recall:.4f}")
    print(f"Test F1-Score: {test_f1:.4f}")
    
    # Prepare test results
    test_results = pd.DataFrame({
        'customer_id': ids_test.values,
        'actual_purchase': y_test.values,
        'predicted_probability': test_predictions,
        'predicted_purchase': test_binary_preds
    })
    
    return model, test_results
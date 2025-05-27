import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from .model import PurchaseModel
import numpy as np

def train_model(train_csv: str, val_csv: str, epochs: int = 10, batch_size: int = 32, lr: float = 0.01):
    """
    Trains a neural network model to predict customer purchases and returns the trained model 
    along with purchase predictions for the validation dataset.
    """
    train_df = pd.read_csv(train_csv)
    X_train = train_df.drop(columns=['customer_id', 'purchase']).values
    y_train = train_df['purchase'].values

    test_df = pd.read_csv(val_csv)
    X_test = test_df.drop(columns=['customer_id']).values

    # Tensor conversion
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    input_size = X_train.shape[1]

    model = PurchaseModel(input_size)
    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for features, labels in dataloader:
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs} - Loss: {running_loss/len(dataloader):.4f}")

    model.eval()
    with torch.no_grad():
        val_preds = model(X_test_tensor).squeeze().numpy()

    validation_predictions = pd.DataFrame({
        'customer_id': test_df['customer_id'].values,
        'purchase': val_preds
    })
    return model, validation_predictions
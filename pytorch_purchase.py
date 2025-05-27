

import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn 
from torch.utils.data import DataLoader, TensorDataset

raw_data = pd.read_csv("raw_customer_data.csv")
# print(f"Before cleaning: {raw_data['time_spent']}")
raw_data['time_spent'].fillna(raw_data['time_spent'].median(), inplace = True)
# print(f"After cleaning: {raw_data['time_spent']}")

# print(f"Before cleaning: {raw_data['pages_viewed']}")
raw_data['pages_viewed'].fillna(raw_data['pages_viewed'].mean(), inplace = True)
raw_data['pages_viewed'] = raw_data['pages_viewed'].astype(int)
# print(f"After cleaning{raw_data['pages_viewed']}")

# print(f"Before cleaning: {raw_data['basket_value']}")
raw_data['basket_value'].fillna(0, inplace = True)
# print(f"After cleaning: {raw_data['basket_value']}")

# print(f"Before cleaning: {raw_data['device_type']}")
raw_data['device_type'].fillna('Unknown', inplace = True)
# print(f"After cleaning: {raw_data['device_type']}")

# print(f"Before cleaning: {raw_data['customer_type']}")
raw_data['customer_type'].fillna('New', inplace = True)
# print(f"After cleaning: {raw_data['customer_type']}")

# Ensure proper data types
raw_data['customer_id'] = raw_data['customer_id'].astype(int)
raw_data['time_spent'] = raw_data['time_spent'].astype(float)
raw_data['pages_viewed'] = raw_data['pages_viewed'].astype(int)
raw_data['basket_value'] = raw_data['basket_value'].astype(float)

clean_data = raw_data

# Write your answer to Task 2 here

model_data = pd.read_csv("model_data.csv")
scaler = MinMaxScaler()
numerical_features = ['time_spent', 'pages_viewed', 'basket_value']
model_data[numerical_features] = scaler.fit_transform(model_data[numerical_features])

categorical_features = ['device_type', 'customer_type']
encoded = pd.get_dummies(model_data[categorical_features], prefix=categorical_features)
# print(model_data)

#Creating the final dataset
model_feature_set = pd.concat([model_data.drop(columns=categorical_features), encoded], axis=1)
print(model_feature_set)

# Write your answer to Task 3 here

train_df = pd.read_csv("input_model_features.csv")
X_train = train_df.drop(columns=['customer_id', 'purchase']).values  #has to be numpy arr for future tenso
# print(X_train)
y_train = train_df['purchase'].values  #has to be numpy for future tensors
# print(y_train)

test_df = pd.read_csv("validation_features.csv")
X_test = test_df.drop(columns=['customer_id']).values
# print(f"Training set: {X_test}")

X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)

train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
dataloader = DataLoader(train_dataset, batch_size = 32, shuffle = True)
input_size = X_train.shape[1]

class PurchaseModel(nn.Module):
    def __init__(self,input_size):
        super(PurchaseModel,self).__init__()
        self.hidden = nn.Linear(input_size, 8)
        self.relu = nn.ReLU()
        self.output = nn.Linear(8, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self,x):
        x = self.hidden(x)
        x = self.relu(x)
        x = self.output(x)
        x = self.sigmoid(x)
        return x

purchase_model = PurchaseModel(input_size)

criterion = nn.BCELoss()
optimizer = torch.optim.Adam(purchase_model.parameters(), lr = 0.01)

for epoch in range(10):
    running_loss = 0.0
    for feature, label in dataloader:
        optimizer.zero_grad()
        output = purchase_model(feature)
        loss = criterion(output, label)
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

purchase_model.eval()
with torch.no_grad():
    val_preds = purchase_model(X_test_tensor).squeeze().numpy()
    
validation_predictions = pd.DataFrame({
    'customer_id': test_df['customer_id'].values,
    'purchase': val_preds
})
print(validation_predictions)
    

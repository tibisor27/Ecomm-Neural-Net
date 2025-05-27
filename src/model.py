import torch
import torch.nn as nn

class PurchaseModel(nn.Module):
    def __init__(self, input_size: int):
        super(PurchaseModel, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)
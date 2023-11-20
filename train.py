from typing import List 
import torch.nn as nn
import torch.optim as optim

class BushfireModel(nn.Module): 
    def __init__(self, n: int, hidden_layers: List[int]): 
        super().__init__() #Inherit from the nn.module
        self.n = n # n input features
        self.hidden_layers = hidden_layers # hidden layers are the middle ones

        self.layers = nn.ModuleList([
            nn.Linear(in_dim, out_dim)
            for in_dim, out_dim in zip([n, *hidden_layers], [*hidden_layers, 1])
        ]) # Single output

    def forward(self, x: torch.tensor):
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i != len(self.layers) - 1:    #We don't want to ReLU the last layer
                x = nn.functional.relu(x)  
        return x # Linear output for regression

from torch.utils.data import DataLoader
def train_model(model, train_loader, loss_criterion, optimizer, num_epochs):
    for epoch in range(num_epochs):
        for i, (features, labels) in enumerate(train_loader):
            # Forward pass
            outputs = model(features)
            loss = loss_criterion(outputs, labels.unsqueeze(1))

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i + 1) % 100 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}')
model = BushfireModel(n=11, hidden_layers=[1000,700])
loss_criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.1)
train_loader = DataLoader(dataset, batch_size=512, shuffle=True)
num_epochs = 8

import ssl
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import RobustScaler
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim

ssl._create_default_https_context = ssl._create_unverified_context

x, y = fetch_california_housing(as_frame=False, return_X_y=True)

x_train_val, x_test, y_train_val, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
x_train, x_val, y_train, y_val = train_test_split(x_train_val, y_train_val, test_size=0.25, random_state=42)

if __name__ == "__main__":
    print(f"Zbiór treningowy (Train): {x_train.shape[0]} wierszy")
    print(f"Zbiór walidacyjny (Validation): {x_val.shape[0]} wierszy")
    print(f"Zbiór testowy (Test): {x_test.shape[0]} wierszy")

scaler = RobustScaler()

x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)
x_val_scaled = scaler.transform(x_val)

x_train_tensor = torch.tensor(x_train_scaled, dtype=torch.float32)
x_val_tensor = torch.tensor(x_val_scaled, dtype=torch.float32)
x_test_tensor = torch.tensor(x_test_scaled, dtype=torch.float32)

y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
y_val_tensor = torch.tensor(y_val, dtype=torch.float32).unsqueeze(1)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

class PropertyValuationDataset(torch.utils.data.Dataset):
    def __init__(self, features, targets):
        self.features = features
        self.targets = targets

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]

train_dataset = PropertyValuationDataset(x_train_tensor, y_train_tensor)
val_dataset = PropertyValuationDataset(x_val_tensor, y_val_tensor)
test_dataset = PropertyValuationDataset(x_test_tensor, y_test_tensor)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

class PropertyValuationModel(torch.nn.Module):
    def __init__(self, input_dim):
        super(PropertyValuationModel, self).__init__()
        self.layer1 = torch.nn.Linear(input_dim, 64)
        self.relu1 = torch.nn.ReLU()
        self.layer2 = torch.nn.Linear(64, 32)
        self.relu2 = torch.nn.ReLU()
        self.output_layer = torch.nn.Linear(32, 1)

    def forward(self, x):
        x = self.layer1(x)
        x = self.relu1(x)
        x = self.layer2(x)
        x = self.relu2(x)
        x = self.output_layer(x)
        return x

model = PropertyValuationModel(input_dim=x_train_tensor.shape[1])
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_episodes = 250
train_losses = []
val_losses = []

if __name__ == "__main__":
    for episode in range(num_episodes):
        model.train()
        running_train_loss = 0
        for features, targets in train_loader:
            optimizer.zero_grad()
            predictions = model(features)
            loss = criterion(predictions, targets)
            loss.backward()
            optimizer.step()
            running_train_loss += loss.item() * features.size(0)

        episode_train_loss = running_train_loss / len(train_loader.dataset)
        train_losses.append(episode_train_loss)

        model.eval()
        running_val_loss = 0
        with torch.no_grad():
            for features, targets in val_loader:
                predictions = model(features)
                loss = criterion(predictions, targets)
                running_val_loss += loss.item() * features.size(0)

        episode_val_loss = running_val_loss / len(val_loader.dataset)
        val_losses.append(episode_val_loss)

        print(f"Episode {episode + 1}/{num_episodes}, Train Loss: {episode_train_loss:.4f}, Validation Loss: {episode_val_loss:.4f}")

    torch.save(model.state_dict(), "property_valuation_model.pth")
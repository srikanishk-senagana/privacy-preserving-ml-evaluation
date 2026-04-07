# src/train_baseline_opacus.py

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from preprocessing import load_adult_data  # relative import

# --- Hyperparameters ---
BATCH_SIZE = 64
LR = 0.01
EPOCHS = 30
SEED = 42

# --- Reproducibility ---
torch.manual_seed(SEED)
np.random.seed(SEED)

# --- Data Loading ---
X_train, X_test, y_train, y_test = load_adult_data()
# Convert to PyTorch tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)
X_test  = torch.tensor(X_test,  dtype=torch.float32)
y_test  = torch.tensor(y_test,  dtype=torch.long)

train_ds = TensorDataset(X_train, y_train)
test_ds  = TensorDataset(X_test,  y_test)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False)

# --- Model Definition ---
class MLP(nn.Module):
    def __init__(self, input_dim=108):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        return self.net(x)

model = MLP(input_dim=X_train.shape[1])
criterion = nn.BCELoss()
optimizer = optim.SGD(model.parameters(), lr=LR, momentum=0.9)

# --- Training Loop ---
for epoch in range(1, EPOCHS + 1):
    model.train()
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        preds = model(X_batch).squeeze()
        loss = criterion(preds, y_batch.float())
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch:02d}/{EPOCHS} — Loss: {loss.item():.4f}")

# --- Evaluation ---
model.eval()
all_preds, all_labels = [], []
with torch.no_grad():
    for X_batch, y_batch in test_loader:
        preds = model(X_batch).squeeze().round().long()
        all_preds.append(preds.numpy())
        all_labels.append(y_batch.numpy())

y_pred = np.concatenate(all_preds)
y_true = np.concatenate(all_labels)

acc = accuracy_score(y_true, y_pred)
f1  = f1_score(y_true, y_pred)
print(f"\nBaseline Non-DP Model — Test Accuracy: {acc*100:.2f}%, F1: {f1:.4f}")

# --- Save Model & Metrics ---
os.makedirs('../models', exist_ok=True)
torch.save(model.state_dict(), '../models/baseline_opacus.pt')
with open('../results/baseline_metrics.txt', 'w') as f:
    f.write(f"accuracy: {acc:.4f}\nf1: {f1:.4f}\n")

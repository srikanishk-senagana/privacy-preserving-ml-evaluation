# src/train_opacus_dp.py

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from opacus import PrivacyEngine
from sklearn.metrics import accuracy_score, f1_score
from preprocessing import load_adult_data  # assumes it's in src/

# --- Hyperparameters ---
BATCH_SIZE = 64
LR         = 0.01
EPOCHS     = 30
DELTA      = 1e-5
SEED       = 42

# --- Privacy budgets to test ---
EPSILONS = [1, 2, 5, 10]

# --- Reproducibility ---
torch.manual_seed(SEED)
np.random.seed(SEED)

# --- Load Data ---
X_train, X_test, y_train, y_test = load_adult_data()
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)
X_test  = torch.tensor(X_test,  dtype=torch.float32)
y_test  = torch.tensor(y_test,  dtype=torch.long)

train_ds = TensorDataset(X_train, y_train)
test_ds  = TensorDataset(X_test,  y_test)

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

# --- Prepare output directories ---
os.makedirs('../models', exist_ok=True)
os.makedirs('../results', exist_ok=True)

# --- Loop over privacy budgets ---
for eps in EPSILONS:
    print(f"\n=== Training with ε = {eps} ===")
    # Initialize model, loss, and optimizer
    model     = MLP(input_dim=X_train.shape[1])
    criterion = nn.BCELoss()
    optimizer = optim.SGD(model.parameters(), lr=LR, momentum=0.9)

    # Attach PrivacyEngine
    privacy_engine = PrivacyEngine()
    model, optimizer, train_loader = privacy_engine.make_private_with_epsilon(
        module=model,
        optimizer=optimizer,
        data_loader=DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True),
        target_epsilon=eps,
        target_delta=DELTA,
        epochs=EPOCHS,
        max_grad_norm=1.0,
    )

    # Training loop
    for epoch in range(1, EPOCHS + 1):
        model.train()
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            out = model(X_batch).squeeze()
            loss = criterion(out, y_batch.float())
            loss.backward()
            optimizer.step()
        if epoch % 10 == 0:
            print(f"Epoch {epoch} — Loss: {loss:.4f}")

    # Evaluation on test set
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for X_batch, y_batch in DataLoader(test_ds, batch_size=BATCH_SIZE):
            preds = model(X_batch).squeeze().round().long()
            all_preds.append(preds.numpy())
            all_labels.append(y_batch.numpy())

    y_pred = np.concatenate(all_preds)
    y_true = np.concatenate(all_labels)
    acc    = accuracy_score(y_true, y_pred)
    f1     = f1_score(y_true, y_pred)
    print(f"ε={eps} — Test Accuracy: {acc*100:.2f}%, F1: {f1:.4f}")

    # Save model and metrics
    model_path   = f'../models/opacus_eps{eps}.pt'
    metrics_path = f'../results/metrics_opacus_eps{eps}.txt'
    torch.save(model.state_dict(), model_path)
    with open(metrics_path, 'w') as f:
        f.write(f"epsilon: {eps}\naccuracy: {acc:.4f}\nf1: {f1:.4f}\n")

# src/mia_shadow.py

import os
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import TensorDataset, DataLoader, Subset
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from train_baseline_opacus import MLP, BATCH_SIZE, LR, EPOCHS, SEED
from preprocessing import load_adult_data

RESULT_DIR = '../results'
NUM_SHADOWS = 5

def train_shadow_models(num_shadows=NUM_SHADOWS):
    # Load the full training set
    X_train, _, y_train, _ = load_adult_data()
    X = torch.tensor(X_train, dtype=torch.float32)
    y = torch.tensor(y_train, dtype=torch.float32)
    dataset = TensorDataset(X, y)

    shadow_scores = []
    shadow_labels = []

    for i in range(num_shadows):
        rng = np.random.RandomState(SEED + i)
        # Split half of data for shadow i
        indices = rng.choice(len(dataset), size=len(dataset)//2, replace=False)
        shadow_ds = Subset(dataset, indices)
        loader = DataLoader(shadow_ds, batch_size=BATCH_SIZE, shuffle=True)

        # Train shadow model (non-private)
        model = MLP(input_dim=X.shape[1])
        optimizer = torch.optim.SGD(model.parameters(), lr=LR, momentum=0.9)
        criterion = nn.BCELoss()
        model.train()
        for _ in range(EPOCHS):
            for xb, yb in loader:
                optimizer.zero_grad()
                loss = criterion(model(xb).squeeze(), yb)
                loss.backward()
                optimizer.step()

        # Collect member scores
        model.eval()
        with torch.no_grad():
            member_scores = model(X[indices]).squeeze().numpy()
        shadow_scores.append(member_scores)
        shadow_labels.append(np.ones_like(member_scores))

        # Collect non-member scores from the other half
        non_indices = np.setdiff1d(np.arange(len(dataset)), indices)
        non_sample = rng.choice(non_indices, size=len(indices), replace=False)
        with torch.no_grad():
            non_scores = model(X[non_sample]).squeeze().numpy()
        shadow_scores.append(non_scores)
        shadow_labels.append(np.zeros_like(non_scores))

    return np.concatenate(shadow_scores), np.concatenate(shadow_labels)

def run_shadow_attack(model_filename='baseline_opacus.pt'):
    # Step 1: train shadow models and get their member/non-member scores
    scores, labels = train_shadow_models()
    scores = scores.reshape(-1, 1)

    # Step 2: train attacker (logistic regression)
    attacker = LogisticRegression(solver='lbfgs', max_iter=1000)
    attacker.fit(scores, labels)

    # Step 3: score the real target model
    X_train, X_test, _, _ = load_adult_data()
    X_all = torch.tensor(np.concatenate([X_train, X_test]), dtype=torch.float32)
    real_labels = np.concatenate([np.ones(len(X_train)), np.zeros(len(X_test))])

    # Load and strip prefix from the baseline model
    state = torch.load(f'../models/{model_filename}', map_location='cpu')
    stripped = {k.replace("_module.", ""): v for k, v in state.items()}
    target_model = MLP(input_dim=X_all.shape[1])
    target_model.load_state_dict(stripped)
    target_model.eval()

    with torch.no_grad():
        real_scores = target_model(X_all).squeeze().numpy().reshape(-1,1)

    # Step 4: evaluate attacker
    preds = attacker.predict(real_scores)
    acc = accuracy_score(real_labels, preds)
    auc = roc_auc_score(real_labels, real_scores)

    print(f"Shadow Attack on {model_filename} → Acc: {acc:.3f}, AUC: {auc:.3f}")

    # Save results
    os.makedirs(RESULT_DIR, exist_ok=True)
    with open(f"{RESULT_DIR}/shadow_{model_filename.replace('.pt','')}.txt", 'w') as f:
        f.write(f"accuracy: {acc:.4f}\nauc: {auc:.4f}\n")

if __name__ == "__main__":
    # Attack each model in the models directory
    for fname in sorted(os.listdir('../models')):
        if not fname.endswith('.pt'):
            continue
        print(f"\n--- Shadow attack on {fname} ---")
        run_shadow_attack(fname)

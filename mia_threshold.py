# src/mia_threshold.py

import os
import torch
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score
from preprocessing import load_adult_data
from train_baseline_opacus import MLP  # imports your model class

# --- Settings ---
MODEL_DIR   = '../models'
RESULT_DIR  = '../results'
THRESHOLD   = 0.5
# We can auto-discover .pt files instead of hard-coding:
# EPSILONS    = ['baseline', 'eps1', 'eps2', 'eps5', 'eps10']

def threshold_attack(model_path, tag):
    # Load data
    X_train, X_test, _, _ = load_adult_data()
    X_train = torch.tensor(X_train, dtype=torch.float32)
    X_test  = torch.tensor(X_test,  dtype=torch.float32)

    # Load model
    model = MLP(input_dim=X_train.shape[1])
    # Strip "_module." prefix from keys saved by Opacus
    raw_state = torch.load(model_path, map_location='cpu')
    new_state = {}
    for k, v in raw_state.items():
        new_key = k.replace("_module.", "") if k.startswith("_module.") else k
        new_state[new_key] = v
    model.load_state_dict(new_state)
    model.eval()

    # Get scores
    with torch.no_grad():
        train_scores = model(X_train).squeeze().numpy()
        test_scores  = model(X_test).squeeze().numpy()

    # Build labels and predictions
    scores_all = np.concatenate([train_scores, test_scores])
    labels_all = np.concatenate([np.ones_like(train_scores), np.zeros_like(test_scores)])
    preds_all  = (scores_all > THRESHOLD).astype(int)

    # Metrics
    acc = accuracy_score(labels_all, preds_all)
    auc = roc_auc_score(labels_all, scores_all)
    adv = acc - 0.5

    # Report & save
    print(f"[{tag}] Threshold Attack → Acc: {acc:.3f}, AUC: {auc:.3f}, Adv: {adv:.3f}")
    os.makedirs(RESULT_DIR, exist_ok=True)
    with open(f"{RESULT_DIR}/threshold_{tag}.txt", 'w') as f:
        f.write(f"accuracy: {acc:.4f}\nauc: {auc:.4f}\nadvantage: {adv:.4f}\n")

if __name__ == "__main__":
    # Auto-discover all .pt files in the models directory
    model_files = sorted(f for f in os.listdir(MODEL_DIR) if f.endswith('.pt'))
    if not model_files:
        raise RuntimeError(f"No .pt files found in {MODEL_DIR}")
    for fname in model_files:
        tag = os.path.splitext(fname)[0]  # e.g. "baseline_opacus" or "opacus_eps1"
        path = os.path.join(MODEL_DIR, fname)
        threshold_attack(path, tag)

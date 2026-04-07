# src/mia_threshold_tf.py

import os
import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score, roc_auc_score
from preprocessing import load_adult_data

MODEL_DIR  = '../models'
RESULT_DIR = '../results'
THRESHOLD  = 0.5
BATCH_SIZE = 64

def load_tf_model(path):
    """
    Load a Keras .h5 model for inference only (compile=False skips optimizer deserialization).
    """
    if not os.path.exists(path):
        return None
    return tf.keras.models.load_model(path, compile=False)

def threshold_attack(model_path, tag):
    # 1) Load data
    X_train, X_test, _, _ = load_adult_data()

    # 2) Load the TFP‐trained model
    model = load_tf_model(model_path)
    if model is None:
        print(f"[{tag}] Model file not found: {model_path}")
        return

    # 3) Obtain confidence scores
    train_scores = model.predict(X_train, batch_size=BATCH_SIZE).flatten()
    test_scores  = model.predict(X_test,  batch_size=BATCH_SIZE).flatten()

    # 4) Build labels (1 for members, 0 for non‐members) and binary predictions
    scores_all = np.concatenate([train_scores, test_scores])
    labels_all = np.concatenate([np.ones_like(train_scores),
                                 np.zeros_like(test_scores)])
    preds_all  = (scores_all > THRESHOLD).astype(int)

    # 5) Compute metrics
    acc = accuracy_score(labels_all, preds_all)
    auc = roc_auc_score(labels_all, scores_all)
    adv = acc - 0.5

    # 6) Print & save
    print(f"[{tag}] TF Threshold → Acc: {acc:.3f}, AUC: {auc:.3f}, Adv: {adv:.3f}")
    os.makedirs(RESULT_DIR, exist_ok=True)
    out_path = os.path.join(RESULT_DIR, f"threshold_{tag}.txt")
    with open(out_path, 'w') as f:
        f.write(f"accuracy: {acc:.4f}\n")
        f.write(f"auc: {auc:.4f}\n")
        f.write(f"advantage: {adv:.4f}\n")

if __name__ == "__main__":
    # Iterate over all TFP model files in ../models
    for fname in sorted(os.listdir(MODEL_DIR)):
        if fname.startswith('tfp_eps') and fname.endswith('.h5'):
            tag = fname.replace('.h5', '')
            path = os.path.join(MODEL_DIR, fname)
            threshold_attack(path, tag)

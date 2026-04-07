# src/mia_shadow_tf.py

import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from preprocessing import load_adult_data

MODEL_DIR   = '../models'
RESULT_DIR  = '../results'
NUM_SHADOWS = 5
BATCH_SIZE  = 64
EPOCHS      = 40

def build_model(input_dim):
    return keras.Sequential([
        keras.layers.InputLayer(input_shape=(input_dim,)),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dense(1, activation='sigmoid')
    ])

def train_shadow_models(num_shadows=NUM_SHADOWS):
    X_train, _, y_train, _ = load_adult_data()
    shadow_scores, shadow_labels = [], []

    for i in range(num_shadows):
        rng = np.random.RandomState(42 + i)
        idx = rng.choice(len(X_train), size=len(X_train)//2, replace=False)
        X_sub, y_sub = X_train[idx], y_train[idx]

        # Train a non-private shadow model
        model = build_model(X_train.shape[1])
        model.compile(optimizer='sgd', loss='binary_crossentropy')
        model.fit(X_sub, y_sub,
                  epochs=EPOCHS,
                  batch_size=BATCH_SIZE,
                  verbose=0)

        # Member scores
        mem = model.predict(X_sub, batch_size=BATCH_SIZE).flatten()
        shadow_scores.append(mem)
        shadow_labels.append(np.ones_like(mem))

        # Non-member scores (random half from complement)
        non_idx = np.setdiff1d(np.arange(len(X_train)), idx)
        non_sample = rng.choice(non_idx, size=len(idx), replace=False)
        non = model.predict(X_train[non_sample], batch_size=BATCH_SIZE).flatten()
        shadow_scores.append(non)
        shadow_labels.append(np.zeros_like(non))

    return np.concatenate(shadow_scores), np.concatenate(shadow_labels)

def run_shadow_attack(model_filename):
    # 1) Gather shadow data
    scores, labels = train_shadow_models()
    scores = scores.reshape(-1, 1)

    # 2) Train attacker
    attacker = LogisticRegression(solver='lbfgs', max_iter=1000)
    attacker.fit(scores, labels)

    # 3) Get real model scores
    X_train, X_test, _, _ = load_adult_data()
    X_all = np.concatenate([X_train, X_test])
    true_labels = np.concatenate([np.ones(len(X_train)), np.zeros(len(X_test))])

    # Load target TFP model for inference
    model = tf.keras.models.load_model(
        os.path.join(MODEL_DIR, model_filename),
        compile=False
    )
    real_scores = model.predict(X_all, batch_size=BATCH_SIZE).flatten().reshape(-1,1)

    # 4) Evaluate attacker
    preds = attacker.predict(real_scores)
    acc   = accuracy_score(true_labels, preds)
    auc   = roc_auc_score(true_labels, real_scores)

    print(f"Shadow TF on {model_filename} → Acc: {acc:.3f}, AUC: {auc:.3f}")
    os.makedirs(RESULT_DIR, exist_ok=True)
    tag = model_filename.replace('.h5','')
    with open(f"{RESULT_DIR}/shadow_{tag}.txt",'w') as f:
        f.write(f"accuracy: {acc:.4f}\nauc: {auc:.4f}\n")

if __name__ == "__main__":
    for fname in sorted(os.listdir(MODEL_DIR)):
        if fname.startswith('tfp_eps') and fname.endswith('.h5'):
            run_shadow_attack(fname)

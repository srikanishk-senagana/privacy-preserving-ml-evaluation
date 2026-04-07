# src/train_tfp_dp.py

import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow_privacy.privacy.optimizers.dp_optimizer_keras import DPKerasSGDOptimizer
from preprocessing import load_adult_data
from sklearn.metrics import accuracy_score, f1_score

# --- Hyperparameters ---
BATCH_SIZE = 64
LR         = 0.01
EPOCHS     = 30
DELTA      = 1e-5
SEED       = 42
EPSILONS   = [1, 2, 5, 10]

np.random.seed(SEED)
tf.random.set_seed(SEED)

# Load data
X_train, X_test, y_train, y_test = load_adult_data()

# Prepare output dirs
os.makedirs('../models',  exist_ok=True)
os.makedirs('../results', exist_ok=True)

# Use a loss with no reduction so that we get per-example losses
loss_fn = tf.keras.losses.BinaryCrossentropy(
    from_logits=False,
    reduction=tf.keras.losses.Reduction.NONE
)

for eps in EPSILONS:
    print(f"\n=== TFP: Training with ε = {eps} ===")
    # Approximate noise multiplier
    noise_multiplier = 1.0 / eps
    print(f"noise_multiplier = {noise_multiplier:.4f} (≈ 1/ε)")

    # Build model
    model = keras.Sequential([
        keras.layers.InputLayer(input_shape=(X_train.shape[1],)),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dense(1, activation='sigmoid')
    ])

    # DP optimizer
    dp_optimizer = DPKerasSGDOptimizer(
        l2_norm_clip=1.0,
        noise_multiplier=noise_multiplier,
        num_microbatches=BATCH_SIZE,  # must divide batch size exactly
        learning_rate=LR
    )

    model.compile(
        optimizer=dp_optimizer,
        loss=loss_fn,
        metrics=['accuracy']
    )

    # Train
    model.fit(
        X_train, y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=(X_test, y_test),
        verbose=2
    )

    # Evaluate
    preds = (model.predict(X_test, batch_size=BATCH_SIZE) > 0.5).astype(int).flatten()
    acc   = accuracy_score(y_test, preds)
    f1    = f1_score(y_test, preds)
    print(f"ε={eps} → Accuracy: {acc*100:.2f}%, F1: {f1:.4f}")

    # Save model and metrics
    model.save(f'../models/tfp_eps{eps}.h5')
    with open(f'../results/metrics_tfp_eps{eps}.txt', 'w') as f:
        f.write(f"epsilon: {eps}\naccuracy: {acc:.4f}\nf1: {f1:.4f}\n")

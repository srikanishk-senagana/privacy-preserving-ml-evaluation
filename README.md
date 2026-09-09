# Privacy-Preserving Machine Learning Evaluation

## Overview

This project evaluates the effectiveness of Differential Privacy (DP) in machine learning by analyzing the trade-off between model utility and privacy.

It compares non-private baseline models with differentially private models trained using:

* Opacus (PyTorch)
* TensorFlow Privacy

The system also evaluates privacy risks using Membership Inference Attacks (MIA), providing a practical framework for assessing privacy leakage in ML systems.

---

## Key Features

* End-to-end ML pipeline (data preprocessing → training → evaluation)
* Differential Privacy training with multiple privacy budgets (ε = 1, 2, 5, 10)
* Membership Inference Attacks:
  * Threshold-based attack
  * Shadow model attack
* Comparison across:
  * Baseline (non-private)
  * Opacus DP models
  * TensorFlow Privacy models
* Automated result aggregation and visualization

---

## Tech Stack

* Python
* PyTorch, TensorFlow
* Opacus, TensorFlow Privacy
* Scikit-learn, NumPy, Pandas
* Matplotlib

---

## Project Workflow

1. Data preprocessing (UCI Adult dataset)
2. Train baseline (non-private) model
3. Train DP models with varying privacy budgets (ε)
4. Perform membership inference attacks
5. Aggregate results and generate evaluation plots

---

## Results & Insights

* Increasing privacy (lower ε) reduces membership inference attack success
* Stronger privacy guarantees may impact model performance
* Provides a comparative analysis between Opacus and TensorFlow Privacy
* The experiments show how privacy settings can affect both model utility and privacy-risk evaluation

---

## Limitations

* The evaluation is performed on the UCI Adult dataset and a single MLP architecture.
* Results depend on the selected training configuration and privacy settings.
* The implemented Membership Inference Attacks represent relatively simple attack scenarios.
* The results should be interpreted within the scope of this experiment rather than as a general benchmark of Differential Privacy.
* Privacy and utility results may vary across datasets, model architectures, training configurations, and attack methodologies.

---

## Repository Structure

privacy-preserving-ml-evaluation/
│
├── src/                 # Core pipeline scripts
├── data/                # Dataset (not included or optional)
├── models/              # Saved models (excluded from repo)
├── results/             # Experiment outputs
├── figs/                # Generated plots
├── requirements.txt
└── README.md

---

## How to Run

### 1. Install dependencies

pip install -r requirements.txt

### 2. Train baseline model

python src/train_baseline_opacus.py

### 3. Train DP models

python src/train_opacus_dp.py
python src/train_tfp_dp.py

### 4. Run attacks

python src/mia_threshold.py
python src/mia_threshold_tf.py
python src/mia_shadow.py
python src/mia_shadow_tf.py

### 5. Generate plots

python src/aggregate_and_plot.py

---

## Dataset

* UCI Adult Dataset
  https://archive.ics.uci.edu/ml/datasets/adult

---

## License

This project is licensed under the MIT License.
Please provide appropriate credit if using or referencing this work.

---

## Author

Srikanishk Senagana

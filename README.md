# Privacy-Preserving Machine Learning Evaluation

## Overview

This project explores the relationship between differential privacy, model utility, and membership inference in tabular machine learning.

It compares a non-private baseline with models trained using Opacus (PyTorch) and TensorFlow Privacy on the UCI Adult Income dataset. It also implements threshold-based and shadow-model membership inference attacks to examine whether model outputs can distinguish training examples from held-out examples.

The project is an exploratory implementation of privacy-preserving training and privacy-risk evaluation, rather than a general-purpose privacy benchmark.

## Key Features

- Non-private baseline for utility comparison
- DP-SGD experiments using Opacus and TensorFlow Privacy
- Experiments configured with privacy budgets ε = 1, 2, 5, and 10
- Threshold-score and shadow-model membership inference attacks
- Accuracy, F1, and attack-metric summaries
- Automated result aggregation and visualisation
- Comparison of utility and membership-inference results across configurations

**Privacy-budget note:** The Opacus training script uses `make_private_with_epsilon` with a target ε and δ = 10⁻⁵. The TensorFlow Privacy script sets its noise multiplier to `1 / ε`; the script does not independently calculate or verify the achieved ε. Therefore, the TensorFlow Privacy labels should be understood as experiment settings, not verified privacy guarantees.

## Methodology

### Dataset and preprocessing

The experiments use the UCI Adult Income dataset. The preprocessing pipeline removes rows with missing values, one-hot encodes categorical features, standardises numerical features, and creates a stratified train/test split.

In the current implementation, the encoder and scaler are fitted on the full dataset before the split. This is a limitation of the experimental setup and should be considered when interpreting the results.

### Model training

A multilayer perceptron (MLP) is used as the classifier. The project compares a non-private baseline with DP-SGD training experiments using Opacus and TensorFlow Privacy.

### Membership inference

Two approaches are implemented:

- **Threshold-score attack:** Uses model prediction scores and a fixed threshold to classify examples as members or non-members.
- **Shadow-model attack:** Trains shadow models and a logistic-regression attacker using model confidence scores.

These are exploratory attack implementations. In particular, the threshold attack applies a threshold of 0.5 to the income-prediction score as a membership decision rule. That threshold is not independently justified as an optimal membership threshold, so its accuracy should be interpreted cautiously.

## Results and Insights

### Model utility

The committed `results/utility_summary.csv` reports the following test metrics:

| Library | ε | Accuracy | F1 |
|---|---:|---:|---:|
| Opacus | 1 | 85.66% | 0.6640 |
| Opacus | 2 | 85.83% | 0.6756 |
| Opacus | 5 | 85.78% | 0.6702 |
| Opacus | 10 | 85.86% | 0.6726 |
| TensorFlow Privacy | 1 | 85.24% | 0.6537 |
| TensorFlow Privacy | 2 | 84.92% | 0.6455 |
| TensorFlow Privacy | 5 | 85.14% | 0.6520 |
| TensorFlow Privacy | 10 | 85.08% | 0.6483 |

The reported accuracies remain relatively close across the tested configurations. The results do not show a perfectly monotonic relationship between the configured ε values and utility, and they should not be treated as a controlled ranking of the two libraries.

### Threshold-based membership inference

The committed `results/threshold_summary.csv` reports AUC values of approximately 0.499–0.502, close to chance-level ranking performance.

This indicates that the implemented score-based threshold attack showed little ability to rank members above non-members in these runs. It does **not** establish that the models have no membership-inference risk, nor does it prove that differential privacy caused the observed AUC values.

The reported accuracy-based advantage is negative in these runs because the fixed membership decision rule performs below 50% accuracy. This is another reason to interpret the fixed-threshold accuracy cautiously.

### Shadow-model membership inference

The committed `results/shadow_summary.csv` reports shadow-attack accuracy and AUC. The reported accuracy is around 0.67 for most configurations, with a TensorFlow Privacy result of 0.20 at ε = 2. The AUC values are very close to those in the threshold summary.

These results should be interpreted cautiously. The current attack uses confidence scores alone, and the target evaluation combines members and non-members in the original train/test proportions. Accuracy can therefore be difficult to interpret as a standalone measure of attack effectiveness. The close AUC values across the two summaries also warrant caution when drawing conclusions from this experiment.

### Overall interpretation

The results illustrate how to implement and inspect utility and membership-inference metrics across differentially private training configurations. They do not establish that lower ε consistently reduces membership-inference success, that one library provides stronger privacy than the other, or that the tested models are free from privacy leakage.

## Figures

The `figs/` directory contains plots of:

- Test accuracy by configured privacy budget
- Accuracy drop relative to the baseline
- Threshold-attack AUC and accuracy-based advantage
- Shadow-attack metrics

Interpret plots alongside the CSV summaries and the limitations above. In particular, tightly scaled axes can make small differences appear larger than their absolute magnitude.

## Limitations

- The experiments use one tabular dataset and one MLP architecture.
- Preprocessing is fitted before the train/test split.
- The TensorFlow Privacy script configures the noise multiplier heuristically and does not verify the achieved ε.
- The threshold attack uses a fixed income-prediction threshold as its membership decision rule.
- The shadow attack uses confidence scores alone, and its evaluation accuracy requires careful interpretation.
- The results are configuration-specific and do not establish generalisation across datasets, architectures, random seeds, or attack methods.
- The experiments should not be interpreted as a comprehensive benchmark of differential privacy implementations.

## Tech Stack

- Python
- PyTorch and TensorFlow
- Opacus and TensorFlow Privacy
- scikit-learn, NumPy, and Pandas
- Matplotlib

## Repository Contents

The Python scripts are located in the repository root:

- `preprocessing.py` — dataset loading and preprocessing
- `train_baseline_opacus.py` — non-private baseline
- `train_opacus_dp.py` — Opacus DP-SGD experiments
- `train_tf_dp.py` — TensorFlow Privacy experiments
- `mia_threshold.py` and `mia_threshold_tf.py` — threshold-score attacks
- `mia_shadow.py` and `mia_shadow_tf.py` — shadow-model attacks
- `aggregate_and_plot.py` — result aggregation and plotting
- `results/` — aggregated CSV summaries
- `figs/` — visualisations
- `requirements.txt` — Python dependencies

## Running the Project

Install the listed dependencies from the repository root:

```bash
pip install -r requirements.txt
```

The scripts expect the Adult dataset and output directories at paths relative to their execution location. In particular, `preprocessing.py` expects `../data/adult.data`, and several training and attack scripts use `../models` and `../results`. Set up the expected data and output directories before running the scripts, and check the path constants in each script if your working directory differs.

The current repository layout and path assumptions have not been validated as a clean, end-to-end installation, so the commands should be treated as a starting point rather than a guarantee that the full pipeline runs without adjustment.

## Dataset

UCI Adult Income Dataset: https://archive.ics.uci.edu/ml/datasets/adult

The dataset is not included in this repository.

## License

This project is licensed under the MIT License. Please provide appropriate credit if using or referencing this work.

## Author

Srikanishk Senagana

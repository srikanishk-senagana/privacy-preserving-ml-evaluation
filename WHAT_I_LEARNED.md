# What I Learned

This project gave me practical experience evaluating the relationship between differential privacy, model utility, and membership inference. Beyond implementing the training and attack pipelines, it taught me to be careful about what experimental results do—and do not—establish.

## 1. Predictive utility and privacy are different questions

I trained a non-private baseline and models using Opacus and TensorFlow Privacy, then compared test accuracy and F1 across configurations.

The reported baseline accuracy was 85.92%. The Opacus runs had reported accuracies between 85.66% and 85.86%, while the TensorFlow Privacy runs ranged from 84.92% to 85.24%.

These results show that the tested models retained relatively high predictive accuracy. However, accuracy alone cannot tell us whether a model reveals information about its training data. Utility and privacy need to be evaluated separately.

## 2. A weak attack result does not prove that privacy leakage is absent

My threshold-based membership-inference experiments produced AUC values close to 0.5, which corresponds to chance-level ranking performance.

This means the particular score-based attack I implemented showed little ability to distinguish members from non-members in these runs. It does not prove that the models have no membership-inference risk. The attack uses income-prediction scores as a membership signal, and those scores may not be sufficiently informative about training membership.

This taught me to distinguish between an attack failing to detect a risk and evidence that the risk does not exist.

## 3. Metrics and evaluation design affect interpretation

The project includes threshold-based and shadow-model membership-inference attacks. Their reported metrics need to be interpreted in the context of how each attack is constructed and evaluated.

For example, the threshold attack applies a fixed threshold of 0.5 to the classifier's prediction score to decide membership. That threshold is not independently established as an appropriate membership decision rule. Accuracy can also be misleading when the member and non-member classes are not balanced.

I learned that metrics cannot be interpreted in isolation. The attack strategy, class proportions, decision rule, and evaluation protocol all affect the conclusions we can draw.

## 4. Privacy guarantees must be verified, not inferred from configuration labels

The Opacus training script uses `make_private_with_epsilon` with a target privacy budget. The TensorFlow Privacy script configures its noise multiplier as `1 / ε`, but does not independently verify the achieved privacy budget.

This distinction matters because a configured noise multiplier is not, by itself, evidence that a run satisfies a particular ε guarantee. A stronger comparison would require appropriate privacy accounting and clearly documented assumptions.

The project taught me that using a privacy library is only one part of implementing differential privacy correctly. Understanding and verifying the guarantee is equally important.

## 5. The entire pipeline matters

The current preprocessing code fits the encoder and scaler before the train/test split. This is a limitation because information from the held-out data influences preprocessing.

I also learned that a comparison between libraries is only meaningful when the experimental setup, metrics, and privacy accounting are sufficiently well defined and comparable. These issues do not erase the implementation work, but they limit the strength of the conclusions.

## 6. The broader lesson: evaluate the evaluation

The main lesson I take from this project is that an evaluation result is only as informative as the method and assumptions behind it.

A model can retain high accuracy without that establishing its privacy properties. An attack can perform near chance without proving that leakage is absent. Similarly, a privacy library's configuration should not be treated as a verified guarantee without appropriate accounting.

This connects to my broader interest in AI safety and evaluation. For any evaluation, I want to ask:

- What exactly does the test measure?
- What assumptions does it make?
- Which failure modes might it miss?
- How far can the results be generalised?
- What additional evidence would justify a stronger conclusion?

I see this project as an exploratory implementation and evaluation study, not a comprehensive privacy benchmark. Its value for me lies both in the technical work and in learning to interpret results critically, state limitations clearly, and identify what a more rigorous next experiment would need to establish.

## Limitations and possible next steps

If I continue this work, I would prioritise:

- Fitting preprocessing only on the training split and applying the fitted transformations to held-out data.
- Verifying the achieved privacy budget for each training run using the appropriate accounting method.
- Evaluating loss-based membership-inference attacks alongside the current score-based baseline.
- Using a clearly defined and comparable evaluation protocol for attack metrics.
- Repeating experiments across multiple random seeds and, eventually, additional datasets or model architectures.

These are directions for future work, not claims that the current experiments have already completed them.

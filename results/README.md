# Research Results

This directory stores outputs generated during the ECG research experiments.

such as LOSO oversampling completed.
Total predictions: 100693
Evaluation time: 2332.91 seconds

Balanced Accuracy: 0.4412
Macro F1: 0.4043

Classification Report:
              precision    recall  f1-score   support

           N       0.95      0.95      0.95     90102
           S       0.00      0.00      0.00      2781
           V       0.56      0.81      0.67      7008
           F       0.00      0.00      0.00       802

    accuracy                           0.91    100693
   macro avg       0.38      0.44      0.40    100693
weighted avg       0.89      0.91      0.90    100693


Confusion Matrix:
[[85454   368  4095   185]
 [ 2649     1   131     0]
 [ 1265     2  5711    30]
 [  580     0   221     1]]


## Organization

- `statistical_analysis/`: Statistical tests, diagnostic summaries, and exploratory analysis.
- `model_comparison/`: Classification metrics and comparisons between modelling approaches.
- `diagnostic_selection/`: Results from diagnostic-driven model selection experiments.
##XGBoost Result

## Reproducibility

Results will be generated from experimental notebooks and reusable Python scripts.

Tables and figures will be based on recorded experimental outputs.
Original results will be preserved to support reproducibility and manuscript preparation.

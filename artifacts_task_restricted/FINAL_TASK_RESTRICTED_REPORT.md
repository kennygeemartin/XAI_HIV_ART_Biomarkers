# Final task-restricted experiment

## Feature policy

The HIV-screening forward pass was restricted to age, sex, height, weight, BMI, fasting status, lipid biomarkers and ratios, albumin, total protein, globulin, albumin/globulin ratio, HbA1c, glucose, diabetes, creatinine, AST, ALT, and lipid-lowering medication indicators. It did not receive ONART, ARTV, HAARTV2, adherence, ART duration, CD4, CD8, or CD4/CD8 ratio tokens.

The ART-monitoring forward pass received the same common features plus ART duration, CD4 count and percentage, CD8 count and percentage, CD4/CD8 ratio, ONART, ART regimen, HAART regimen, and adherence. The feature tokenizer and Transformer encoder parameters remained shared, but separate forward passes constructed task-permitted token sequences before the corresponding output head.

## FT-Transformer results

| Task | n | Accuracy | Sensitivity | Specificity | Precision | F1 | AUROC | AUPRC |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Non-invasive HIV screening | 877 | 0.685 | 0.893 | 0.312 | 0.700 | 0.785 | 0.702 | 0.812 |
| ART monitoring, eligible only | 500 | 0.914 | 0.625 | 0.945 | 0.545 | 0.583 | 0.886 | 0.573 |

The HIV confusion matrix was `[[98, 216], [60, 503]]`; the ART confusion matrix was `[[427, 25], [18, 30]]`. The validation checkpoint was selected at epoch 6, and training stopped at epoch 16.

## Leakage-safe benchmark comparison

### HIV screening

| Model | Accuracy | Sensitivity | Specificity | Precision | F1 | AUROC | AUPRC |
|---|---:|---:|---:|---:|---:|---:|---:|
| FT-Transformer | 0.685 | **0.893** | 0.312 | 0.700 | 0.785 | 0.702 | 0.812 |
| Logistic regression | 0.674 | 0.671 | **0.678** | **0.789** | 0.726 | **0.725** | **0.821** |
| Random forest | 0.691 | 0.762 | 0.564 | 0.758 | 0.760 | 0.719 | 0.811 |
| XGBoost | 0.673 | 0.691 | 0.640 | 0.775 | 0.731 | 0.723 | 0.813 |
| MLP | **0.714** | 0.861 | 0.449 | 0.737 | **0.794** | 0.723 | 0.817 |

At the fixed 0.50 threshold, the FT-Transformer prioritized sensitivity but had low specificity. Logistic regression had the highest AUROC and AUPRC, while the MLP had the highest accuracy and F1. Threshold tuning was not performed on test data.

### ART monitoring

| Model | Accuracy | Sensitivity | Specificity | Precision | F1 | AUROC | AUPRC |
|---|---:|---:|---:|---:|---:|---:|---:|
| FT-Transformer | **0.914** | 0.625 | **0.945** | **0.545** | **0.583** | 0.886 | 0.573 |
| Logistic regression | 0.786 | **0.833** | 0.781 | 0.288 | 0.428 | 0.888 | 0.444 |
| Random forest | 0.846 | **0.833** | 0.847 | 0.367 | 0.510 | 0.901 | **0.597** |
| XGBoost | 0.858 | 0.750 | 0.869 | 0.379 | 0.503 | **0.901** | 0.589 |
| MLP | 0.868 | 0.583 | 0.898 | 0.378 | 0.459 | 0.843 | 0.468 |

The FT-Transformer had the strongest threshold-dependent accuracy, specificity, precision, and F1. Random forest and XGBoost had higher ranking discrimination. All results are from synthetic proxy data and require external validation before clinical interpretation.

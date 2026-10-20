# FT-Transformer model-development results

## Data configuration and composition

| Partition | Participants | Dataset share | HIV-negative | HIV-positive | ART eligible | ART suppressed | ART unsuppressed |
|---|---:|---:|---:|---:|---:|---:|---:|
| Overall | 5,844 | 100.0% | 2,092 | 3,752 | 3,294 | 2,979 | 315 |
| Training | 4,090 | 70.0% | 1,464 | 2,626 | 2,296 | 2,083 | 213 |
| Validation | 877 | 15.0% | 314 | 563 | 498 | 444 | 54 |
| Test | 877 | 15.0% | 314 | 563 | 500 | 452 | 48 |

Each participant occurs once, so the fixed participant-level split contains no cross-partition participant overlap. Overall HIV-positive prevalence was 64.2%. ART unsuppression prevalence was 9.3% in eligible training participants, 10.8% in validation, and 9.6% in test.

| Processing stage | Rows | Predictors | Numerical tokens | Categorical tokens | Missing selected cells | Added synthetic rows |
|---|---:|---:|---:|---:|---:|---:|
| Raw dataset | 5,844 | 54 candidate/non-outcome columns | — | — | 42,195 across the file | 0 |
| After cleaning and selection | 5,844 | 40 | 27 | 13 | 21,993 | 0 |
| After imputation and transformation | 5,844 | 40 | 27 | 13 | 0 | 0 |
| FT training matrix after augmentation | 4,918 | 40 | 27 | 13 | 0 | 828 |

The 40 retained predictors comprised 27 continuous variables and 13 categorical variables. No selected variable exceeded the 50% training-missingness threshold. Multivariate iterative imputation removed numerical missingness; mode imputation removed categorical missingness. Seven positively skewed features received `log1p` transformation, after which every continuous feature was standardized using training-only parameters. No value was simultaneously outside its three-IQR fence and the broad clinical plausibility limits, so no record was deleted as an erroneous outlier. Viral-load variables, direct labels, eligibility variables, `HIVPOS`, identifiers, dates, and split metadata were excluded from the model matrix.

Training-only TabDDPM-style augmentation added 828 unsuppressed ART records, changing the eligible training balance from 213:2,083 to 1,041:2,083. Validation and test composition remained unchanged.

## Evaluation design

The multitask FT-Transformer used 64-dimensional feature tokens, three Transformer blocks, eight attention heads, a 128-unit feed-forward sublayer, dropout of 0.20, Adam optimization, and masked binary cross-entropy for ART monitoring. Early stopping selected epoch 5. Logistic regression, random forest, XGBoost, and a conventional MLP were evaluated as benchmarks. Metrics use the fixed 0.50 decision threshold and the untouched participant-level test partition. ART metrics include only the 500 eligible test participants.

## HIV screening

| Model | Accuracy | Sensitivity | Specificity | Precision | F1 | AUROC | AUPRC |
|---|---:|---:|---:|---:|---:|---:|---:|
| FT-Transformer | 0.989 | 0.984 | 0.997 | 0.998 | 0.991 | 0.999 | 1.000 |
| Logistic regression | 0.977 | 0.970 | 0.990 | 0.995 | 0.982 | 0.992 | 0.997 |
| Random forest | 0.993 | 0.993 | 0.994 | 0.996 | 0.995 | 0.999 | 1.000 |
| XGBoost | **0.994** | **0.995** | 0.994 | 0.996 | **0.996** | **1.000** | **1.000** |
| MLP | 0.982 | 0.984 | 0.978 | 0.988 | 0.986 | 0.995 | 0.998 |

The FT-Transformer correctly classified 867 of 877 participants (`TN=313`, `FP=1`, `FN=9`, `TP=554`). Its discrimination was near perfect, but XGBoost achieved the strongest point estimates overall. Differences at this performance level are very small and were not tested for statistical significance.

## ART-response monitoring

| Model | Accuracy | Sensitivity | Specificity | Precision | F1 | AUROC | AUPRC |
|---|---:|---:|---:|---:|---:|---:|---:|
| FT-Transformer | **0.920** | 0.542 | **0.960** | **0.591** | **0.565** | **0.904** | **0.572** |
| Logistic regression | 0.834 | **0.688** | 0.850 | 0.327 | 0.443 | 0.859 | 0.424 |
| Random forest | 0.852 | 0.625 | 0.876 | 0.349 | 0.448 | 0.880 | 0.530 |
| XGBoost | 0.888 | 0.667 | 0.912 | 0.444 | 0.533 | 0.895 | 0.566 |
| MLP | 0.864 | 0.646 | 0.887 | 0.378 | 0.477 | 0.857 | 0.521 |

The FT-Transformer produced `TN=434`, `FP=18`, `FN=22`, and `TP=26`. It provided the best accuracy, specificity, precision, F1, AUROC, and AUPRC. Logistic regression produced the highest sensitivity, but at the cost of 68 false positives and substantially lower precision. Threshold selection should therefore be based on the intended clinical costs: a lower FT-Transformer threshold would be expected to improve sensitivity while reducing specificity.

## SHAP interpretation

Permutation SHAP was calculated for 60 deterministic held-out participants per task using 20 training-background records. The analysis was restricted to lipid and protein biomarkers. Mean absolute SHAP values quantify global contribution magnitude and do not establish causality.

For HIV prediction, the five leading biomarkers were triglycerides (0.01394), albumin/globulin ratio (0.01354), triglyceride/HDL ratio (0.01127), total protein (0.00995), and albumin (0.00980).

For ART unsuppression prediction, the leading biomarkers were albumin (0.03248), triglyceride/HDL ratio (0.02107), globulin (0.01836), total cholesterol (0.01705), and total-cholesterol/HDL ratio (0.01620). The SHAP distribution suggests that lower albumin generally increased the predicted probability of unsuppression, while higher triglyceride/HDL and total-cholesterol/HDL ratios generally pushed predictions upward. These are model associations in synthetic data, not clinical effects.

## Figures

- Dataset and class composition: `data_configuration/dataset_and_class_composition.png`
- Missingness before and after preprocessing: `data_configuration/missingness_before_after.png`
- Selected raw and processed distributions: `data_configuration/selected_distributions_before_after.png`
- ART class composition before and after augmentation: `data_configuration/art_augmentation_before_after.png`
- FT-Transformer learning curve: `ft_training_curves.png`
- HIV model comparison: `hiv_metric_comparison.png`
- HIV ROC and precision-recall curves: `hiv_roc_pr_curves.png`
- HIV confusion matrix: `hiv_ft_confusion_matrix.png`
- HIV global and patient-level SHAP: `hiv_shap_global_bar.png`, `hiv_shap_beeswarm.png`, `hiv_shap_patient_waterfall.png`
- ART model comparison: `art_metric_comparison.png`
- ART ROC and precision-recall curves: `art_roc_pr_curves.png`
- ART confusion matrix: `art_ft_confusion_matrix.png`
- ART global and patient-level SHAP: `art_shap_global_bar.png`, `art_shap_beeswarm.png`, `art_shap_patient_waterfall.png`

## Interpretation constraints

The supplied dataset is explicitly synthetic, so these findings validate the modeling workflow and cannot be presented as empirical MWCCS clinical performance. Treatment and immune-status inputs are downstream of HIV diagnosis and likely explain the near-perfect HIV-screening results. Prospective screening claims require an index-time feature policy, removal of all post-diagnosis predictors, external validation, confidence intervals, calibration assessment, and clinically justified threshold selection.

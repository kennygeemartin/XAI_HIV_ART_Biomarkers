# Preprocessed and augmented multitask FT-Transformer

This model was trained on the supplied synthetic MWCCS-style proxy dataset. It is for pipeline development and simulation, not empirical MWCCS clinical claims.

## Preprocessing

- Removed exact duplicate rows (none found); participants were already unique.
- Coerced selected continuous fields to numeric format.
- Fit all preprocessing on the 4,090-row training split only and applied it unchanged to validation and test.
- Used a 50% training-missingness exclusion threshold; no selected feature exceeded it.
- Combined 3-IQR fences with broad clinical bounds. No values violated both criteria, so no rows or values were removed as clearly erroneous.
- Applied `log1p` based on training skew greater than 1 to triglycerides, TC/HDL, LDL/HDL, TG/HDL, AST, ALT, and HSRAT.
- Applied multivariate iterative imputation to continuous variables and training-mode imputation to categorical variables.
- Z-standardized every continuous input after transformation and imputation.
- Retained categorical variables as FT-Transformer embedding tokens.
- Excluded direct outcomes, eligibility fields, IDs, dates, split metadata, `HIVPOS`, and viral-load fields. Viral load was not log-transformed because it defines the synthetic ART outcome and would leak the target.

The iterative imputer reached its configured 15 iterations but emitted a convergence warning. Its fitted state was still used consistently for all splits and was saved with the model artifacts.

## Training-only augmentation

The eligible ART training subset contained 2,083 suppressed and 213 unsuppressed records. Because the majority/minority ratio exceeded 4:1, a local TabDDPM-style Gaussian diffusion model was fitted only to the 213 minority training records. It generated 828 synthetic minority records, producing a minority-to-majority ratio of approximately 1:2. Validation and test data were neither augmented nor used to fit the generator.

## Model

The model uses 64-dimensional feature tokens, three Transformer blocks, eight attention heads, FFN dimension 128, dropout 0.20, Adam learning rate 0.001, weight decay 0.00001, and masked ART binary cross-entropy. Early stopping selected epoch 5 and ended training after epoch 15.

## Held-out synthetic test results

| Task | n | ROC-AUC | Average precision | Balanced accuracy | F1 | Brier |
|---|---:|---:|---:|---:|---:|---:|
| HIV screening | 877 | 0.9994 | 0.9997 | 0.9904 | 0.9911 | 0.00858 |
| ART monitoring, eligible only | 500 | 0.9035 | 0.5720 | 0.7509 | 0.5652 | 0.06001 |

At a 0.5 threshold, the ART confusion matrix (`[[TN, FP], [FN, TP]]`) was `[[434, 18], [22, 26]]`.

Treatment and immune-status predictors remain downstream of HIV diagnosis, as required by the proposed shared architecture. Consequently, near-perfect HIV screening performance is still likely optimistic and should not be interpreted as prospective screening validity. A real screening model needs an index-time feature policy that excludes post-diagnosis information.

# Synthetic MWCCS multitask FT-Transformer

## Intended use

Pipeline development and simulation only. `MWCCS_data_card.csv` states that every row is synthetic and that this dataset is not actual MWCCS participant data. Results must not be presented as empirical clinical performance.

## Model and training

- 27 numerical and 13 categorical feature tokens, plus `[CLS]`
- Token dimension 64; 3 Transformer blocks; 8 attention heads; FFN dimension 128
- Dropout 0.20; Adam learning rate 0.001; weight decay 0.00001
- Batch size 64; maximum 100 epochs; early-stopping patience 10
- Shared encoder with HIV-screening and ART-monitoring binary-logit heads
- ART BCE loss restricted to `ART_MONITORING_ELIGIBLE == 1`
- Fixed participant-level train/validation/test splits; preprocessing fit on train only
- Best validation loss 0.211830 at epoch 5; training stopped after epoch 15

## Held-out synthetic test results

| Task | n | Prevalence | ROC-AUC | Average precision | Balanced accuracy | F1 | Brier |
|---|---:|---:|---:|---:|---:|---:|---:|
| HIV screening | 877 | 0.642 | 1.000 | 1.000 | 1.000 | 1.000 | 0.00000009 |
| ART monitoring (eligible only) | 500 | 0.096 | 0.896 | 0.591 | 0.704 | 0.512 | 0.0592 |

At threshold 0.5, the ART confusion matrix (`[[TN, FP], [FN, TP]]`) is `[[439, 13], [27, 21]]`.

## Leakage controls and limitations

`HIVPOS`, viral-load values and flags, outcome labels, eligibility fields, identifiers, dates, and split metadata were excluded. Viral load is excluded because the synthetic ART label is defined from `VLOAD >= 1000`.

Treatment predictors were retained because they are explicitly part of the proposed shared architecture. They are downstream of HIV diagnosis, however, and structural missingness in those variables strongly identifies HIV status. The perfect synthetic HIV result is therefore not evidence of a deployable screening model. Before real screening evaluation, define an index time and restrict screening inputs to variables available before HIV testing (normally excluding ART exposure, ART regimen, ART adherence, ART duration, CD4/CD8, and other post-diagnosis measurements).

## Interpretability

SHAP was not run because the `shap` dependency is not installed in the current environment. The saved checkpoint and preprocessing object are ready for a separately validated SHAP workflow. Interpretation should be conducted only after the leakage-safe, task-specific feature set is finalized.

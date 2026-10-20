# Multitask FT-Transformer training

This project trains a 64-dimensional, 3-block, 8-head FT-Transformer with separate HIV-screening and ART-monitoring heads. ART loss is masked to `ART_MONITORING_ELIGIBLE == 1`.

Run:

```powershell
python train_ft_transformer.py
```

The fixed participant-level `DATA_SPLIT` is respected. Preprocessing is fit on training data only. Direct target proxies (`HIVPOS`, viral-load fields), labels, eligibility fields, identifiers, raw dates, and split metadata are excluded from inputs. Outputs are written to `artifacts/`.

Important: the supplied data is synthetic. Metrics validate the pipeline and must not be reported as empirical MWCCS clinical performance. Treatment predictors are retained to match the proposed architecture, but are downstream of HIV diagnosis and can inflate screening performance; a screening-only deployment model should omit them.

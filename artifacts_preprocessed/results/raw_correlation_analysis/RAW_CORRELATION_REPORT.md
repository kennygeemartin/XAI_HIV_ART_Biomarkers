# Correlation analysis before preprocessing

## Analysis configuration

The original 5,844-row CSV was analyzed without duplicate removal, imputation, outlier treatment, logarithmic transformation, standardization, feature filtering, or augmentation. Pearson and Spearman correlations used pairwise-complete raw observations. Therefore, the sample size differs by variable pair and is recorded in `raw_numeric_pairwise_n.csv`. Binary-outcome associations are point-biserial correlations, equivalent to Pearson correlation with a 0/1 outcome. ART correlations include only the 3,294 eligible records with observed ART outcomes. Bias-corrected Cramér's V was used for categorical pairs.

Participant identifiers, dates, the constant synthetic flag, explicit task masks, and duplicate outcome definitions were excluded from ranked correlation analysis because their numeric codes do not have a meaningful continuous interpretation. `HIVPOS` was excluded because it directly duplicates the HIV-screening label. Viral load is shown in the raw ART analysis for descriptive completeness but is a direct source of target leakage because the synthetic ART outcome was defined from viral load.

## Strongest raw numerical relationships

| Rank | Variable 1 | Variable 2 | Pearson r | Pairwise n | Interpretation |
|---:|---|---|---:|---:|---|
| 1 | TC/HDL ratio | LDL/HDL ratio | 0.961 | 5,090 | Shared HDL denominator and related lipid components |
| 2 | Total cholesterol | Non-HDL cholesterol | 0.935 | 5,403 | Non-HDL is mathematically derived from the lipid panel |
| 3 | CD4 count | CD4 percentage | 0.924 | 3,752 | Closely related immune-status measurements |
| 4 | Non-HDL cholesterol | Measured LDL-C | 0.885 | 5,289 | Overlapping atherogenic cholesterol content |
| 5 | Total cholesterol | Measured LDL-C | 0.874 | 5,294 | LDL-C contributes substantially to total cholesterol |
| 6 | Globulin | Albumin/globulin ratio | -0.871 | 5,348 | Globulin is the denominator of the ratio |
| 7 | Weight | BMI | 0.867 | 5,739 | BMI is calculated from weight and height |
| 8 | HDL | TC/HDL ratio | -0.805 | 5,405 | HDL is the denominator of the ratio |
| 9 | Triglycerides | TG/HDL ratio | 0.792 | 5,411 | Triglycerides are the numerator of the ratio |
| 10 | Total protein | Globulin | 0.776 | 5,348 | Globulin is derived from total protein and albumin |

Spearman analysis produced almost the same ordering. This indicates that the dominant relationships were robust monotonic associations and not solely driven by skewed extremes.

## Raw correlations with HIV-screening outcome

| Rank | Raw variable | Correlation | Available n |
|---:|---|---:|---:|
| 1 | Albumin/globulin ratio | -0.297 | 5,348 |
| 2 | Globulin | 0.270 | 5,348 |
| 3 | Triglycerides | 0.186 | 5,625 |
| 4 | TG/HDL ratio | 0.171 | 5,411 |
| 5 | Albumin | -0.166 | 5,584 |
| 6 | Non-HDL cholesterol | 0.133 | 5,614 |
| 7 | Total cholesterol | 0.124 | 5,619 |
| 8 | Total protein | 0.116 | 5,600 |
| 9 | TC/HDL ratio | 0.091 | 5,405 |
| 10 | Measured LDL-C | 0.078 | 5,506 |

The strongest retained raw association was still modest (`|r| < 0.30`). HIV-positive status was associated with a lower albumin/globulin ratio and albumin, and with higher globulin, triglycerides, and TG/HDL ratio in this synthetic dataset. Correlation alone does not adjust for treatment, demographic variables, missingness, or dependencies between biomarkers.

## Raw correlations with ART unsuppression

| Rank | Raw variable | Correlation | Eligible n | Interpretation |
|---:|---|---:|---:|---|
| 1 | Viral load | 0.616 | 3,294 | Direct target leakage; outcome defined from viral load |
| 2 | CD4 count | -0.414 | 3,294 | Moderate inverse association |
| 3 | CD4 percentage | -0.374 | 3,294 | Moderate inverse association |
| 4 | CD4/CD8 ratio | -0.279 | 3,294 | Weak-to-moderate inverse association |
| 5 | Albumin/globulin ratio | -0.236 | 3,025 | Weak inverse association |
| 6 | Globulin | 0.194 | 3,025 | Weak positive association |
| 7 | Albumin | -0.177 | 3,155 | Weak inverse association |
| 8 | CD8 count | 0.138 | 3,294 | Weak positive association |
| 9 | TG/HDL ratio | 0.097 | 3,061 | Very weak positive association |
| 10 | CD8 percentage | 0.084 | 3,294 | Very weak positive association |

Viral load must not be used as an ART predictor in this dataset because the label is explicitly `VLOAD >= 1000`. Among non-leaking measurements, lower CD4 count and percentage had the largest raw associations with unsuppression.

## Raw categorical associations

| Rank | Variable pair | Cramér's V |
|---:|---|---:|
| 1 | ONART – adherence category | 1.000 |
| 2 | ONART – HAART regimen | 1.000 |
| 3 | ONART – ART regimen | 0.999 |
| 4 | ART regimen – adherence category | 0.500 |
| 5 | HAART regimen – adherence category | 0.500 |
| 6 | ART regimen – HAART regimen | 0.409 |

These near-perfect treatment-variable associations reflect common treatment eligibility and structural coding rather than independent measurements. All remaining categorical-pair associations were small; the largest was site with diabetes (`V=0.047`).

## Modeling implications

The raw matrix contains substantial multicollinearity among calculated lipid ratios, their component biomarkers, protein-derived ratios, BMI and weight, and immune-status measures. FT-Transformer attention can accommodate correlated tokens, but correlation does not make redundant derived variables independent sources of information. Conventional regression should use regularization or a prespecified reduced feature set. For prospective HIV screening, treatment and immune-status variables require an index-time review because they can reveal previous diagnosis. Viral load must remain excluded from ART prediction.

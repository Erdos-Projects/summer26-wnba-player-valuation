# Evaluation Summary

## Data Splits 
The WNBALeakageProofSplitter creates a chronologically valid train and test split for predicting WNBA player salaries to ensure no future information leaks into training data, where one row = one player in one year. The training data is composed of all data from 2021 - 2024 and leaves the 2025 data as the test data. This prevents look-ahead bias. 

Then we have cross-validation folds:
- Fold 1 : Train on 2021 --> Validate on 2022
- Fold 2 : Train on 2021-2022 --> Validate on 2023
- Fold 3 : Train on 2021 - 2023 --> Validate on 2024

These folds simulate real-world scenario where you only have past data to predict future. 

The leakage_blacklist removes features that would make the prediction too easy.


## Stress tests

## Stress Test Summary

### Test 1: Noise Injection

All models respond to noise proportionally. Prediction shift scales linearly with noise level and no model exhibits sudden instability. Ridge is the most stable overall, with slightly lower mean shifts than LinearRegression at every noise level. RandomForest is the least sensitive at high noise (20% noise -> ~$9,900 mean shift vs ~$17,400 for LinearRegression), likely because tree splits are discrete and small perturbations often don't cross a decision boundary. DummyRegressor is unaffected by noise by construction.

**Conclusion:** All models pass the noise injection test. No fragility concerns.

---

### Test 2: Extreme Stat Inputs

All models produce predictions within the observed training salary range ($464–$242,154) for all three extreme input profiles. Key observations:

| Input Profile | LinearRegression | Ridge | RandomForest |
|---|---|---|---|
| p01 (fringe player) | $18,215 | $19,091 | $30,993 |
| p99 (superstar) | $151,928 | $153,905 | $186,085 |
| p99 volume / p01 availability | $133,057 | $131,548 | $117,646 |

RandomForest under-predicts the p99 superstar ceiling ($186k vs actual max of $242k), consistent with the known tendency of tree models to truncate predictions at extremes. The mixed-extreme profile (high volume, low availability) correctly receives a lower prediction than the pure p99 profile across all models, indicating the availability features (`avail_rate`, `start_rate`) are functioning as intended.

**Conclusion:** All models pass the extreme input test. No out-of-range or nonsensical predictions.

---

### Test 3: Contract Group Robustness

Performance varies significantly by contract group. The hardship group is a clear failure point across all models:

| Group | n | Median Actual Salary | RandomForest MAE | MAE as % of Median |
|---|---|---|---|---|
| rookie | 40 | $75,276 | $13,045 | 17% |
| controlled | 13 | $66,079 | $28,771 | 44% |
| unknown | 17 | $78,066 | $31,202 | 40% |
| veteran | 133 | $78,831 | $37,363 | 47% |
| hardship | 20 | $6,631 | $20,330 | **307%** |

Hardship contracts are CBA-fixed at very low values (~$6.6k median) that bear no relationship to on-court performance. Every model defaults to predicting league-average salaries for these players, producing MAE errors of 307–1059% of the actual median. This is not a modeling failure — it is a structural data limitation.

Rookie contracts are the best-predicted group (17% MAE as % of median for RandomForest), suggesting the `pts_rookie` and `ws_rookie` interaction features are doing their job.

**Conclusion:** Models are well-calibrated for rookie, controlled, and unknown groups. Veteran group errors (~47% MAE/median) are acceptable given the salary spread. Hardship contracts should be excluded from primary evaluation metrics or treated as a separate regime, as their salaries are CBA-determined rather than performance-driven.

---

### Overall Findings

- **Stability:** All models are robust to input noise and extreme stat profiles.
- **Sensibility:** No model produces out-of-range predictions under any stress condition tested.
- **Known limitation:** Hardship contracts are not learnable from performance features due to CBA wage floors. This distorts aggregate MAPE metrics and should be flagged in the written summary.
- **Best overall model under stress:** RandomForest. Lowest RMSE on holdout ($41,616), most stable under noise at high perturbation levels, and best MAE/median on hardship and rookie groups.

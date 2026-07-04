# Final Executive Summary

## 1. Problem Definition

The WNBA (Women's National Basketball Association) is one of the fastest-growing sports markets, with sponsorship deals surging over 40%, in-arena attendance at a 22-year high, and a 130% increase in young viewers all in the last year. As sponsorship and attendance grow, teams need better ways to evaluate whether player salaries align with on-court production and whether rosters are being built efficiently.

The project is aimed to answer the following questions:

- Which playeres appear underpaid or overpaid?
- Are teams spending wisely?

To answer this, we build a salary valuation model that using data from the 2021-2024 seasons to train our model, with the 2025 season as holdout dataset to estimate each player's predicted(FMV) salary. In this project, FMV refers to the model-predicted salary.

The analysis has two levels:

- Player-level valuation: compare actual salary with predicted(FMV) salary to identify players who appear underpaid or overpaid relative to the model.

- Team-level ROI valuation: compare actual Cost Per Win (CPW) with predicted FMV CPW. Actual CPW is calculated using actual team payroll for each teams dividede by team wins, while predicted(FMV) CPW is calculated using the sum of model-predicted player salaries for each team divided by team wins. This gives a team-level measure of whether a roster is spending more or less per win than its model-implied fair market value.

The goal of this project is to provide a quantitative framework(model) for evaluating player valuation and team spending efficiency, instead of making definitive claims about player quality, contract fairness, or front-office decision-making. 

---

## 2. Data Sources

This project uses public WNBA salary, player performance, team context, and team standings data. The modeling data covers the 2021–2025 seasons, with 2021–2024 used for model training and 2025 used as the final holdout evaluation season.

The salary data comes from Her Hoop Stats, while the player statistics, team statistics, and standings data come from Basketball Reference. These raw annual files are combined, cleaned, and transformed through the project’s preprocessing, feature engineering, and final modeling notebooks before training and evaluating the final model.

Key data sources include:

| Dataset | Source | Raw Files | Unit of Analysis | Role in Project |
|---|---|---|---|---|
| WNBA salary data | Her Hoop Stats salary | `data/raw/salary_[year].csv` | Player-season | Salary target and salary reference |
| WNBA player advanced stats | Basketball Reference | `data/raw/[year]_advanced.csv` | Player-season | Player efficiency and advanced performance features |
| WNBA player per-game stats | Basketball Reference | `data/raw/[year]_per_game.csv` | Player-season | Per-game production features |
| WNBA player total stats | Basketball Reference | `data/raw/[year]_totals.csv` | Player-season | Volume and counting-stat features |
| WNBA team advanced stats | Basketball Reference | `data/raw/[year]_advanced-team.csv` | Team-season | Team-level context features |
| WNBA standings | Basketball Reference | `data/raw/[year]_wnba_standings.csv` | Team-season | Team wins used for Cost Per Win analysis |

`[year]` refers to each season from 2021 through 2025.

The final 2025 evaluation dataset contains 223 player rows. Salary data is our prediction target, while the Basketball Reference datasets provide the player-level and team-level features used to estimate FMV. Team standings are used separately in the team-level ROI analysis to calculate Cost Per Win.

---

## 3. KPI Reporting

### 3.1 Model Prediction KPIs

The final model is evaluated on the 2025 holdout set by comparing actual salary with model-predicted salary.

The primary KPI is RMSE, which reports the typical size of the model’s salary prediction error in dollars. A lower RMSE means the model’s predicted salaries are closer to actual salaries.

The secondary KPI is MAPE, which reports prediction error as a percentage of actual salary. This helps compare errors across players with different salary levels, but it can become very large for low-salary contracts such as hardship contracts.

Final model KPI results on the 2025 holdout set:

| Metric | Value |
|---|---:|
| RMSE | $41,368.60 |
| MAE | $30,376.25 |
| MAPE | 128.70% |
| R² | 0.6342 |

The model was trained on 742 rows from 2021–2024 and evaluated on 223 rows from 2025. The final model uses 25 features.

Because salary rules vary across contract groups, we also report MAPE by contract group as a secondary fairness check.

| Contract Group | n | Median Actual Salary | MAPE |
|---|---:|---:|---:|
| hardship | 20 | $6,630.50 | 289.7% |
| veteran | 133 | $78,831.00 | 156.0% |
| controlled | 13 | $66,079.00 | 64.1% |
| unknown | 17 | $78,066.00 | 37.3% |
| rookie | 40 | $75,276.00 | 17.1% |

The high MAPE for hardship contracts suggests that these contracts are structurally difficult to predict from performance features alone. 

### 3.2 Business KPI: Team-Level Cost Per Win

The project also reports a team-level ROI valuation via Cost Per Win (CPW). Cost Per Win measures how much salary spending is associated with each team win.

`Actual CPW = actual total team payroll / team wins`

`Predicted(FMV) CPW = predicted total team payroll / team wins`

The CPW gap is calculated as:

`CPW gap = actual CPW - predicted FMV CPW`

A negative CPW gap means a team’s actual cost per win is below the model-implied fair market cost per win, suggesting more efficient spending. A positive CPW gap means a team’s actual cost per win is above the model-implied fair market cost per win, suggesting less efficient spending.

Most efficient spenders:

| Team | Wins | Actual CPW | Predicted FMV CPW | CPW Gap |
|---|---:|---:|---:|---:|
| CON | 11 | $101,140.91 | $125,357.31 | -$24,216.40 |
| GSV | 23 | $47,492.65 | $54,309.99 | -$6,817.34 |
| PHO | 27 | $52,823.70 | $57,859.08 | -$5,035.38 |
| WAS | 16 | $46,664.56 | $51,447.96 | -$4,783.40 |
| IND | 24 | $51,681.42 | $54,868.73 | -$3,187.32 |
| SEA | 23 | $52,320.09 | $53,782.95 | -$1,462.86 |

Least efficient spenders:

| Team | Wins | Actual CPW | Predicted FMV CPW | CPW Gap |
|---|---:|---:|---:|---:|
| CHI | 10 | $147,319.70 | $130,368.15 | $16,951.55 |
| ATL | 30 | $49,697.83 | $40,550.31 | $9,147.52 |
| DAL | 10 | $114,788.50 | $106,331.66 | $8,456.84 |
| LAS | 21 | $62,683.67 | $54,379.86 | $8,303.81 |
| LVA | 30 | $46,728.77 | $42,285.46 | $4,443.30 |
| MIN | 34 | $42,789.71 | $38,761.88 | $4,027.83 |

This team-level ROI metric shows which teams appear to be getting wins more cheaply than the model-implied fair market value, and which teams appear to be paying a premium per win.

---

## 4. Final Results

The final results focus on player-level and team-level valuation outputs.

At the player level, the model compares each player's actual salary with player's predicted fair market value (FMV). Players are classified as underpaid, fair, or overpaid using a ±15% valuation threshold based on `value_pct`.

The 2025 holdout set contains:

| Valuation Tag | Count |
|---|---:|
| Underpaid | 107 |
| Fair | 68 |
| Overpaid | 48 |

Top undervalued players by valuation gap include:

| Player | Team | Group | Actual Salary | Predicted FMV Salary | Valuation Gap |
|---|---|---|---:|---:|---:|
| Amy Okonkwo | DAL | veteran | $2,915 | $54,734.31 | 1777.7% |
| Odyssey Sims | TOT | veteran | $7,949 | $138,897.39 | 1647.4% |
| Grace Berger | TOT | hardship | $1,666 | $19,834.81 | 1090.6% |
| Chloe Bibby | TOT | veteran | $3,887 | $42,231.84 | 986.5% |
| Haley Jones | TOT | hardship | $4,998 | $52,447.53 | 949.4% |

Top overvalued players by valuation gap include:

| Player | Team | Group | Actual Salary | Predicted FMV Salary | Valuation Gap |
|---|---|---|---:|---:|---:|
| Yvonne Anderson | MIN | veteran | $85,000 | $12,216.42 | -85.6% |
| Moriah Jefferson | CHI | veteran | $145,500 | $22,577.85 | -84.5% |
| Cecilia Zandalasini | GSV | hardship | $100,000 | $21,231.10 | -78.8% |
| Julie Allemand | LAS | hardship | $85,000 | $19,240.64 | -77.4% |
| Karlie Samuelson | MIN | veteran | $118,450 | $34,809.56 | -70.6% |

At the team level, the model identifies teams whose actual spending per win is below or above the model-implied fair market benchmark. Connecticut shows the largest efficient-spending gap, while Chicago shows the largest inefficient-spending gap.

The feature importance results show that `start_rate` is the strongest driver in the final Random Forest model, followed by `pts_vet`, `fga`, `pca1`, and `group_hardship`, etc.


The final results show that the model can estimate player salary, identify player-level market (in)efficiencies, and convert those predictions into team-level spending efficiency thur CPW.

---

## 5. Final Model Choice and Justification

The final selected model is a tuned Gradient Boosting regression model.

After filtering duplicate salary records so that each player keeps one primary salary entry, the final feature set was updated with four additional features: `pts_hardship`, `ws_hardship`, `ws_vet`, and `group_other`. Because the training data and feature set changed, the model tuning comparison was rerun.

During model tuning, we compared Random Forest and Gradient Boosting using only the 2021–2024 training period. The 2025 holdout set was reserved for final testing. Models were selected using cross-validated RMSE, with RMSE treated as the primary model-selection metric.

Gradient Boosting was selected because it performed slightly better than Random Forest under the RMSE-first tuning rule. This fits the structure of the salary prediction problem: WNBA salary is unlikely to be explained by a purely linear relationship with box-score production alone. Salary may depend on player role, playing time, scoring production, contract group, team context, and nonlinear interactions among these variables. Also, `model_comparison fixed.ipynb` also showed that Gradient Boosting had a smaller train-validation gap than Random Forest, suggesting less overfitting in this tuning setup.

The model tuning and comparison details can be found in:

- `notebooks/Deduplicated/model_tuning fixed.ipynb`
- `notebooks/model_comparison fixed.ipynb`

We also compared the final model against several baseline models:

- Dummy Heuristic(Mean)
- Single-feature baseline(pts)
- Linear Regression(OLS)
- Ridge Regression
- Decision Tree Regressor
- Random Forest Regressor

The baseline comparison can be found in:

- `notebooks/modeling_baselines.ipynb`

The final visuals notebook also includes a lift analysis comparing the production Random Forest model against these historical baselines using both RMSE and the business-facing Cost Per Win valuation metric.

The lift analysis can be found in:

- `notebooks/final_visuals.ipynb`

The final production pipeline is implemented in:

- `notebooks/final_pipeline.ipynb`

The serialized final model is saved as:

- `artifacts/final_model.joblib`

The final prediction and metric outputs are saved under:

- `results/final/`

---

## 6. Limitations

- Salary depends on more than player performance. For instance, contract timing, veteran status, draft status, free agency, injuries, roster needs, salary cap rules, and team strategy may not be fully captured by the available features.

- Hardship contracts are difficult to model because their salaries are often driven by contract rules or roster mechanisms rather than on-court performance alone, leads to much higher MAPE for the hardship group.

- Missing contract-specific or context-specific variables may cause the model to classify some players as underpaid or overpaid even when their salaries are reasonable in context.

- Players listed under `TOT` represent multi-team or aggregate season records. They are useful for player-level valuation but excluded from team-level spending analysis because they cannot be assigned to only one team.

---

## 7. Next Steps

- Add off-court value features, such as social media presence, sponsorship activity, and player marketability.

- Add more contract-specific features, such as contract length, free agency status, draft position, and injury history.

- Collect player-team split data for multi-team players. This would allow `TOT` players to be assigned to the team where they played the most games, improving team-level CPW analysis.



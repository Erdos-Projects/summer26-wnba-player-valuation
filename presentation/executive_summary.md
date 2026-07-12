# Final Executive Summary

## 1. Problem Definition

The WNBA is one of the fastest-growing sports markets, with sponsorship deals surging over 40%, in-arena attendance at a 22-year high, and a 130% increase in young viewers, all in the last year (Her Hoop Stats, 2025). As sponsorship and attendance grow, teams need better ways to evaluate whether player salaries align with on-court production and whether rosters are being built efficiently.

This project aims to answer the following questions:

- Which players appear underpaid or overpaid relative to their on-court production?
- Are teams spending wisely relative to the wins they produce?

To answer these questions, we build a salary valuation model using data from the 2021–2024 seasons to train our model, with the 2025 season as a holdout dataset to estimate each player's predicted salary.

> **Fair Market Value (FMV):** the salary our model predicts a player should earn based on on-court production.

The analysis operates at two levels:

- **Player-level valuation:** compare each player's actual salary with their predicted FMV salary to identify players who appear underpaid or overpaid relative to the model.
- **Team-level ROI valuation:** compare actual Cost Per Win (CPW) with predicted FMV CPW. Actual CPW is calculated as actual team payroll divided by team wins; predicted FMV CPW is calculated as the sum of model-predicted player salaries for each team divided by team wins. This gives a team-level measure of whether a roster is spending more or less per win than its model-implied fair market value.

This project provides a quantitative framework for evaluating player valuation and team spending efficiency. Results should be interpreted as model-based estimates rather than definitive judgments about player quality or contract fairness.

---

## 2. Data Sources

This project uses public WNBA salary, player performance, team context, and team standings data. The modeling data covers the 2021–2025 seasons, with 2021–2024 used for model training and 2025 used as the final holdout evaluation season.

The salary data comes from Her Hoop Stats, while the player statistics, team statistics, and standings data come from Basketball Reference. These raw annual files are combined, cleaned, and transformed through the project's preprocessing and feature engineering notebooks (`notebooks/03_preprocessing_pipeline.ipynb`) before training and evaluating the final model (`notebooks/08_final_pipeline.ipynb`).

Key data sources include:

| Dataset | Source | Raw Files | Unit of Analysis | Role in Project |
|---|---|---|---|---|
| WNBA salary data | Her Hoop Stats | `data/raw/salary_[year].csv` | Player-season | Salary target and salary reference |
| WNBA player advanced stats | Basketball Reference | `data/raw/[year]_advanced.csv` | Player-season | Player efficiency and advanced performance features |
| WNBA player per-game stats | Basketball Reference | `data/raw/[year]_per_game.csv` | Player-season | Per-game production features |
| WNBA player total stats | Basketball Reference | `data/raw/[year]_totals.csv` | Player-season | Volume and counting-stat features |
| WNBA team advanced stats | Basketball Reference | `data/raw/[year]_advanced-team.csv` | Team-season | Team-level context features |
| WNBA standings | Basketball Reference | `data/raw/[year]_wnba_standings.csv` | Team-season | Team wins used for Cost Per Win analysis |

`[year]` refers to each season from 2021 through 2025. The combined dataset contains 965 player-season rows across all five seasons: 742 used for training (2021–2024) and 223 used for final evaluation (2025).

---

## 3. KPI Reporting

### 3.1 Model Prediction KPIs

The final model is evaluated on the 2025 holdout set by comparing actual salary with model-predicted salary.

The primary KPI is **RMSE**, which reports the typical size of the model's salary prediction error in dollars. A lower RMSE means the model's predicted salaries are closer to actual salaries.

The secondary KPI is **MAPE**, which reports prediction error as a percentage of actual salary. This helps compare errors across players with different salary levels.

Final model KPI results on the 2025 holdout set:

| Metric | Value |
|---|---:|
| RMSE | $41,368.60 |
| MAE | $30,376.25 |
| MAPE | 128.70% |
| R² | 0.6342 |

The aggregate MAPE of 128.70% is heavily skewed by hardship contracts, whose CBA-fixed wages bear no relationship to on-court production. Excluding hardship contracts, MAPE drops substantially, as shown in the contract group breakdown below.

Because salary rules vary across contract groups, we also report MAPE by contract group as a secondary fairness check:

| Contract Group | n | Median Actual Salary | MAPE |
|---|---:|---:|---:|
| hardship | 20 | $6,630.50 | 289.7% |
| veteran | 133 | $78,831.00 | 156.0% |
| controlled | 13 | $66,079.00 | 64.1% |
| unknown | 17 | $78,066.00 | 37.3% |
| rookie | 40 | $75,276.00 | 17.1% |

The high MAPE for hardship contracts confirms that these contracts are structurally difficult to predict from performance features alone, as their salaries are governed by CBA minimums rather than on-court output.

### 3.2 Business KPI: Team-Level Cost Per Win

The project also reports a team-level ROI valuation via Cost Per Win (CPW), which measures how much salary spending is associated with each team win.

```
Actual CPW = actual total team payroll / team wins
Predicted (FMV) CPW = predicted total team payroll / team wins
CPW gap = actual CPW - predicted FMV CPW
```

A **negative** CPW gap means a team's actual cost per win is below the model-implied fair market cost per win, suggesting more efficient spending. A **positive** CPW gap means a team's actual cost per win is above the model-implied fair market cost per win, suggesting less efficient spending.

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

---

## 4. Final Results

The final results focus on player-level and team-level valuation outputs.

At the player level, the model compares each player's actual salary with their predicted fair market value (FMV). Players are classified as underpaid, fair, or overpaid using a ±15% valuation threshold based on `value_pct = residual / actual`.

The 2025 holdout set contains:

| Valuation Tag | Count |
|---|---:|
| Underpaid | 107 |
| Fair | 68 |
| Overpaid | 48 |

Top undervalued players by valuation gap:

| Player | Team | Group | Actual Salary | Predicted FMV Salary | Valuation Gap |
|---|---|---|---:|---:|---:|
| Amy Okonkwo | DAL | veteran | $2,915 | $54,734.31 | 1777.7% |
| Odyssey Sims | TOT | veteran | $7,949 | $138,897.39 | 1647.4% |
| Grace Berger | TOT | hardship | $1,666 | $19,834.81 | 1090.6% |
| Chloe Bibby | TOT | veteran | $3,887 | $42,231.84 | 986.5% |
| Haley Jones | TOT | hardship | $4,998 | $52,447.53 | 949.4% |

> **Note:** extreme valuation gaps for the most undervalued players are driven by CBA-minimum and hardship contract salary floors, not modeling error. Players earning near the league minimum will always appear statistically underpaid relative to their production under this framework.

Top overvalued players by valuation gap:

| Player | Team | Group | Actual Salary | Predicted FMV Salary | Valuation Gap |
|---|---|---|---:|---:|---:|
| Yvonne Anderson | MIN | veteran | $85,000 | $12,216.42 | -85.6% |
| Moriah Jefferson | CHI | veteran | $145,500 | $22,577.85 | -84.5% |
| Cecilia Zandalasini | GSV | hardship | $100,000 | $21,231.10 | -78.8% |
| Julie Allemand | LAS | hardship | $85,000 | $19,240.64 | -77.4% |
| Karlie Samuelson | MIN | veteran | $118,450 | $34,809.56 | -70.6% |

Two patterns stand out:

- Players on hardship or rookie-scale contracts dominate the undervalued list because CBA salary floors cap their earnings regardless of on-court production.
- Veterans dominate the overpaid list because their current salaries still reflect past performance, while their recent on-court production has declined.

The feature importance results show that `start_rate` is the strongest driver in the final tuned Random Forest model. The top five features by importance are `start_rate`, `pts_vet`, `fg_per_g`, `avail_rate`, and `pts_rookie`. The full feature importance chart is available at `results/final/figures/feature_importance.png`.

Together, these results show that salary inefficiencies in the WNBA are systematic and measurable — clustering by contract type at the player level and by roster construction strategy at the team level.

---

## 5. Final Model Choice and Justification

The final selected model is a tuned Random Forest regression model.

During model tuning, we compared Random Forest and Gradient Boosting using only the 2021–2024 dataset. The 2025 holdout set was reserved for final testing. Models were selected using cross-validated RMSE as the primary model-selection metric.

Random Forest was selected because it performed best within the tuning scope and fits the structure of the salary prediction problem. WNBA salary is unlikely to be explained by a purely linear relationship with box-score production alone. Salary depends on player role, playing time, scoring production, contract group, team context, and nonlinear interactions among these variables. Random Forest provides the best balance between predictive performance and nonlinear modeling flexibility. It also offers interpretability through feature importance scores, which are directly useful for communicating which factors drive the model's salary estimates to non-technical stakeholders.

Model tuning and comparison details can be found in:

- `notebooks/05_model_comparison.ipynb`

We also compared the final model against several baseline models:

- Dummy Heuristic (Mean)
- Single-feature baseline (pts)
- Linear Regression (OLS)
- Ridge Regression
- Decision Tree Regressor
- Random Forest Regressor (untuned)

The baseline comparison can be found in:

- `notebooks/04_modeling_baseline.ipynb`

The final visuals notebook also includes a lift analysis comparing the production Random Forest model against these baselines using both RMSE and the business-facing Cost Per Win metric:

- `notebooks/09_final_visuals.ipynb`

The final pipeline is implemented in:

- `notebooks/08_final_pipeline.ipynb`

The serialized final model is saved as:

- `artifacts/final_model.joblib`

Final prediction and metric outputs are saved under:

- `results/final/`

---

## 6. Deployment and Use in Context

This section describes how the model's outputs can be used in practice by the three primary stakeholder groups identified in section 1.

### 6.1 GMs and Front Offices

The most direct application is **free agency targeting**. At the start of each offseason, a front office can load `results/final/final_predictions_enriched.csv` and filter for players whose predicted FMV exceeds their most recent actual salary by more than 15% — the resulting list is a prioritized set of players who may be acquirable below their statistical value. Sorting by valuation gap rather than raw dollar residual ensures the comparison is fair across rookie-scale and veteran-scale contracts alike.

The team-level CPW output in `results/final/team_cpw.csv` gives a complementary roster-level view. A front office can benchmark their own CPW gap against the rest of the league and identify whether their inefficiency is concentrated in a few high-salary veterans or distributed more broadly across the roster. Teams with a large positive CPW gap — paying significantly more per win than the model implies is fair — may benefit most from restructuring veteran contracts in the next CBA cycle.

### 6.2 Agents and Players

An agent negotiating a contract renewal for an underpaid player can use the model's FMV estimate as a quantitative anchor in contract discussions. This is particularly valuable for players on hardship or rookie-scale contracts, whose salaries are structurally below their production value due to CBA minimums rather than any assessment of their worth. A model-backed FMV number provides an empirical counterpoint to a team's offer that goes beyond comparable contracts alone.

For players approaching free agency, the valuation gap percentage (`value_pct` in the enriched predictions file) provides a straightforward summary statistic: a player with a valuation gap of +500% has a strong quantitative case that her current contract undervalues her production.

### 6.3 Operational Notes

- **Retraining cadence:** The model should be retrained each offseason after new salary and performance data becomes available. WNBA market conditions, salary cap levels, and roster compositions shift year to year, and a model trained on stale data will produce increasingly unreliable FMV estimates over time.

- **Hardship contract interpretation:** FMV predictions for players on hardship contracts should be treated with caution. As documented in section 3, hardship salaries are CBA-fixed and do not reflect on-court production. The model's predictions for these players are not wrong — they reflect what the player would likely earn on a standard contract — but the gap between predicted and actual salary is driven by contract mechanics, not market mispricing in the traditional sense.

- **Using the serialized model:** The final trained model is available at `artifacts/final_model.joblib` and can be loaded using `joblib.load()`. The model bundle includes the feature names list, model parameters, and evaluation metrics needed to reproduce predictions on new data. New data must be preprocessed to match the feature schema in `data/processed/X_train_processed.csv` before generating predictions.

---

## 7. Limitations

- **Salary depends on more than on-court performance.** Contract timing, veteran status, injury history, roster needs, and team strategy all influence salary in ways not fully captured by box-score statistics.

- **Hardship contracts are structurally unlearnable.** Their salaries are governed by CBA minimums rather than on-court production, leading to very high MAPE for this contract group. This is a known structural limitation rather than a modeling failure.

- **WNBA contract structures and salary cap rules evolve quickly**, meaning a model trained on 2021–2024 data may not fully reflect the current market.

- **Players listed under `TOT` represent multi-team or aggregate season records.** They are included in player-level valuation but excluded from team-level spending analysis because their statistics cannot be attributed to a single team.

---

## 8. Next Steps

- **Add off-court value features**, such as social media following or endorsement deal counts, which are increasingly trackable through public data and are likely correlated with veteran salary premiums.

- **Add contract-specific features**, such as contract length, free agency status, draft position, and injury history, to better distinguish performance-based salary from market-condition-based salary.

- **Collect player-team split data for multi-team players.** This would allow players currently listed as `TOT` to be assigned to the team where they played the most games, improving both player-level attribution and team-level CPW analysis.

- **Extend to salary trajectory prediction.** Rather than predicting a single-season point estimate, a future version of this model could predict how a player's FMV is likely to change over multiple seasons, making it a more actionable tool for multi-year contract negotiations.

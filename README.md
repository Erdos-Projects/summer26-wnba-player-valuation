# WNBA Fair Market Value (FMV) Engine

Predicting what WNBA players should earn based on performance metrics to optimize roster construction.

A data science project developed for the Summer 2026 Erdös Institute Data Science Boot Camp.

## Authors
* Elena Axinn
* Aklima Khanam
* Nahyun Lee
* Ling Le
* Tiana Johnson

## Project Overview

The WNBA is experiencing unprecedented commercial growth, with skyrocketing viewership, attendance, and sponsorship revenue. However, team salary caps remain strictly regulated under the Collective Bargaining Agreement (CBA). Front offices must optimize roster construction under tight financial ceilings without sophisticated statistical tools.

This project delivers a data-driven Fair Market Value (FMV) engine that predicts what a WNBA player should earn based on on-court production. By matching predicted salaries against actual contracts, our framework solves two core problems:
1. Identifying Market Inefficiencies: Pinpointing statistically undervalued assets or overvalued veterans.
2. Evaluating Franchise ROI: Quantifying which franchises generate the highest on-court returns per dollar spent on player payroll.

### Key Performance Indicators (KPIs)
* Primary Predictive KPI: Root Mean Squared Error (RMSE) to track overall dollar-value variance and penalize costly star-player miscalculations.
* Secondary Fairness KPI: Mean Absolute Percentage Error (MAPE) to evaluate equity across varying contract scales (rookie vs. supermax).
* Business ROI KPI: Cost Per Win Share (CPWS) to establish franchise spending efficiency benchmarks.

## Dataset and Feature Engineering

The engine integrates multi-source historical performance and compensation data covering the 2021-2025 WNBA seasons, compiling 250+ player records and over 150 raw attributes sourced from HerHoopStats and Basketball Reference.

* Core Feature Selection: Using Lasso, SelectKBest, and Random Forest estimators, we narrowed the field down to 25 primary performance drivers. Start rate, points per game, and field goal attempts emerged as the most salient indicators.
* Lagged Variables: To simulate realistic front-office environments, features are engineered as lagged variables, matching concurrent pay with preceding production histories.

## Leakage-Proof Validation and Robustness

To ensure validity and prevent look-ahead bias, we implemented a specialized chronological splitting mechanism:
* Train Set: Sequential historical pools from 2021-2024.
* Test Holdout: The entire 2025 season.

### Cross-Validation Folds
Our custom cross-validation architecture groups records strictly by player identity to account for multi-year data replication:
* Fold 1: Train on 2021 and validate on 2022.
* Fold 2: Train on 2021-2022 and validate on 2023.
* Fold 3: Train on 2021-2023 and validate on 2024.

### Stress Test Diagnostics
Our framework underwent rigorous testing to ensure enterprise-level stability:
1. Noise Injection: Prediction shifts scaled linearly with white noise. The Random Forest model proved highly robust against data perturbations.
2. Extreme Profiles: Inputting superstar vs. fringe profiles yielded logical, well-bounded projections within actual league parameters.
3. Structural Limitations (Hardship Contracts): Diagnostics revealed that hardship contracts are structurally unlearnable due to rigid CBA wage floors ($6.6k median) that ignore on-court metrics. The engine functions beautifully for rookie, controlled, and veteran tiers, but hardship records must be evaluated under a separate regime.

## Modeling and Performance Summary

We tested several architectures against naive baseline heuristics:

* Tuned Random Forest Regressor: This was our top performer. It achieved a 2025 holdout RMSE of $41,368.60 and a Mean Absolute Error (MAE) of $30,376.25. It explains a moderate amount of variance with an R-squared of 0.634 and shows excellent stability under stress.
* Linear Regression and Ridge: These provided a useful diagnostic linear baseline but failed to capture nonlinear efficiency thresholds.
* Dummy Heuristic (Mean): This served as a trivial baseline reference standard that simply predicts the historical mean salary.

Note on MAPE: The global MAPE sits at roughly 128.7%. This distortion is primarily driven by small contract denominators (like hardship contracts) inflating percentage errors, rather than large dollar-value failures. The framework prioritizes dollar-denominated RMSE for evaluation.

## Repository Architecture

```text
.
├── artifacts
│   ├── best_model.joblib                  # Top performing model found during hyperparameter tuning
│   └── final_model.joblib                 # Final serialized model bundle trained for production deployment[cite: 1]
├── data
│   ├── processed
│   │   ├── engineered_features_only_2025.csv # Newly generated features specifically for the 2025 season
│   │   ├── feature_engineered_full_2025.csv  # Combined raw data and engineered features for the 2025 test set
│   │   ├── feature_names.csv                 # Text list tracking final features kept for modeling[cite: 1]
│   │   ├── model_ready.csv                   # Fully engineered and merged multi-year training dataset
│   │   ├── player_lookup_test_2025.csv       # Name/team metadata index to map predictions back to 2025 players[cite: 1]
│   │   ├── player_lookup_train.csv           # Name/team metadata index to map predictions back to training players[cite: 1]
│   │   ├── preprocessor.pkl                  # Serialized Scikit-Learn preprocessing scaling/encoding pipeline
│   │   ├── X_processed.csv                   # Consolidated, scaled training features
│   │   ├── X_test_2025_processed.csv         # Final processed features for the untouched 2025 holdout set[cite: 1]
│   │   ├── X_train_processed.csv             # Final processed training features (2021-2024)[cite: 1]
│   │   ├── y_salary.csv                      # Consolidated target variable (salaries) for full training set
│   │   ├── y_test_2025.csv                   # Target variable (salaries) for the 2025 holdout set[cite: 1]
│   │   └── y_train.csv                       # Target variable (salaries) for the training set split[cite: 1]
│   └── raw
│       ├── 2021_advanced-team.csv            # Advanced team statistics per season (2021-2025)
│       ├── 2021_advanced.csv                 # Advanced player statistics per season (2021-2025)
│       ├── 2021_per_game.csv                 # Traditional per-game player box scores (2021-2025)
│       ├── 2021_totals.csv                   # Cumulative seasonal player stat totals (2021-2025)
│       ├── 2021_wnba_standings.csv           # Team records, win percentages, and standings (2021-2025)
│       │   ... [Repeated for seasons 2022, 2023, 2024, 2025]
│       └── salary_2021.csv                   # Scraped individual player contract salaries (2021-2025)
├── notebooks
│   ├── 00_data_assessment.ipynb          # Initial data health check, shape analysis, and missing value logs
│   ├── 01_eda_multiyear.ipynb            # Exploratory analysis tracking salary distributions and correlations over time
│   ├── 02_assessing_learnability_multiyear.ipynb # Baseline data signal tests and feature importance exploration
│   ├── 03_preprocessing_pipeline.ipynb   # Cleans data, handles scaling, encodes categories, and exports to data/processed
│   ├── 04_modeling_baseline.ipynb        # Builds an initial simple model to establish performance benchmarks
│   ├── 05_model_comparison.ipynb         # Evaluates multiple algorithms (Linear, Tree-based) against metrics
│   ├── 06_scrape_data.ipynb              # Python scraping scripts for capturing player contract and ranking records
│   ├── 07_interpretability.ipynb         # Features SHAP/feature importance analysis explaining model choices
│   ├── 08_final_pipeline.ipynb           # Production script training final model bundle and evaluating 2025 holdout[cite: 1]
│   ├── 09_final_visuals.ipynb            # Generates executive reporting graphs, charts, and diagnostic plots
│   └── 10_stress_test.ipynb              # Runs edge-case checks and data perturbation tests to ensure model stability
├── presentation
│   ├── executive_summary.md              # High-level overview of findings, key takeaways, and engine mechanics
│   └── WNBA_FMV_Engine.pdf               # Slide deck presentation covering model business value and insights
├── README.md                             # Main project documentation, setup instructions, and architecture layout
├── results
│   ├── figures
│   │   ├── feature_importance.png         # Plot highlighting the metrics most critical to salary predictions
│   │   ├── mape_by_contract_group.png     # Error rate analysis broken down by player contract types (veteran vs rookie)
│   │   ├── player_valuation_leaderboard.png # Visual identifying the most underpaid and overpaid players
│   │   ├── predicted_vs_actual.png        # Scatter plot measuring predicted contract value against true salary
│   │   ├── team_cpw_comparison.png        # Cost-Per-Win efficiency analysis across franchises
│   │   └── team_efficiency_leaderboard.png # Ranking of franchises optimized by output vs total payroll expenditure
│   ├── final_metrics.csv                 # Core holdout performance metrics (RMSE, MAE, MAPE, R2)[cite: 1]
│   ├── final_model_metadata.json         # Human-readable parameters, environment details, and performance summaries[cite: 1]
│   ├── final_predictions_enriched.csv    # Holdout outputs mapped back to names, teams, and percentage errors
│   ├── final_predictions.csv             # Raw model prediction values paired with actual 2025 holdout salaries[cite: 1]
│   ├── model_comparison.csv              # Aggregated metrics contrasting different algorithm iterations
│   ├── stress_test_log.csv               # Audit trail documenting edge-case evaluation data behaviors
│   ├── team_cpw.csv                      # Underlying data calculations for Cost-Per-Win metrics
│   ├── tuned_predictions.csv             # Holdout predictions generated using hyperparameter optimization
│   └── tuning_results.csv                # Metrics tracking across grid-search or randomized search iterations
└── run.sh                                # Bash execution script automating the entire pipeline end-to-end
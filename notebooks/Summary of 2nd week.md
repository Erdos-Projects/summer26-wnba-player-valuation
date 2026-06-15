# Data Audit Summary

## 1. Project Goal
This project is aimed to build a data-driven Fair Market Value(FMV) engine to predict what a WNBA player should earn based on their on-court evaluation. The model should resolve two things:
- Team - level: Evaluate Franchise ROI (Return on Investment)
- Player - level: Identify underpaid and overpaid players

## 2. Data Sources and Coverage
The current dataset contains data from 2021 to 2025 that can be splitted into two categories:

- Player - level:
 - Salary data
 - Totals data
 - Per - game statistics
 - Advanced statistics

- Team - level:
 - Team standings
 - Team advanced statistics

## 3. Data Granularity and Merge Issues
A major issue in this project is that the raw files are not all recorded at the same level of observation. The salary data is recorded at the player-season level. In contrast, the team standings and team advanced statistics are team-season level datasets.Since the final modeling goal is player salary prediction, the final dataset should use one row per player per season. Therefore, team-level variables need to be merged into the player-level dataset by team and season. After merging, team statistics are repeated for all players on the same team, so they should be interpreted as team context rather than individual player performance. 

There are some main merge risks:

- Duplicated player-season rows caused by players changing teams during the season
- Team-level statistics being repeated across multiple player rows after merging
- Different contract categories may not be directly comparable (at least, I don't know how we can compare a rookie contract with veteran.)

## 4. Preprocessing

The preprocessing work focused on preparing the merged WNBA dataset for modeling. The processed data files are stored in the `data/processed/` directory. This folder contains the main outputs needed for modeling, including `model_ready.csv`, `X_processed.csv`, `y_salary.csv`, `feature_names.csv`, and `preprocessor.pkl`.

`model_ready.csv` is the full engineered dataframe before the final modeling split. `X_processed.csv` contains the model-ready feature matrix, while `y_salary.csv` stores the salary target values. `feature_names.csv` lists the final selected features, and `preprocessor.pkl` stores the fitted preprocessing object, so the same transformation steps can be reused later without rebuilding the preprocessing process from scratch.

## 5. Feature Selection Summary

The feature selection notebook used several methods to explore which variables are useful for salary prediction, including SelectKBest, Lasso, and Random Forest feature importance. These methods were used as a guide to identify stable predictors, rather than as the only source of the final feature list.

The results showed that player production and playing-time variables were consistently important. Examples include `pts`, `mp`, `g`, `fg`, `fga`, `ft`, `fta`, `blk`, `tov`, and related shooting or availability variables.

The final processed feature list is stored in `feature_names.csv`. This list is not exactly the same as the feature selection notebook outputs, because it also includes engineered variables and contract group indicators such as `pca1`, `pts_rookie`, `pts_vet`, `ws_rookie`, `group_rookie`, and `group_veteran`.

## 6. Engineered Features Summary

The `feature_engineering.ipynb` notebook created candidate features related to player role, availability, efficiency, contract type, and production by contract group. These features were designed to capture salary-related patterns.

Not all engineered features were kept in the final model-ready dataset. The final selected engineered features in `feature_names.csv` include variables such as `avail_rate`, `start_rate`, `team_min`, `pts_rookie`, `pts_vet`, `ws_rookie`, and `pca1`.

Overall, the engineering step created new candidate predictors, while the final processed dataset kept only the selected subset used for modeling.







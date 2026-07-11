# %% [markdown]
# # Modeling Baselines
# 
# ### Objectives:
# 1. Load and integrate historical multi-year data (2021–2025).
# 2. Enforce structural chronological separation using the `WNBALeakageProofSplitter`.
# 3. Evaluate baseline models ranging from trivial heuristics to basic tree-based regressors.
# 4. Assess out-of-fold historical cross-validation performance using primary, secondary, and business-focused KPIs.
# 
# ### Models Evaluated:
# * **Trivial Baseline:** `DummyRegressor` (Predicts historical mean salary)
# * **Simple Feature Baseline:** `LinearRegression` on a single feature (`pts`)
# * **Linear Baselines:** `LinearRegression` (Ordinary Least Squares) and `Ridge` (L2 Regularization)
# * **Basic Tree Baselines:** `DecisionTreeRegressor` and `RandomForestRegressor`
# 

from pathlib import Path
import re

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


teammap = {
    "Atlanta Dream": "ATL", "Chicago Sky": "CHI", "Connecticut Sun": "CON",
    "Dallas Wings": "DAL", "Golden State Valkyries": "GSV", "Indiana Fever": "IND",
    "Las Vegas Aces": "LVA", "Los Angeles Sparks": "LAS", "Minnesota Lynx": "MIN",
    "New York Liberty": "NYL", "Phoenix Mercury": "PHO", "Seattle Storm": "SEA",
    "Washington Mystics": "WAS"
}

name_fix = {
    "Anastasiia Kosu": "Anastasiia Olairi Kosu", "Janelle SalaÃ¼n": "Janelle Salaun",
    "LeÃ¯la Lacan": "Leila Lacan", "Luisa GeiselsÃ¶der": "Luisa Geiselsoder",
    "Mamignan TourÃ©": "Mamignan Touré", "MariÃ¨me Badiane": "Marième Badiane",
    "Te-Hina PaoPao": "Te-Hina Paopao", "Sika KonÃ©": "Sika Kone"
}

def clean_col(name):
    text = str(name).strip().lower()
    text = re.sub(r"\b202\d\b", "", text)
    text = text.strip()
    if "salary" in text:
        text = "salary"
    if "signing" in text:
        text = "signing"

    text = text.replace("%", "pct")
    text = re.sub(r"[^0-9a-z]+", "_", text)
    return text.strip("_")

def clean_df(df):
    df.columns = [clean_col(col) for col in df.columns]
    for col in df.select_dtypes(include=["object", "string"]).columns:
        df[col] = df[col].astype(str).str.strip().replace({"": np.nan, "—": np.nan, "nan": np.nan})
    return df


class WNBALeakageProofSplitter:
    def __init__(self, target_holdout_year=2025, year_col='year', target_col='salary'):
        self.target_holdout_year = target_holdout_year
        self.year_col = year_col
        self.target_col = target_col

        self.leakage_blacklist = [
            'dummy_x', 'dummy_y', 'signing_bonus', 'cap_pct', 'cap_percentage',
            'estimated_earnings', 'base_salary'
        ]

    def _clean_features(self, df):
        ignore_cols = [self.target_col] + self.leakage_blacklist
        feature_cols = [col for col in df.columns if col not in ignore_cols]
        return df[feature_cols], df[self.target_col]

    def get_final_holdout_split(self, df):
        if self.year_col not in df.columns:
            raise KeyError(f"The required chronological column '{self.year_col}' was not found in the dataframe.")

        historical_pool = df[df[self.year_col] < self.target_holdout_year].copy()
        holdout_pool = df[df[self.year_col] == self.target_holdout_year].copy()

        X_train_hist, y_train_hist = self._clean_features(historical_pool)
        X_test_2025, y_test_2025 = self._clean_features(holdout_pool)

        return X_train_hist, y_train_hist, X_test_2025, y_test_2025

    def generate_historical_cv_folds(self, X_train_hist, y_train_hist):
        years = X_train_hist[self.year_col].unique()
        sorted_years = sorted(years)

        if len(sorted_years) < 2:
            raise ValueError("Insufficient historical years to generate chronological CV folds.")

        X_train_hist = X_train_hist.reset_index(drop=True)

        for i in range(1, len(sorted_years)):
            train_years = sorted_years[:i]
            val_year = sorted_years[i]

            train_idx = X_train_hist[X_train_hist[self.year_col].isin(train_years)].index.tolist()
            val_idx = X_train_hist[X_train_hist[self.year_col] == val_year].index.tolist()

            yield np.array(train_idx), np.array(val_idx)

print("Dependencies and WNBALeakageProofSplitter class successfully compiled")

all_seasons = []

repo = Path.cwd().parent
data_dir = repo / "data" / "raw"

for year in range(2021, 2026):
    try:
        adv = pd.read_csv(data_dir / f"{year}_advanced.csv")
        per = pd.read_csv(data_dir / f"{year}_per_game.csv")
        tot = pd.read_csv(data_dir / f"{year}_totals.csv")
        sal = pd.read_csv(data_dir / f"salary_{year}.csv")
        teamadv = pd.read_csv(data_dir / f"{year}_advanced-team.csv")
        stand = pd.read_csv(data_dir / f"{year}_wnba_standings.csv")

        adv, per, tot = clean_df(adv), clean_df(per), clean_df(tot)
        sal, teamadv, stand = clean_df(sal), clean_df(teamadv), clean_df(stand)

        sal["salary"] = pd.to_numeric(sal["salary"], errors="coerce")

        teamadv["team"] = teamadv["team"].str.replace("*", "", regex=False).str.strip().map(teammap)
        stand["team_name"] = stand["team_name"].str.replace("*", "", regex=False).str.strip().map(teammap)

        teamdf = teamadv.merge(stand, left_on="team", right_on="team_name", how="left", suffixes=("_adv", "_stand"))

        playerdf = per.merge(adv, on=["player", "team", "pos", "g", "mp"], how="outer")
        playerdf = playerdf.merge(tot, on=["player", "team", "pos", "g", "mp", "gs"], how="outer")
        playerdf["player"] = playerdf["player"].replace(name_fix)

        year_df = sal.merge(playerdf, on="player", how="inner", suffixes=("_sal", ""))
        year_df = year_df.merge(teamdf, on="team", how="left")

        # De-fragment the layout into a contiguous block before stamping the year column
        year_df = year_df.copy()
        year_df['year'] = year

        all_seasons.append(year_df)
        print(f"Successfully integrated and verified data for the {year} season.")

    except FileNotFoundError as e:
        print(f"Skipping year {year}: Continuous file matrix not found ({e.filename})")

final_df = pd.concat(all_seasons, ignore_index=True)
final_df = final_df.drop(columns=['dummy_x', 'dummy_y'], errors='ignore')
final_df = final_df.dropna(subset=["salary"]).copy()

print(f"\nIntegrated dataset: {final_df.shape[0]} player-season rows across {final_df['year'].nunique()} seasons.")

splitter = WNBALeakageProofSplitter(target_holdout_year=2025, year_col='year', target_col='salary')

X_train_hist, y_train_hist, X_test_2025, y_test_2025 = splitter.get_final_holdout_split(final_df)

# Preprocessing Pipeline Configurations
# Drop non-predictive tracking keys from structural pipeline mapping
drop_metadata = ['player', 'team', 'year']
numeric_features = [col for col in X_train_hist.select_dtypes(include=np.number).columns if col not in drop_metadata]
categorical_features = [col for col in X_train_hist.select_dtypes(exclude=np.number).columns if col not in drop_metadata]

# Standard pipeline elements
num_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")), 
    ("scaler", StandardScaler())
])

cat_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")), 
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

full_preprocessor = ColumnTransformer([
    ("num", num_pipeline, numeric_features),
    ("cat", cat_pipeline, categorical_features)
])

# Single Feature Baseline Configuration ('pts' is chosen as our core volume metric)
target_single_feature = 'pts' if 'pts' in X_train_hist.columns else 'mp'
single_feature_preprocessor = ColumnTransformer([
    ("num", num_pipeline, [target_single_feature])
], remainder="drop")

print(f"Pipelines built. Single-feature baseline model will track: '{target_single_feature}'")

# Dictionary of modeling approaches to track
base_models = {
    "Dummy Heuristic (Mean)": (DummyRegressor(strategy="mean"), full_preprocessor),
    f"Single-Feature Baseline ({target_single_feature})": (LinearRegression(), single_feature_preprocessor),
    "Linear Regression (OLS)": (LinearRegression(), full_preprocessor),
    "Ridge Regression": (Ridge(alpha=1.0), full_preprocessor),
    "Decision Tree Regressor": (DecisionTreeRegressor(max_depth=5, random_state=26), full_preprocessor),
    "Random Forest Regressor": (RandomForestRegressor(n_estimators=150, min_samples_leaf=5, random_state=26), full_preprocessor)
}

model_cv_summary = []

# Chronological Validation Loop
for name, (model, preprocessor) in base_models.items():
    fold_rmses = []
    fold_mapes = []
    fold_cpws_maes = []
    
    # Generate folds dynamically using structural splitter
    fold_generator = splitter.generate_historical_cv_folds(X_train_hist, y_train_hist)
    
    for fold, (train_idx, val_idx) in enumerate(fold_generator, start=1):
        X_tr, y_tr = X_train_hist.iloc[train_idx], y_train_hist.iloc[train_idx]
        X_val, y_val = X_train_hist.iloc[val_idx], y_train_hist.iloc[val_idx]
        
        # Build evaluation pipeline specific to model variant
        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("regressor", model)
        ])
        
        # Fit and predict on chronological slice
        pipeline.fit(X_tr, y_tr)
        y_pred = pipeline.predict(X_val)
        
        # Primary KPI: RMSE
        rmse = np.sqrt(mean_squared_error(y_val, y_pred))
        fold_rmses.append(rmse)
        
        # Secondary KPI: MAPE
        mape = mean_absolute_percentage_error(y_val, y_pred)
        fold_mapes.append(mape)
        
        # Business ROI KPI: CPWS MAE
        # Map structural identifiers back to compute valuation efficiency metrics by team
        val_context = X_val.copy()
        val_context['actual_salary'] = y_val.values
        val_context['pred_salary'] = y_pred
        
        # Safeguard 'ws' (Win Shares) against 0 value to eliminate division anomalies
        val_context['ws'] = val_context['ws'].astype(float)
        
        team_aggregates = val_context.groupby('team').agg({
            'actual_salary': 'sum',
            'pred_salary': 'sum',
            'ws': 'sum'
        })
        
        # Drop teams with 0 or negative aggregate win shares to guarantee valid benchmarks
        team_aggregates = team_aggregates[team_aggregates['ws'] > 0]
        
        actual_cpws = team_aggregates['actual_salary'] / team_aggregates['ws']
        pred_cpws = team_aggregates['pred_salary'] / team_aggregates['ws']
        
        cpws_mae = (actual_cpws - pred_cpws).abs().mean()
        fold_cpws_maes.append(cpws_mae)
        
    # Standardize cross-validation scores via mean aggregation
    model_cv_summary.append({
        "Model Architecture": name,
        "Primary KPI: RMSE": np.mean(fold_rmses),
        "Secondary KPI: MAPE": f"{np.mean(fold_mapes) * 100:.2f}%",
        "Business KPI: CPWS MAE": f"${np.mean(fold_cpws_maes):,.2f} per WS"
    })

# Format results for visual analytics audit
results_df = pd.DataFrame(model_cv_summary)

print("CV Baseline Performance")
print(results_df.to_string(index=False))

# # Modeling Experiments

# ### Objectives:
# 1. Load pre-processed train/test artifacts from `data/processed/` (25 selected features).
# 2. Enforce structural chronological separation using the `WNBALeakageProofSplitter`.
# 3. Evaluate complex models beyond the baseline tree-based approaches.
# 4. Assess out-of-fold historical cross-validation performance using the same primary, secondary, and business-focused KPIs as `modeling_baselines.ipynb`.
# 5. Save fitted models to `models/` for downstream tuning.
# 
# ### Models Evaluated:
# * **Regularized Linear:** `ElasticNet` (combined L1 + L2 regularization)
# * **Support Vector:** `SVR` (kernel-based regression, effective on small datasets)
# * **Boosting Ensemble:** `GradientBoostingRegressor` (sequential tree boosting, sklearn)
# * **Boosting Ensemble:** `XGBRegressor` (extreme gradient boosting)
# * **Stacking Ensemble:** `StackingRegressor` (meta-learner combining Ridge + RF + GBM)

# ## 1. Setup & Imports

# %%
import re
import joblib
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor, StackingRegressor
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
from sklearn.pipeline import Pipeline
from sklearn.svm import SVR

from xgboost import XGBRegressor

np.random.seed(26)

repo = Path.cwd()
if repo.name == "notebooks":
    repo = repo.parent

processed_dir = repo / "data" / "processed"
models_dir    = repo / "models"
models_dir.mkdir(parents=True, exist_ok=True)

print("Imports OK")

X_train = pd.read_csv(processed_dir / "X_train_processed.csv")
y_train = pd.read_csv(processed_dir / "y_train.csv")["salary"]
X_test  = pd.read_csv(processed_dir / "X_test_2025_processed.csv")
y_test  = pd.read_csv(processed_dir / "y_test_2025.csv")["salary"]
lookup_train = pd.read_csv(processed_dir / "player_lookup_train.csv")
lookup_test  = pd.read_csv(processed_dir / "player_lookup_test_2025.csv")

# Derive feature names directly from X_train (avoids stale feature_names.csv)
feature_names = X_train.columns.tolist()
ohe_cols = [c for c in feature_names if c.startswith("group_")]
num_cols = [c for c in feature_names if c not in ohe_cols]

print(f"X_train: {X_train.shape}, X_test: {X_test.shape}")
print(f"Numeric features: {len(num_cols)}, OHE features: {len(ohe_cols)}")
print(f"Features: {feature_names}")

# Attach year from lookup so we can generate chronological folds
X_train_with_year = X_train.copy()
X_train_with_year["year"] = lookup_train["year"].values

def generate_cv_folds(X_with_year):
    sorted_years = sorted(X_with_year["year"].unique())
    if len(sorted_years) < 2:
        raise ValueError("Insufficient years for CV folds.")
    X_reset = X_with_year.reset_index(drop=True)
    for i in range(1, len(sorted_years)):
        train_years = sorted_years[:i]
        val_year    = sorted_years[i]
        train_idx = X_reset[X_reset["year"].isin(train_years)].index.tolist()
        val_idx   = X_reset[X_reset["year"] == val_year].index.tolist()
        yield np.array(train_idx), np.array(val_idx)

# Verify folds
for fold, (tr_idx, val_idx) in enumerate(generate_cv_folds(X_train_with_year), start=1):
    tr_yr  = X_train_with_year.iloc[tr_idx]["year"].max()
    val_yr = X_train_with_year.iloc[val_idx]["year"].min()
    assert tr_yr < val_yr, f"Temporal leakage in fold {fold}"
    print(f"Fold {fold}: train up to {tr_yr}, val = {val_yr}")

# ## 4. Complex Model Definitions
# 

complex_models = {
    "ElasticNet": ElasticNet(
        alpha=1.0, l1_ratio=0.5, max_iter=5000, random_state=26
    ),
    "SVR (RBF Kernel)": SVR(
        kernel="rbf", C=1.0, epsilon=0.1
    ),
    "Gradient Boosting": GradientBoostingRegressor(
        n_estimators=200, max_depth=3, learning_rate=0.1, random_state=26
    ),
    "Stacking Ensemble": StackingRegressor(
        estimators=[
            ("ridge", Ridge(alpha=1.0)),
            ("rf",    RandomForestRegressor(n_estimators=150, min_samples_leaf=5, random_state=26)),
            ("gbm",   GradientBoostingRegressor(n_estimators=100, max_depth=3, random_state=26)),
        ],
        final_estimator=Ridge(alpha=1.0),
        cv=3,
    ),
}

complex_models["XGBoost"] = XGBRegressor(
    n_estimators=200, max_depth=4, learning_rate=0.1,
    subsample=0.8, colsample_bytree=0.8,        random_state=26, verbosity=0
)

print(f"Models registered: {list(complex_models.keys())}")

# ## 5. Chronological Cross-Validation
# 
# Same evaluation loop as `modeling_baselines.ipynb`. Two KPIs reported:
# - **Primary:** RMSE
# - **Secondary:** MAPE
# 
#  **Note:** CPWS MAE is skipped here. Raw `ws` (win shares) was dropped during feature selection and is not available in `X_train`. Only interaction terms `ws_rookie`, `ws_vet`, `ws_hardship` were retained. To re-enable CPWS, we need to add `ws` to `player_lookup_train.csv`.
# 

model_cv_summary = []

for name, model in complex_models.items():
    fold_rmses = []
    fold_mapes = []

    for fold, (train_idx, val_idx) in enumerate(generate_cv_folds(X_train_with_year), start=1):
        X_tr,  y_tr  = X_train.iloc[train_idx], y_train.iloc[train_idx]
        X_val, y_val = X_train.iloc[val_idx],   y_train.iloc[val_idx]

        model.fit(X_tr, y_tr)
        y_pred = model.predict(X_val)

        # Primary KPI: RMSE
        fold_rmses.append(np.sqrt(mean_squared_error(y_val, y_pred)))

        # Secondary KPI: MAPE
        fold_mapes.append(mean_absolute_percentage_error(y_val, y_pred))

    # Note: CPWS MAE skipped — raw 'ws' was dropped during feature selection and
    # is not available in X_train. Add 'ws' to player_lookup_train.csv to re-enable.
    model_cv_summary.append({
        "Model Architecture":  name,
        "Primary KPI: RMSE":   np.mean(fold_rmses),
        "Secondary KPI: MAPE": f"{np.mean(fold_mapes) * 100:.2f}%",
    })
    print(f"{name}: RMSE=${np.mean(fold_rmses):,.0f}")

results_df = pd.DataFrame(model_cv_summary)


print("CV Experiment Performance")
print(results_df.to_string(index=False))




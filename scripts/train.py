

from pathlib import Path
import re

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
from sklearn.linear_model import LassoCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import VarianceThreshold

import warnings

import joblib
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.model_selection import GridSearchCV
import json
import platform
import sklearn
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error, r2_score





# Same data pre-processing as the learnability and KPI analysis

# Load and integrate data
repo = Path.cwd().parent
data_dir = repo / "data" / "raw"

# Load raw CSV components
adv = pd.read_csv(data_dir / "2025_advanced.csv")
per = pd.read_csv(data_dir / "2025_per_game.csv")
tot = pd.read_csv(data_dir / "2025_totals.csv")
sal = pd.read_csv(data_dir / "salary_2025.csv")
teamadv = pd.read_csv(data_dir / "2025_advanced-team.csv")
stand = pd.read_csv(data_dir / "2025_wnba_standings.csv")


# Standardize column names into a code-friendly format
def clean_col(name):
    text = str(name).strip().lower()
    text = text.replace("2025 salary", "salary")
    text = text.replace("2025 signing", "signing")
    text = text.replace("%", "pct")
    text = re.sub(r"[^0-9a-z]+", "_", text)
    return text.strip("_")


# Apply column-name standardizations and strip whitespace from text features
def clean_df(df):
    df.columns = [clean_col(col) for col in df.columns]
    for col in df.select_dtypes(include=["object", "string"]).columns:
        df[col] = df[col].astype(str).str.strip().replace({"": np.nan, "—": np.nan, "nan": np.nan})
    return df


adv, per, tot = clean_df(adv), clean_df(per), clean_df(tot)
sal, teamadv, stand = clean_df(sal), clean_df(teamadv), clean_df(stand)

# Target preparation
sal["salary"] = pd.to_numeric(sal["salary"], errors="coerce")

# Map full team names to uniform abbreviations
teammap = {
    "Atlanta Dream": "ATL", "Chicago Sky": "CHI", "Connecticut Sun": "CON",
    "Dallas Wings": "DAL", "Golden State Valkyries": "GSV", "Indiana Fever": "IND",
    "Las Vegas Aces": "LVA", "Los Angeles Sparks": "LAS", "Minnesota Lynx": "MIN",
    "New York Liberty": "NYL", "Phoenix Mercury": "PHO", "Seattle Storm": "SEA",
    "Washington Mystics": "WAS"
}

teamadv["team"] = teamadv["team"].str.replace("*", "", regex=False).str.strip().map(teammap)
stand["team_name"] = stand["team_name"].str.replace("*", "", regex=False).str.strip().map(teammap)

# Merge structural sub-tables
teamdf = teamadv.merge(stand, left_on="team", right_on="team_name", how="left", suffixes=("_adv", "_stand"))
playerdf = per.merge(adv, on=["player", "team", "pos", "g", "mp"], how="outer")
playerdf = playerdf.merge(tot, on=["player", "team", "pos", "g", "mp", "gs"], how="outer")

# Correct name spelling/encoding anomalies to maximize merge coverage
name_fix = {
    "Anastasiia Kosu": "Anastasiia Olairi Kosu", "Janelle SalaÃ¼n": "Janelle Salaun",
    "LeÃ¯la Lacan": "Leila Lacan", "Luisa GeiselsÃ¶der": "Luisa Geiselsoder",
    "Mamignan TourÃ©": "Mamignan Touré", "MariÃ¨me Badiane": "Marième Badiane",
    "Te-Hina PaoPao": "Te-Hina Paopao", "Sika KonÃ©": "Sika Kone"
}
playerdf["player"] = playerdf["player"].replace(name_fix)

# Construct final modeling table
final_df = sal.merge(playerdf, on="player", how="inner", suffixes=("_sal", ""))
final_df = final_df.merge(teamdf, on="team", how="left")


final_df.columns.tolist()



#performance-only features
performance_cols = ['g', 'gs', 'mp', 'fg', 'fga', 'fg3', 'fg3a',
                    'ft', 'fta', 'orb', 'drb', 'trb', 'ast', 'stl', 'blk',
                    'tov', 'pf', 'pts', 'per', 'drb_pct', 'trb_pct', 'ast_pct', 
                    'stl_pct', 'blk_pct', 'usg_pct', 'ows', 'dws', 'ws', 
]

#team features
team_features = ['wins_stand', 'losses_stand']

X = final_df[performance_cols + team_features]



y = final_df['salary']
X = final_df.select_dtypes(include=[np.number])
X = X.drop(columns=['salary'])


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state = 42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)



feature_selection_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')), 
    ('scaler', StandardScaler()),                   
    ('selector', SelectKBest(mutual_info_regression, k=20))
])



X_train_selected = feature_selection_pipeline.fit_transform(X_train, y_train)

X_test_selected = feature_selection_pipeline.transform(X_test)

print(f"Original training shape: {X_train.shape}")
print(f"After selection: {X_train_selected.shape}")



selected_features = feature_selection_pipeline.get_feature_names_out(input_features=X.columns)

print(f"Selected {len(selected_features)} features:")
print(selected_features)



#Kbest selected features
X_test_selectedkbest = feature_selection_pipeline.transform(X_test)



# Based on SelectKBest, the following features are the top 20 deemed the most relevant features in the dataset: 'g_sal', 'g', 'mp', 'mp_per_g', 'fg_per_g', 'fga_per_g', 'fg2_per_g'
#  'pts_per_g', 'dws', 'fg', 'fga', 'fg2', 'fg2a', 'ft', 'fta', 'stl', 'blk', 'tov', 'pf', 'pts'


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state = 42)


lasso_pipeline = Pipeline([('imputer', SimpleImputer(strategy='median')), 
                          ('scaler', StandardScaler()),
                          ('lasso', LassoCV(cv=5, random_state=42, max_iter=10000))
                          ])

lasso_pipeline.fit(X_train, y_train)

lasso_model = lasso_pipeline.named_steps['lasso']



constant_filter = VarianceThreshold(threshold=0)
constant_filter.fit(X_train)
non_concstant_mask = constant_filter.get_support()
feature_names_after_filter = X_train.columns[non_concstant_mask].tolist()



coeffs = lasso_model.coef_
selected_features_lasso = [feature_names_after_filter[i] for i in range(len(coeffs)) if coeffs[i]!=0]

print(f"Original features: {len(X_train.columns)}")
print(f"Non-constant features: {len(feature_names_after_filter)}")
print(f"Lasso selected {len(selected_features_lasso)}")
print(f"Selected {selected_features_lasso[:10]}")



X_train_lasso_selected = X_train[selected_features_lasso]
X_test__lassoselected = X_test[selected_features_lasso]


# Based on the Lasso model, the following features are the most useful in determining 'salary': 'g_sal', 'gs_sal', 'g', 'gs', 'blk_per_g', 'ast_pct', 'dws', 'fg3a', 'blk', 'pts'
# Some of these overlap with the features from SelectedKBest, which could be a great sign for model predictability. 


rf = RandomForestRegressor(n_estimators=100, random_state=43)
rf.fit(X_train, y_train)
importances = pd.Series(rf.feature_importances_, index=X.columns)
top_features = importances.nlargest(20).index.tolist()

print(f"Top RF features: {top_features[:10]}")



X_train_rf_selected = X_train[top_features]
X_test_rf_selected = X_test[top_features]


# Using RandomForest, the following features are deemed important: 'mp', 'pts', 'ft', 'fg', 'fta', 'fg_per_g', 'g_sal', 'fga', 'g', 'gs_sal'



# # Feature Engineering
# 
# The main issue from EDA and KPI analysis is that salary is not only based on production. Rookie contracts, hardship contracts, and veteran contracts follow different salary rules.
# 
# Instead of building separate rookie, veteran, and hardship models, this notebook uses one model with contract-aware features. Separate models could show each contract group's salary structure more directly, but the current dataset is small, so splitting the data may make the models unstable.




# Same data pre-processing as the learnability and KPI analysis

# Load and integrate data
repo = Path.cwd().parent
data_dir = repo / "data" / "raw"

# Load raw CSV components
adv = pd.read_csv(data_dir / "2025_advanced.csv")
per = pd.read_csv(data_dir / "2025_per_game.csv")
tot = pd.read_csv(data_dir / "2025_totals.csv")
sal = pd.read_csv(data_dir / "salary_2025.csv")
teamadv = pd.read_csv(data_dir / "2025_advanced-team.csv")
stand = pd.read_csv(data_dir / "2025_wnba_standings.csv")

# Standardize column names into a code-friendly format
def clean_col(name):
    text = str(name).strip().lower()
    text = text.replace("2025 salary", "salary")
    text = text.replace("2025 signing", "signing")
    text = text.replace("%", "pct")
    text = re.sub(r"[^0-9a-z]+", "_", text)
    return text.strip("_")

# Apply column-name standardizations and strip whitespace from text features
def clean_df(df):
    df.columns = [clean_col(col) for col in df.columns]
    for col in df.select_dtypes(include=["object", "string"]).columns:
        df[col] = df[col].astype(str).str.strip().replace({"": np.nan, "—": np.nan, "nan": np.nan})
    return df

adv, per, tot = clean_df(adv), clean_df(per), clean_df(tot)
sal, teamadv, stand = clean_df(sal), clean_df(teamadv), clean_df(stand)

# Target preparation
sal["salary"] = pd.to_numeric(sal["salary"], errors="coerce")

# Map full team names to uniform abbreviations
teammap = {
    "Atlanta Dream": "ATL", "Chicago Sky": "CHI", "Connecticut Sun": "CON",
    "Dallas Wings": "DAL", "Golden State Valkyries": "GSV", "Indiana Fever": "IND",
    "Las Vegas Aces": "LVA", "Los Angeles Sparks": "LAS", "Minnesota Lynx": "MIN",
    "New York Liberty": "NYL", "Phoenix Mercury": "PHO", "Seattle Storm": "SEA",
    "Washington Mystics": "WAS"
}

teamadv["team"] = teamadv["team"].str.replace("*", "", regex=False).str.strip().map(teammap)
stand["team_name"] = stand["team_name"].str.replace("*", "", regex=False).str.strip().map(teammap)

# Merge structural sub-tables
teamdf = teamadv.merge(stand, left_on="team", right_on="team_name", how="left", suffixes=("_adv", "_stand"))
playerdf = per.merge(adv, on=["player", "team", "pos", "g", "mp"], how="outer")
playerdf = playerdf.merge(tot, on=["player", "team", "pos", "g", "mp", "gs"], how="outer")

# Correct name spelling/encoding anomalies to maximize merge coverage
name_fix = {
    "Anastasiia Kosu": "Anastasiia Olairi Kosu", "Janelle SalaÃ¼n": "Janelle Salaun",
    "LeÃ¯la Lacan": "Leila Lacan", "Luisa GeiselsÃ¶der": "Luisa Geiselsoder",
    "Mamignan TourÃ©": "Mamignan Touré", "MariÃ¨me Badiane": "Marième Badiane",
    "Te-Hina PaoPao": "Te-Hina Paopao", "Sika KonÃ©": "Sika Kone"
}
playerdf["player"] = playerdf["player"].replace(name_fix)

# Construct final modeling table
final_df = sal.merge(playerdf, on="player", how="inner", suffixes=("_sal", ""))
final_df = final_df.merge(teamdf, on="team", how="left")


# ## Feature Ideas
# 
# The new features are grouped into four simple categories.
# 
# | Group | Features | Why |
# | --- | --- | --- |
# | Contract | `group`, `rookie_flag`, `hardship_flag`, `vet_flag`, `unknown_flag` | Salary rules differ by contract type. |
# | Role | `avail_rate`, `start_rate`, `team_min`, `starter`, `rotation` | Salary may reflect role and availability. |
# | Efficiency | `pts40`, `ast40`, `reb40`, `stocks`, `stocks40`, `pps`, `ft_rate`, `three_rate`, `ast_tov`, `ws_game`, `ws40` | Makes production easier to compare across minutes. |
# | Team Context | `ts_diff`, `efg_diff`, `team_power`| Compares player efficiency with team efficiency and team strength. |
# | Interaction | `ws_rookie`, `ws_vet`, `ws_hardship`, `pts_rookie`, `pts_vet`, `pts_hardship` | Allows production to matter differently for rookies and veterans. |

# ### Contract Features
# 
# | Feature | Formula | Rationale |
# | --- | --- | --- |
# | group | Grouped from signing | Simplifies raw signing status into broader contract groups because rookie, hardship, and veteran contracts follow different salary rules. |
# | rookie_flag | 1 if group == "rookie", else 0 | Identifies rookie-contract players, whose salaries may be lower because of rookie-scale rules. |
# | hardship_flag | 1 if group == "hardship", else 0 | Identifies hardship or temporary-contract players, whose salaries may be prorated or not reflect full-season value. |
# | vet_flag | 1 if group == "veteran", else 0 | Identifies veteran-market players, whose salaries are more likely to reflect market negotiation and production. |
# | unknown_flag | 1 if group == "unknown", else 0 | Marks players with missing signing status so missing contract information is not ignored. |


# Convert raw signing status into broader contract groups
def get_group(value):
    if pd.isna(value):
        return "unknown"
    value = str(value).strip().lower()
    if "rookie" in value:
        return "rookie"
    if "hardship" in value or "susp" in value:
        return "hardship"
    if value in ["ufa", "rfa", "core"]:
        return "veteran"
    if value in ["udfa", "reserved"]:
        return "controlled"
    return "other"

# Create binary flags so models can use contract status numerically
final_df["group"] = final_df["signing"].apply(get_group)
final_df["rookie_flag"] = (final_df["group"] == "rookie").astype(int)
final_df["hardship_flag"] = (final_df["group"] == "hardship").astype(int)
final_df["vet_flag"] = (final_df["group"] == "veteran").astype(int)
final_df["unknown_flag"] = (final_df["group"] == "unknown").astype(int)

final_df[["player", "salary", "signing", "group", "rookie_flag", "hardship_flag", "vet_flag"]].head()


# ### Role Features
# 
# | Feature | Formula | Rationale |
# | --- | --- | --- |
# | `avail_rate` | `g / 44` | Measures how much of the full 2025 regular season the player appeared in. |
# | `start_rate` | `gs / g` if `g > 0`, else 0 | Measures how often the player started when she appeared in games. |
# | `team_min` | `mp / 44` | Measures average minutes per team game, including games the player missed. |
# | `starter` | 1 if `start_rate >= 0.5`, else 0 | Identifies players who started at least half of the games they appeared in. |
# | `rotation` | 1 if mp_per_g >= 15, else 0 | Identifies regular rotation players based on playing at least 15 minutes per game. |


season_games = 44

# Create role and availability features from games, starts, and minutes
final_df["avail_rate"] = (final_df["g"] / season_games) 
final_df["start_rate"] = np.where(final_df["g"] > 0, final_df["gs"] / final_df["g"], 0) 
final_df["team_min"] = final_df["mp"] / season_games 

# Starter if she started at least half of her games
final_df["starter"] = (final_df["start_rate"] >= 0.5).astype(int)
# Rotation player if she played at least 15 minutes per game
final_df["rotation"] = (final_df["mp_per_g"] >= 15).astype(int)

role_cols = ["avail_rate", "start_rate", "team_min", "starter", "rotation"]
final_df[["player", "team", "g", "gs", "mp_per_g", *role_cols]].head()


# ### Efficiency Features
# 
# | Feature | Formula | Rationale |
# |----------|----------|----------|
# | `pts40` | `pts_per_g * (40 / mp_per_g)` | Standardizes scoring production to a 40-minute rate so players with different playing time can be compared more fairly. |
# | `ast40` | `ast_per_g * (40 / mp_per_g)` | Standardizes assist production to a 40-minute rate. |
# | `reb40` | `trb_per_g * (40 / mp_per_g)` | Standardizes rebounding production to a 40-minute rate. |
# | `stocks` | `stl_per_g + blk_per_g` | Combines steals and blocks into one simple defensive activity measure. |
# | `stocks40` | `stocks * (40 / mp_per_g)` | Standardizes steals plus blocks to a 40-minute rate. |
# | `pps` | `pts_per_g / (fga_per_g + 0.44 * fta_per_g)` | Measures scoring efficiency by comparing points to estimated shooting possessions. |
# | `ft_rate` | `fta_per_g / fga_per_g` | Measures how often a player gets to the free throw line relative to field goal attempts. |
# | `three_rate` | `fg3a_per_g / fga_per_g` | Measures how much of a player's shot profile comes from three-point attempts. |
# | `ast_tov` | `ast_per_g / tov_per_g` | Measures playmaking efficiency by comparing assists to turnovers. |
# | `ws_game` | `ws / g` | Normalizes Win Shares by games played. |
# | `ws40` | `(ws / mp) * 40` | Normalizes Win Shares to a 40-minute rate. |


# Division helper to avoid dividing by zero
def div(a, b):
    return np.where(pd.Series(b).astype(float) != 0, a / b, np.nan)

# Multiplier for converting per-game stats to per-40-minute rates
per40 = div(40, final_df["mp_per_g"])

# Per-40 production features to compare players with different playing time
final_df["pts40"] = final_df["pts_per_g"] * per40
final_df["ast40"] = final_df["ast_per_g"] * per40
final_df["reb40"] = final_df["trb_per_g"] * per40

# Combine steals and blocks as a simple defensive activity measure
final_df["stocks"] = final_df["stl_per_g"] + final_df["blk_per_g"]
final_df["stocks40"] = final_df["stocks"] * per40

# Scoring efficiency and shot profile features
final_df["pps"] = div(final_df["pts_per_g"], final_df["fga_per_g"] + 0.44 * final_df["fta_per_g"])
final_df["ft_rate"] = div(final_df["fta_per_g"], final_df["fga_per_g"])
final_df["three_rate"] = div(final_df["fg3a_per_g"], final_df["fga_per_g"])

# Playmaking efficiency for assists compared with turnovers
final_df["ast_tov"] = div(final_df["ast_per_g"], final_df["tov_per_g"])

# Normalize win shares by games and 40 minutes
final_df["ws_game"] = div(final_df["ws"], final_df["g"])
final_df["ws40"] = div(final_df["ws"], final_df["mp"]) * 40

eff_cols = ["pts40", "ast40", "reb40", "stocks", "stocks40", "pps", "ft_rate", "three_rate", "ast_tov", "ws_game", "ws40"]
final_df[["player", "mp_per_g", "pts_per_g", "ws", *eff_cols]].head()


# ### Team Context Features
# 
# These features compare each player's shooting efficiency to her team's overall efficiency, so the model can see whether a player performed above or below her team context.
# 
# | Feature | Formula | Rationale |
# |----------|----------|----------|
# | `ts_diff` | `ts_pct_x - ts_pct_y` | Compares a player's True Shooting % with her team's True Shooting %. Positive values indicate the player was more efficient than the team average. |
# | `efg_diff` | `efg_pct_x - efg_pct_y` | Compares a player's Effective Field Goal % with her team's Effective Field Goal %. Positive values indicate the player shot more efficiently than the team average. |
# | `team_power` | `mean(z(win_loss_pct), z(net_rtg), z(srs))` | Creates a simple team strength index using available team-level metrics. |


# _x columns = player-level stats & _y columns = team-level stats
# Compare player shooting efficiency with team shooting efficiency
if "ts_pct_x" in final_df.columns and "ts_pct_y" in final_df.columns:
    final_df["ts_diff"] = final_df["ts_pct_x"] - final_df["ts_pct_y"]
else:
    final_df["ts_diff"] = np.nan

if "efg_pct_x" in final_df.columns and "efg_pct_y" in final_df.columns:
    final_df["efg_diff"] = final_df["efg_pct_x"] - final_df["efg_pct_y"]
else:
    final_df["efg_diff"] = np.nan

# Use available team-level metrics to create a simple team strength index
strength_cols = [col for col in ["win_loss_pct", "net_rtg", "srs"] if col in final_df.columns]
if strength_cols:
    temp = final_df[strength_cols].apply(pd.to_numeric, errors="coerce")
    temp = temp.fillna(temp.median())
    # Standardize columns because they use different scales
    temp = (temp - temp.mean()) / temp.std(ddof=0).replace(0, 1)
    final_df["team_power"] = temp.mean(axis=1)
else:
    final_df["team_power"] = np.nan

team_cols = ["ts_diff", "efg_diff", "team_power"]
final_df[["player", "team", *team_cols]].head()


# ### Interaction Features
# 
# These features are a direct response to the contract-type issue. The same production may relate to salary differently depending on contract type. For example, a rookie and a veteran can have similar `ws` or `pts_per_g`, but their salaries may follow different rules.
# 
# | Feature | Formula | Rationale |
# |----------|----------|----------|
# | `ws_rookie` | `ws * rookie_flag` | Captures how Win Shares relate to salary specifically for rookie players. |
# | `ws_vet` | `ws * vet_flag` | Captures how Win Shares relate to salary specifically for veteran players. |
# | `ws_hardship` | `ws * hardship_flag` | Captures how Win Shares relate to salary specifically for hardship players. |
# | `pts_rookie` | `pts_per_g * rookie_flag` | Captures how points per game relate to salary specifically for rookie players. |
# | `pts_vet` | `pts_per_g * vet_flag` | Captures how points per game relate to salary specifically for veteran players. |
# | `pts_hardship` | `pts_per_g * hardship_flag` | Captures how points per game relate to salary specifically for hardship players. |
# 
# Interaction features allow one model to learn these differences without building separate models for each contract group.
# 
# For example, in a linear regression model, the relationship can be interpreted like this:
# 
# ```python
# salary = base
#         + b1 * ws
#         + b2 * rookie_flag
#         + b3 * vet_flag
#         + b4 * ws_rookie
#         + b5 * ws_vet


# Win Shares interaction for each players
final_df["ws_rookie"] = final_df["ws"] * final_df["rookie_flag"]
final_df["ws_vet"] = final_df["ws"] * final_df["vet_flag"]
final_df["ws_hardship"] = final_df["ws"] * final_df["hardship_flag"]

# Points per game interaction for each players
final_df["pts_rookie"] = final_df["pts_per_g"] * final_df["rookie_flag"]
final_df["pts_vet"] = final_df["pts_per_g"] * final_df["vet_flag"]
final_df["pts_hardship"] = final_df["pts_per_g"] * final_df["hardship_flag"]

int_cols = ["ws_rookie", "ws_vet", "ws_hardship", "pts_rookie", "pts_vet", "pts_hardship"]
final_df[["player", "group", "ws", "pts_per_g", *int_cols]].head()


# ### PCA Features
# 
# PCA is a method that compresses several correlated numeric variables into a few summary variables. In this project, many basketball production stats overlap with each other. For example, players with high minutes often also have higher points, field goal attempts, rebounds, assists, and Win Shares. So even though there are many columns, some of them repeat similar information about overall player production.
# 
# In this notebook, PCA takes production-related variables such as `mp`, `pts`, `fg`, `fga`, `ft`, `trb`, `ast`, `ws`, `pts40`, and `ast40`, then summarizes them into two new features:
# 
# | Feature | Formula | Rationale |
# |----------|----------|----------|
# | `pca1` | First PCA component from selected production statistics | Summarizes the largest overall pattern in player production. |
# | `pca2` | Second PCA component from selected production statistics | Summarizes the second-largest production pattern that is independent of `pca1`. |
# 
# These two features are production summary scores. They are useful for model testing and for summarizing a player's overall production profile.
# 
# One limitation is that PCA features are harder to interpret than domain-driven features. For example, `pts40` clearly means points per 40 minutes, but `pca1` is a
# mixture of many production stats. Because of this, I treat PCA features as supporting features rather than the main explanation for salary.
# 
# PCA cannot handle missing values directly, so the small number of missing values in the PCA input columns were filled with the median before running PCA.
# 
# The PCA variance explained output shows how much information the first two components capture. 

# In[ ]:


# Select production-related numeric features to summarize with PCA
pca_cols = [
    "mp", "g", "gs", "pts", "fg", "fga", "fg3", "fg3a", "ft", "fta",
    "trb", "ast", "stl", "blk", "tov", "per", "usg_pct", "ows", "dws", "ws",
    "pts40", "ast40", "reb40", "stocks40", "pps", "ast_tov"
]
# Keep only columns that actually exist in final_df
pca_cols = [col for col in pca_cols if col in final_df.columns]

# Convert to numeric and fill missing values before PCA
x = final_df[pca_cols].apply(pd.to_numeric, errors="coerce")
x = x.fillna(x.median())

# Standardize features so large-scale stats like minutes do not dominate
x = (x - x.mean()) / x.std(ddof=0).replace(0, 1)

# Run PCA using SVD and keep the first two components
u, s, vt = np.linalg.svd(x.to_numpy(), full_matrices=False)
pcs = x.to_numpy() @ vt[:2].T

# Add PCA scores as summary production features
final_df["pca1"] = pcs[:, 0]
final_df["pca2"] = pcs[:, 1]

# Check how much variation the first two PCA components explain
var = (s ** 2) / max(len(x) - 1, 1)
var_ratio = var / var.sum()
print("PCA variance explained:", var_ratio[:2])

final_df[["player", "pca1", "pca2"]].head()


# In this result, `pca1` explains about 57.2% of the differences across the selected production stats, and `pca2` explains about 10.4%. Together, they summarize about 67.6% of the production variation.

# ### Final Feature List


# Group engineered features by their main purpose
feature_groups = {
    "contract type": [
        "group", "rookie_flag", "hardship_flag", "vet_flag", "unknown_flag"
    ],
    "role / availability": [
        "avail_rate", "start_rate", "team_min", "starter", "rotation"
    ],
    "efficiency / salary prediction": [
        "pts40", "ast40", "reb40", "stocks", "stocks40", "pps",
        "ft_rate", "three_rate", "ast_tov", "ws_game", "ws40"
    ],
    "team context": [
        "ts_diff", "efg_diff", "team_power"
    ],
    "contract interaction": [
        "ws_rookie", "ws_vet", "ws_hardship",
        "pts_rookie", "pts_vet", "pts_hardship"
    ],
    "pca summary": [
        "pca1", "pca2"
    ]
}

# Convert the grouped feature dictionary into a summary table
rows = []
for category, features in feature_groups.items():
    for feature in features:
        rows.append({
            "feature": feature,
            "category": category
        })

feature_notes = pd.DataFrame(rows)
feature_notes


# # Model Tuning



repo = Path.cwd()
if repo.name == "notebooks":
    repo = repo.parent

processed_dir = repo / "data" / "processed"
result_dir = repo / "results"
art_dir = repo / "artifacts"

art_dir.mkdir(exist_ok=True)

x_train = pd.read_csv(processed_dir / "X_train_processed.csv")
y_train = pd.read_csv(processed_dir / "y_train.csv")["salary"]
lookup_train = pd.read_csv(processed_dir / "player_lookup_train.csv")

print(x_train.shape, y_train.shape)
print(lookup_train.groupby("year").size())



# Prepare chronological CV folds for GridSearchCV
cv_folds = []
years = sorted(lookup_train["year"].unique())

# Each fold trains on past seasons and validates on the next season
for val_year in years[1:]:
    train_years = [year for year in years if year < val_year]
    train_idx = lookup_train.index[lookup_train["year"].isin(train_years)].to_numpy()
    val_idx = lookup_train.index[lookup_train["year"] == val_year].to_numpy()
    cv_folds.append((train_idx, val_idx))

for i, (train_idx, val_idx) in enumerate(cv_folds, start=1):
    train_years = [int(year) for year in sorted(lookup_train.loc[train_idx, "year"].unique())]
    val_years = [int(year) for year in sorted(lookup_train.loc[val_idx, "year"].unique())]

    train_label = ",".join(map(str, train_years))
    val_label = "-".join(map(str, val_years))

    print(f"fold {i}: train {train_label} ({len(train_idx)} rows) -> val {val_label} ({len(val_idx)} rows)")


# ### Hyperparameter Search Design
# 
# `modeling_experiments.ipynb` showed that the Stacking Ensemble and ElasticNet had strong CV RMSE values, but the Stacking Ensemble is less interpretable and more complex because it combines several models, which can be risky with a small dataset. ElasticNet performed well, but it is part of the regularized linear model family already represented by Ridge. For this optimization step, we focused on RandomForest and Gradient Boosting because they are defensible tree-based ensemble models that can capture nonlinear relationships while keeping the tuning workflow simple and reproducible.
# 
# 
# 1. The search space is centered around the baseline model settings.
# 2. Because the dataset is relatively small, we avoid an overly large grid that could overfit the CV folds.
# 3. We tune only the main complexity-control parameters for Random Forest and Gradient Boosting.
# 4. We limit tuning to two model families to avoid a “soup model” approach and keep model selection defensible.


# RandomForestRegressor hyperparameter
RandomForest_grid = {
    "n_estimators": [200, 400],
    "max_depth": [None, 6, 10],
    "min_samples_leaf": [3, 5, 8],
}

# GradientBoostingRegressor hyperparameter
GradientBoosting_grid = {
    "n_estimators": [100, 200],
    "learning_rate": [0.03, 0.05, 0.1],
    "max_depth": [2, 3],
    "min_samples_leaf": [3, 5],
}

searches = {
    "RandomForest": (
        RandomForestRegressor(random_state=26),
        RandomForest_grid,
        art_dir / "best_RandomForest.joblib",
    ),
    "GradientBoosting": (
        GradientBoostingRegressor(random_state=26),
        GradientBoosting_grid,
        art_dir / "best_GradientBoosting.joblib",
    ),
}


# Initialize containers for grid-search results and best model tracking
rows = []
best_rows = []
best_score = np.inf
best_name = None
best_model = None

# Run GridSearchCV for each model family defined in searches
for name, (model, grid, path) in searches.items():
    search = GridSearchCV(
        estimator=model,
        param_grid=grid,
        scoring="neg_root_mean_squared_error",  # RMSE, negative for sklearn scoring
        cv=cv_folds,
        refit=True,
        n_jobs=-1,
        return_train_score=True,
    )
    search.fit(x_train, y_train)

    # Convert the full GridSearchCV output into a cleaner results table
    result = pd.DataFrame(search.cv_results_)
    keep_cols = [
        "params",
        "mean_test_score",  # Mean validation score across folds
        "std_test_score",   # Fold-to-fold validation score variation
        "rank_test_score",  # Rank of each parameter setting
        "mean_train_score", # Mean training score across folds
    ]
    result = result[keep_cols]
    result["model"] = name
    result["cv_rmse"] = -result["mean_test_score"]
    result["train_rmse"] = -result["mean_train_score"]
    result = result.drop(columns=["mean_test_score", "mean_train_score"])
    rows.append(result)

    # Store the best CV result for this model family
    cv_rmse = -search.best_score_
    best_row = {
        "model": name,
        "cv_rmse": cv_rmse,
        "params": search.best_params_,
        "artifact": str(path.relative_to(repo)),
    }
    best_rows.append(best_row)

    # Save this model family's best estimator as a reusable artifact
    joblib.dump(search.best_estimator_, path)

    if cv_rmse < best_score:
        best_score = cv_rmse
        best_name = name
        best_model = search.best_estimator_

all_rows = pd.concat(rows, ignore_index=True)
best_df = pd.DataFrame(best_rows).sort_values("cv_rmse")

all_rows.to_csv(result_dir / "tuning_results.csv", index=False)
joblib.dump(best_model, art_dir / "best_model.joblib")

print("best model:", best_name)
display(best_df)


# # Final Serialized Pipeline / Model
# Inputs:
# - `data/processed/X_train_processed.csv`
# - `data/processed/y_train.csv`
# - `data/processed/X_test_2025_processed.csv`
# - `data/processed/y_test_2025.csv`
# - `data/processed/player_lookup_train.csv`
# - `data/processed/player_lookup_test_2025.csv`
# 
# Outputs:
# - `artifacts/final_model.joblib`
# - `results/final/final_metrics.csv`
# - `results/final/final_predictions.csv`
# - `results/final/final_model_metadata.json`


pd.set_option("display.max_columns", None)
pd.set_option("display.max_colwidth", 120)

ROOT = Path.cwd()
if not (ROOT / "data").exists():
    ROOT = ROOT.parent

processed_dir = ROOT / "data" / "processed"
artifact_dir = ROOT / "artifacts"
final_result_dir = ROOT / "results" / "final"

RANDOM_STATE = 26
TRAIN_YEARS = [2021, 2022, 2023, 2024]
TEST_YEAR = 2025
RESIDUAL_DEFINITION = "predicted - actual"


# ### Load Processed Train/Test Data

X_train = pd.read_csv(processed_dir / "X_train_processed.csv")
y_train = pd.read_csv(processed_dir / "y_train.csv")["salary"]
X_test = pd.read_csv(processed_dir / "X_test_2025_processed.csv")
y_test = pd.read_csv(processed_dir / "y_test_2025.csv")["salary"]
lookup_train = pd.read_csv(processed_dir / "player_lookup_train.csv")
lookup_test = pd.read_csv(processed_dir / "player_lookup_test_2025.csv")

feature_names = X_train.columns.tolist()

if X_train.shape[1] != X_test.shape[1]:
    raise ValueError(f"Train/test feature count mismatch: {X_train.shape[1]} vs {X_test.shape[1]}")

if X_train.columns.tolist() != X_test.columns.tolist():
    raise ValueError("Train/test feature columns are not aligned in the same order.")

if len(X_train) != len(y_train):
    raise ValueError("X_train and y_train row counts do not match.")

if len(X_test) != len(y_test) or len(X_test) != len(lookup_test):
    raise ValueError("X_test, y_test, and lookup_test row counts do not match.")

print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("Number of final features:", len(feature_names))
display(pd.DataFrame({"feature": feature_names}).head(30))


# ### Train Final Model
# 
# The final model uses the tuned Random Forest configuration selected during optimization. This setup balances predictive performance with basic overfitting control.
# 
# The model uses 400 trees for stable ensemble predictions, `min_samples_leaf=5` to avoid overly specific leaf nodes, no fixed tree-depth limit because it performed best in tuning, and a fixed random seed (`random_state=26`) for reproducibility.


# Final tuned Random Forest selected by CV RMSE during optimization
final_model = RandomForestRegressor(
    n_estimators=400,
    min_samples_leaf=5,
    max_depth=None,
    random_state=RANDOM_STATE,
)

final_model.fit(X_train, y_train)

print("Final model:", final_model)
print("Model parameters:")

# Show all model hyperparameters for auditability, including sklearn defaults
display(pd.Series(final_model.get_params()))


# ### Holdout Prediction and Evaluation
# 
# This section applies the final trained model to the untouched 2025 holdout set. It creates player-level predictions with actual salary, predicted salary, residual, and absolute error, then computes the final KPI metrics.
# 
# Residuals are defined as `predicted - actual`. Positive residuals indicate the model predicted a higher salary than the player actually earned, while negative residuals indicate the model predicted a lower salary than the player actually earned.
# 
# The largest absolute errors are displayed as a quick diagnostic to identify where the model missed most.


# Generate holdout salary predictions for the untouched 2025 test set
pred = final_model.predict(X_test)

# Build a player-level prediction table with identifiers, actual salary, predicted salary, and error columns
pred_df = lookup_test.copy()
pred_df["actual"] = y_test.to_numpy()
pred_df["predicted"] = pred
pred_df["residual"] = pred_df["predicted"] - pred_df["actual"]
pred_df["abs_error"] = pred_df["residual"].abs()

# Keep the final prediction output columns in a clear reporting order
pred_cols = ["player", "team", "year", "group", "actual", "predicted", "residual", "abs_error"]
pred_df = pred_df[pred_cols]

# Compute final holdout KPIs and store split/model context for traceability
metrics = {
    "model": "RandomForestRegressor",
    "train_years": "2021-2024",
    "test_year": TEST_YEAR,
    "n_train": int(len(X_train)),
    "n_test": int(len(X_test)),
    "n_features": int(len(feature_names)),
    "RMSE": float(np.sqrt(mean_squared_error(y_test, pred))),
    "MAE": float(mean_absolute_error(y_test, pred)),
    "MAPE": float(mean_absolute_percentage_error(y_test, pred)),
    "R2": float(r2_score(y_test, pred)),
    "residual_definition": RESIDUAL_DEFINITION,
}

# Convert metrics to a one-row table for display and CSV export
metrics_df = pd.DataFrame([metrics])

display(metrics_df)

# Show the largest player-level prediction errors as a quick model diagnostic
display(pred_df.sort_values("abs_error", ascending=False).head(10))


# ### Save Final Results


# Save the final KPI table for reporting and reproducibility
metrics_path = final_result_dir / "final_metrics.csv"
# Save player-level final predictions for error analysis and plots
predictions_path = final_result_dir / "final_predictions.csv"

metrics_df.to_csv(metrics_path, index=False)
pred_df.to_csv(predictions_path, index=False)


# ### Save Final Model Bundle
# 
# This section saves the final trained model as a reusable `joblib` bundle and saves a separate JSON metadata file for easy inspection.
# 
# The `final_model.joblib` file is used to reload the trained model in Python. It includes the model, feature names, model parameters, metrics, split information, target name, residual definition, and metadata.
# 
# The `final_model_metadata.json` file is a human-readable model summary. It records the model type, selected features, train/test split, final KPI results, input/output files, and package versions.


# Store the chronological train/test split details used to build the final model
split_info = {
    "train_years": TRAIN_YEARS,
    "test_year": TEST_YEAR,
    "n_train": int(len(X_train)),
    "n_test": int(len(X_test)),
}

# Build metadata for the JSON file and also include it inside the joblib bundle
metadata = {
    "artifact_name": "final_model.joblib",
    "model_type": type(final_model).__name__,
    "model_params": final_model.get_params(),
    "feature_names": feature_names,
    "target": "salary",
    "split_info": split_info,
    "metrics": metrics,
    "residual_definition": RESIDUAL_DEFINITION,
    "environment": {
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "pandas_version": pd.__version__,
        "numpy_version": np.__version__,
        "sklearn_version": sklearn.__version__,
        "joblib_version": joblib.__version__,
    },
}

# Joblib bundle stores the trained model and everything needed to reuse it safely
model_bundle = {
    "model": final_model,
    "feature_names": feature_names,
    "model_params": final_model.get_params(),
    "metrics": metrics,
    "split_info": split_info,
    "target": "salary",
    "residual_definition": RESIDUAL_DEFINITION,
    "metadata": metadata,
}

model_path = artifact_dir / "final_model.joblib"
metadata_path = final_result_dir / "final_model_metadata.json"

joblib.dump(model_bundle, model_path)

with open(metadata_path, "w", encoding="utf-8") as f:
    json.dump(metadata, f, indent=2)


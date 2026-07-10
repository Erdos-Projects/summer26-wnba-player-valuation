#!/usr/bin/env python
# coding: utf-8

# In[42]:


from pathlib import Path
import re

import numpy as np
import pandas as pd



# This notebook is for feature elimination decisions. 

# In[43]:


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


# In[44]:


final_df.columns.tolist()


# In[45]:


#performance-only features
performance_cols = ['g', 'gs', 'mp', 'fg', 'fga', 'fg3', 'fg3a',
                    'ft', 'fta', 'orb', 'drb', 'trb', 'ast', 'stl', 'blk',
                    'tov', 'pf', 'pts', 'per', 'drb_pct', 'trb_pct', 'ast_pct', 
                    'stl_pct', 'blk_pct', 'usg_pct', 'ows', 'dws', 'ws', 
]

#team features
team_features = ['wins_stand', 'losses_stand']

X = final_df[performance_cols + team_features]


# In[46]:


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
from sklearn.linear_model import LassoCV
from sklearn.ensemble import RandomForestRegressor


# In[47]:


y = final_df['salary']
X = final_df.select_dtypes(include=[np.number])
X = X.drop(columns=['salary'])


# In[48]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state = 42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# In[49]:


from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer


feature_selection_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')), 
    ('scaler', StandardScaler()),                   
    ('selector', SelectKBest(mutual_info_regression, k=20))
])


# In[50]:


X_train_selected = feature_selection_pipeline.fit_transform(X_train, y_train)

X_test_selected = feature_selection_pipeline.transform(X_test)

print(f"Original training shape: {X_train.shape}")
print(f"After selection: {X_train_selected.shape}")


# In[51]:


selected_features = feature_selection_pipeline.get_feature_names_out(input_features=X.columns)

print(f"Selected {len(selected_features)} features:")
print(selected_features)


# In[52]:


#Kbest selected features
X_test_selectedkbest = feature_selection_pipeline.transform(X_test)



# Based on SelectKBest, the following features are the top 20 deemed the most relevant features in the dataset: 'g_sal', 'g', 'mp', 'mp_per_g', 'fg_per_g', 'fga_per_g', 'fg2_per_g'
#  'pts_per_g', 'dws', 'fg', 'fga', 'fg2', 'fg2a', 'ft', 'fta', 'stl', 'blk', 'tov', 'pf', 'pts'

# In[53]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state = 42)


lasso_pipeline = Pipeline([('imputer', SimpleImputer(strategy='median')), 
                          ('scaler', StandardScaler()),
                          ('lasso', LassoCV(cv=5, random_state=42, max_iter=10000))
                          ])

lasso_pipeline.fit(X_train, y_train)

lasso_model = lasso_pipeline.named_steps['lasso']




# In[54]:


from sklearn.feature_selection import VarianceThreshold

constant_filter = VarianceThreshold(threshold=0)
constant_filter.fit(X_train)
non_concstant_mask = constant_filter.get_support()
feature_names_after_filter = X_train.columns[non_concstant_mask].tolist()


# In[55]:


coeffs = lasso_model.coef_
selected_features_lasso = [feature_names_after_filter[i] for i in range(len(coeffs)) if coeffs[i]!=0]

print(f"Original features: {len(X_train.columns)}")
print(f"Non-constant features: {len(feature_names_after_filter)}")
print(f"Lasso selected {len(selected_features_lasso)}")
print(f"Selected {selected_features_lasso[:10]}")


# In[56]:


X_train_lasso_selected = X_train[selected_features_lasso]
X_test__lassoselected = X_test[selected_features_lasso]


# Based on the Lasso model, the following features are the most useful in determining 'salary': 'g_sal', 'gs_sal', 'g', 'gs', 'blk_per_g', 'ast_pct', 'dws', 'fg3a', 'blk', 'pts'
# Some of these overlap with the features from SelectedKBest, which could be a great sign for model predictability. 

# In[57]:


rf = RandomForestRegressor(n_estimators=100, random_state=43)
rf.fit(X_train, y_train)
importances = pd.Series(rf.feature_importances_, index=X.columns)
top_features = importances.nlargest(20).index.tolist()

print(f"Top RF features: {top_features[:10]}")


# In[58]:


X_train_rf_selected = X_train[top_features]
X_test_rf_selected = X_test[top_features]


# Using RandomForest, the following features are deemed important: 'mp', 'pts', 'ft', 'fg', 'fta', 'fg_per_g', 'g_sal', 'fga', 'g', 'gs_sal'

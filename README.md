# WNBA Fair Markey Value Engine

Predicting what WNBA players should earn based on production. A 2026 Erdös Institute Project.

# Authors

Elena Axinn
Aklima Khanam
Nahyun Lee
Ling Le
Tiana Johnson

# Summary 
We analyze WNBA stats and salary data in an effort to develop a data-driven Fair Market Value engine that predicts what a player should earn based on their on-court production. We compare predicted contracts vs. actual contracts. 

# Background

The WNBA (Women's National Basketball Association) is one of the fastest-growing sports markets, with sponsorship deals surging over 40%, in-arena attendance at a 22-year high, and a 130% increase in young viewers all in the last year. This project is a Moneyball style efficiency analysis of WNBA player contracts vs. on-court production. We will attempt to predict what a WNBA player should earn based on performance and which teams get the most production per dollar. Datasets include multiple years of advanced statistics, WNBA salary history, play-type metrics, and salary tier structure from the recent Collective Bargaining Agreement. Methods could include regression modeling, feature engineering, residual analysis, and more.

# Datasets

Our datasets cover the 2021-2025 WNBA seasons with 250+ player records and 25 features. Most features are player production statistics (points, assists, usage). This also includes salaries, contract types and team context. The data was sourced from HerHoopStats and Basketball Reference. 

We used feature selection tools SelectKBest, Lasso, and Random Forest. All three methods suggested starting rate, points, and field goal attempts as the most salient features. The engineered features include availability rate, starting rate, and contract group indicators. 

# Stakeholders 

- WNBA Team General Managers (GMs)
- WNBA Players
- WNBA Coaches 

# Key Performance Indicators (KPIs)

* **Primary Predictive Metric:** Root Mean Squared Error (RMSE) to track overall dollar-value variance.
* **Secondary Fairness Metric:** Mean Absolute Percentage Error (MAPE) to evaluate compensation equity across different contract scales.
* **Tertiary Business ROI Metric:** Cost Per Win Share (CPWS) Mean Absolute Error to quantify the framework's utility for franchise-level macro analysis.


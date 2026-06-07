
## Project Goal
This project is a Moneyball style efficiency analysis of WNBA player contracts vs. on-court production (it would be interesting to look at off-court value like social media and sponsorship deals as well). We will attempt to predict what a WNBA player should earn based on performance and which teams get the most production per dollar. Datasets include multiple years of advanced statistics, WNBA salary history, play-type metrics, and salary tier structure from the recent Collective Bargaining Agreement. Methods could include regression modeling, feature engineering, residual analysis, and more.

Here, my understanding is that we have two goals:
1. Player-season level: we want to use previous-season performance data to predict a player's next-season salary. For example, we could use 2023-2024 performance data to predict 2024-2025 salary. This would better reflect the decision a team faces before signing a player. 
2. Team-season level: we also want to understand which team gets the most production per dollar. This can be studied using team-level salary efficiency metrics after aggregating player-level salary and performance. Off-court value, such as social media presence, jersey sales, and sponsorship visibility, could be considered as a later extension if those data become available.

## Initial Data Assessment Summary

Volume and coverage: The raw data folder contains six datasets. The salary file has 227 rows and 27 columns, while the three player-level Basketball Reference files each have 182 rows. This suggests that salary data and player performance data do not perfectly overlap, so player-name matching and merge coverage will be an important issue.I do not fully understand why we have mismatched players here, but my guess is team may kick out some players or sign new players during the currecnt season. The final regression dataset may contain fewer observations after merging salary data with player performance data. However, the current raw data only includes 2025, so the initial analysis is closer to an ex-post valuation: comparing 2025 salary with 2025 observed production. Thus, I do not think the current raw data is enough for us to build a true predictive player-valuation model.

Granularity: Our main regression task is player-season, meaning that each row should represent one player in one season. 
1. Player-season level: this matches the salary dataset and the three player-level performance datasets: advanced stats, per-game stats, and total stats. These files can potentially be merged by player name, and possibly team, to create the main modeling dataset.

2. Team-season level: the team-level datasets have a different granularity. Each row represents one team in the 2025 season, not one player. Therefore, these files are more appropriate for team-level efficiency analysis, such as production per dollar by team, rather than direct player-level salary prediction. If we want to use team-level information in the player-level model, we would need to merge team-level features onto each player by team.

Limitations: 
1. Although the project description mentions multiple years of data, the current raw folder only contains 2025 datasets. This limits the initial analysis to a single-season assessment unless additional years are collected later.

2. Salary and performance data have different coverage. The salary dataset has 227 rows, while the player-level performance datasets each have 182 rows. This means the final regression dataset may lose observations after merging. We need to check whether unmatched players form a systematic subgroup.

3. The current data is more suitable for ex-post valuation than true prediction. If we use 2025 performance data to explain 2025 salary, we are comparing salary with observed production after the fact. A true predictive model would require previous-season performance data to predict next-season salary.

4. Salary may not be determined only by current on-court performance. The salary file includes a `2025 Signing` column with contract/status categories such as nan, SuspCE, RFA, Core, Rookie, UFA, UDFA, Hardship, and Reserved. These categories may affect salary independently of performance. However, the raw data does not include full contract details, signing dates, contract length, or salary-cap context, so the model may omit important non-performance factors.

Bias and Representativeness: There may be selection bias after merging the salary and performance datasets. Since the salary dataset has 227 rows but the player-level performance datasets have 182 rows, some players may be excluded from the final modeling dataset. If the unmatched players are mostly low-minute players, injured players, temporary-contract players, or players who were waived or signed during the season, then the final regression dataset may overrepresent regular rotation players and underrepresent fringe roster players.

This matters because the fitted model would describe the salary-performance relationship only for players who appear in both the salary and performance datasets, not necessarily for all WNBA players.

## Notes for data_inventory.csv
- `key_columns`: Important columns for understanding the dataset
- `merge_keys`: Columns used to join this dataset with another dataset
Player names may need to be standardized before merging because the salary file uses `Player`, while other files use `player`.
- `target_or_features`: Describes the modeling role of the dataset.
- `coverage_notes`: Describes what the dataset covers and what it does not cover.
- `granularity_notes`: Describes the level of detail in the dataset.
- `bias_notes`: Describes possible selection bias or missing subpopulations.
- `limitations`: Describes known problems or cautions for using the dataset.

## Contract / Signing Status Notes (AI part starts from now)

The salary dataset includes a `2025 Signing` column. This column is not an on-court performance variable. Instead, it describes the player's contract or roster status. These categories may matter because salary is affected not only by performance, but also by contract rules and player status.

- `RFA`: Restricted Free Agent. A player whose current team may have the right to match other contract offers.
- `UFA`: Unrestricted Free Agent. A player who is generally free to sign with any team.
- `Rookie`: A player on a rookie contract or rookie-scale contract.
- `Core`: A player given a core designation by her team. This status may restrict free agency and affect salary negotiations.
- `UDFA`: Undrafted Free Agent. A player who was not drafted but signed as a free agent.
- `Hardship`: A player signed under a hardship exception, usually because a team needs temporary roster help due to injuries or limited available players.
- `Reserved`: A player whose rights are reserved by a team under league contract rules.
- `SuspCE`: A "suspended contract" in the WNBA refers to a procedural move where a player's salary remains on the team’s salary cap, but the player is removed from the active roster and does not occupy a standard roster spot. This is most commonly used for medical reasons or overseas/national team commitments. (Source:Yahoo Sports)
- `nan`: Missing value. The signing/status information is not available for that row.

This matters for modeling because players with different signing statuses may have different salary rules or negotiation conditions. Therefore, salary may not be fully explained by performance statistics alone. Before using `2025 Signing` as a model feature, we should clean and document these categories carefully.

## Terms explanations:

## Salary_2025 Dataset Column Notes

The `salary_2025.csv` file includes salary information, signing/status information, and basic player performance statistics. The columns are:

- `Player`: Player name.
- `2025 Salary`: Player salary for the 2025 season. This is the likely target variable for the regression model.
- `2025 Signing`: Player contract or signing status for the 2025 season. This is not an on-court performance variable, but it may affect salary.
- `G`: Games played.
- `GS`: Games started.
- `MIN`: Minutes played per game.
- `PTS`: Points per game.
- `FGM`: Field goals made per game.
- `FGA`: Field goals attempted per game.
- `FG%`: Field goal percentage.
- `2PM`: Two-point field goals made per game.
- `2PA`: Two-point field goals attempted per game.
- `2P%`: Two-point field goal percentage.
- `3PM`: Three-point field goals made per game.
- `3PA`: Three-point field goals attempted per game.
- `3P%`: Three-point field goal percentage.
- `FTM`: Free throws made per game.
- `FTA`: Free throws attempted per game.
- `FT%`: Free throw percentage.
- `ORB`: Offensive rebounds per game.
- `DRB`: Defensive rebounds per game.
- `TRB`: Total rebounds per game.
- `AST`: Assists per game.
- `TOV`: Turnovers per game.
- `STL`: Steals per game.
- `BLK`: Blocks per game.
- `PF`: Personal fouls per game.

From a modeling perspective, `2025 Salary` is the target variable. The basketball statistics such as `PTS`, `TRB`, `AST`, `STL`, `BLK`, `MIN`, and shooting percentages are potential performance-based features. The `2025 Signing` column is a categorical variable related to contract status, so it should be treated differently from numerical performance statistics.

## 2025_Advanced Statistics Dataset Column Notes

The `2025_advanced.csv` file contains player-level advanced statistics from the 2025 WNBA season. These variables are potential features for the regression model because they describe player efficiency, shooting profile, usage, and estimated contribution to winning.

- `pos`: Player position, such as guard, forward, or center.
- `mp`: Total minutes played.
- `per`: Player Efficiency Rating. A general player efficiency metric.
- `ts_pct`: True shooting percentage. This measures scoring efficiency while accounting for field goals, three-point shots, and free throws.
- `efg_pct`: Effective field goal percentage. This adjusts field goal percentage by giving extra weight to made three-point shots.
- `fg3a_per_fga_pct`: Three-point attempt rate. This is the proportion of field goal attempts that are three-point attempts.
- `fta_per_fga_pct`: Free throw attempt rate. This is the number of free throw attempts relative to field goal attempts.
- `orb_pct`: Offensive rebound percentage. This estimates the percentage of available offensive rebounds a player gets while on the court.
- `trb_pct`: Total rebound percentage. This estimates the percentage of available rebounds a player gets while on the court.
- `ast_pct`: Assist percentage. This estimates the percentage of teammate field goals assisted by the player while on the court.
- `stl_pct`: Steal percentage. This estimates the percentage of opponent possessions that end with a steal by the player.
- `blk_pct`: Block percentage. This estimates the percentage of opponent two-point field goal attempts blocked by the player.
- `tov_pct`: Turnover percentage. This estimates turnovers per offensive play used by the player.
- `usg_pct`: Usage percentage. This estimates the percentage of team offensive possessions used by the player while on the court.
- `off_rtg`: Offensive rating. This estimates points produced per 100 possessions.
- `def_rtg`: Defensive rating. This estimates points allowed per 100 possessions. Lower values are generally better, but player-level defensive ratings should be interpreted carefully.
- `DUMMY`: This column is unclear from the raw data alone. It may be an artifact from scraping or table formatting and should be inspected before modeling.
- `ows`: Offensive win shares. This estimates the player's contribution to team wins from offense.
- `dws`: Defensive win shares. This estimates the player's contribution to team wins from defense.
- `ws`: Total win shares. This combines offensive and defensive win shares.
- `ws_per_40`: Win shares per 40 minutes. This standardizes win shares by playing time.

From a modeling perspective, this dataset provides performance-based features rather than the salary target. Variables such as `per`, `ts_pct`, `usg_pct`, `ows`, `dws`, `ws`, and `ws_per_40` may be especially relevant for player valuation. However, some variables may be highly correlated with each other, so feature selection or regularization may be needed before fitting regression models.

## 2025_Per-Game Statistics Dataset Column Notes

The `2025_per_game.csv` file contains player-level per-game statistics for the 2025 WNBA season. This dataset provides basic box-score production variables and can be used as a feature dataset for salary or player valuation modeling.

- `mp_per_g`: Minutes per game.
- `fg_per_g`: Field goals made per game.
- `fga_per_g`: Field goals attempted per game.
- `fg_pct`: Field goal percentage.
- `fg3_per_g`: Three-point field goals made per game.
- `fg3a_per_g`: Three-point field goals attempted per game.
- `fg3_pct`: Three-point field goal percentage.
- `fg2_per_g`: Two-point field goals made per game.
- `fg2a_per_g`: Two-point field goals attempted per game.
- `fg2_pct`: Two-point field goal percentage.
- `ft_per_g`: Free throws made per game.
- `fta_per_g`: Free throws attempted per game.
- `ft_pct`: Free throw percentage.
- `orb_per_g`: Offensive rebounds per game.
- `trb_per_g`: Total rebounds per game.
- `ast_per_g`: Assists per game.
- `stl_per_g`: Steals per game.
- `blk_per_g`: Blocks per game.
- `tov_per_g`: Turnovers per game.
- `pf_per_g`: Personal fouls per game.
- `pts_per_g`: Points per game.

From a modeling perspective, this dataset gives direct player production features such as minutes, points, rebounds, assists, steals, blocks, turnovers, and shooting efficiency. These variables are easier to interpret than many advanced statistics, so they may be useful for a simple baseline regression model.

However, many of these features are mechanically related. For example, `fg_per_g`, `fga_per_g`, and `pts_per_g` are strongly connected, and `mp_per_g` may explain a large part of counting statistics such as points and rebounds. Because of this, we should check correlations before putting all per-game features into the same regression model.

## 2025_Total Statistics Dataset Notes

The `2025_totals.csv` file contains player-level season total statistics for the 2025 WNBA season. Most columns have the same basketball meaning as the per-game dataset, but the values are season totals rather than per-game averages.

For example, `pts` represents total points scored during the season, while `pts_per_g` in the per-game dataset represents points per game. 

From a modeling perspective, the total statistics dataset captures both player production and playing time. Players who play more games or minutes will naturally accumulate higher totals. Because of this, total statistics may be strongly influenced by availability and role, not only per-minute performance or efficiency.

## 2025_Team Advanced Statistics Dataset Notes

The `2025_advanced-team.csv` file contains team-level advanced statistics for the 2025 WNBA season. 

- `ranker`: Team ranking/index in the table.
- `team`: Team name.
- `age`: Average team age.
- `wins`: Number of wins.
- `losses`: Number of losses.
- `wins_pyth`: Pythagorean expected wins, based on scoring margin.
- `losses_pyth`: Pythagorean expected losses, based on scoring margin.
- `mov`: Margin of victory. Average point differential per game.
- `sos`: Strength of schedule. This measures the difficulty of the team's schedule.
- `srs`: Simple Rating System. A team rating that combines margin of victory and strength of schedule.
- `net_rtg`: Net rating. Difference between offensive rating and defensive rating.
- `pace`: Estimated number of possessions per game.
- `ts_pct`: True shooting percentage. Scoring efficiency including field goals, three-point shots, and free throws.
- `efg_pct`: Effective field goal percentage.
- `tov_pct`: Turnover percentage.
- `orb_pct`: Offensive rebound percentage.
- `ft_rate`: Free throw rate.
- `opp_efg_pct`: Opponent effective field goal percentage.
- `opp_tov_pct`: Opponent turnover percentage.
- `drb_pct`: Defensive rebound percentage.
- `opp_ft_rate`: Opponent free throw rate.
- `arena_name`: Home arena name.

From a modeling perspective, this dataset should not be directly used as the main player-level regression dataset because the unit of analysis is team-season, not player-season. However, it can be useful for team-level efficiency analysis. For example, after aggregating player salaries by team, we could compare total team salary with team performance measures such as wins, net rating, SRS, or offensive/defensive rating.

Warning: if we want to use team-level variables in a player-level salary model, we would need to merge team-level features onto each player by team. This should be done carefully because all players on the same team would receive the same team-level values, which may create dependence among observations.

## 2025_Team Standings Dataset Notes

The `2025_wnba_standings.csv` file contains team-level standings information for the 2025 WNBA season. Each row represents one team, not one player.

- `team_name`: Team name.
- `wins`: Number of wins.
- `losses`: Number of losses.
- `win_loss_pct`: Winning percentage.
- `gb`: Games behind the leading team. This measures how far a team is behind the top team in the standings.

This dataset should be treated as contextual team-level information. It can help answer questions about which teams get the most production per dollar, but it should not be directly mixed with player-level salary data without a clear merge strategy.
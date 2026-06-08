# Project Overview: WNBA Player Valuation Analysis

## Problem Definition
The WNBA is experiencing unprecedented commercial growth, with skyrocketing viewership, attendance, and sponsorship revenue. However, team salary caps remain strictly regulated under the current Collective Bargaining Agreement (CBA). Thus, front offices must optimize roster construction with limited financial resources.

The objective of this project is to develop a data-driven **Fair Market Value (FMV) engine** that predicts what a WNBA player should earn based on their on-court production. By comparing predicted salaries against actual contracts, this project aims to solve two core problems:
1. **Identifying Market Inefficiencies:** Pinpointing which players are statistically undervalued (providing high production at a low cost) or overvalued (providing low production at a high cost).
2. **Evaluating Franchise ROI:** Quantifying which teams are generating the highest on-court returns per dollar spent on player payroll.

## Dataset Description
The analysis utilizes an integrated, multi-source dataset capturing the **2025 WNBA season**. It includes both financial compensation with on-court performance metrics, resulting in a dataset of **224 player records and 123 distinct feature columns**. 

The dataset components include:
* **Financial Data:** Individual player salaries, and contract types (e.g., Rookie, Core, Restricted Free Agent).
* **Player Production Statistics:** Traditional per-game box scores (points, assists, rebounds), season totals, and advanced efficiency metrics (Player Efficiency Rating [PER], Usage Percentage, and Win Shares).
* **Team Context:** Team advanced statistics, season standings, and overall win-loss.

## Project Stakeholders
* **WNBA General Managers and Front Offices:** Seeking to maximize roster talent, identify high-efficiency free agents, and avoid overpaying for production.
* **Sports Agents and Players:** Requiring empirical valuation benchmarks to leverage during contract negotiations to ensure fair compensation based on on-court output.
* **Sports Analytics Firms and Media:** Objectively evaluating executive performance and ranking team management efficiency.

## Project KPIs
To evaluate the success of our valuation engine and its business utility, we track the following primary and secondary metrics:

* **Primary Predictive KPI: Root Mean Squared Error (RMSE)**
  * *Metric:* Measures the square root of the average squared differences between predicted and actual salaries. 
  * *Business Goal:* Minimize overall prediction variance and heavily penalize large errors, avoiding costly miscalculations on star players.
* **Secondary Fairness KPI: Mean Absolute Percentage Error (MAPE)**
  * *Metric:* Measures the percentage error relative to contract size.
  * *Business Goal:* Minimize error variance across contract scales, ensuring the model evaluates low-earning rookie contracts ($65k+) similarly to veteran supermax contracts ($250k+).
* **Business ROI KPI: Cost Per Win Share (CPWS)**
  * *Metric:* Calculated as **Total Team Payroll / Total Season Win Shares**.
  * *Business Goal:* Establish a benchmark for team spending efficiency, defining the optimal dollar amount required to "buy" a winning team in the modern WNBA market.

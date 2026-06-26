# Modeling Summary

We utilize a DummyRegressor as our trivial baseline that predicts historical mean salary. 
Linear Regression for a simple feature baseline that predicts using only points. 
Both OLS and Ridge Regression to establish the linear baseline. Our Basic Tree Baselines consist of Decision Tree 
and Random Forest Regressors. 

These models align with the primary, secondary, and business focused KPIs. 

`modeling_experiments.ipynb` showed that the Stacking Ensemble and ElasticNet had strong CV RMSE values, but the Stacking Ensemble is less interpretable and more complex because it combines several models, which can be risky with a small dataset. ElasticNet performed well, but it is part of the regularized linear model family already represented by Ridge. For this optimization step, we focused on RandomForest and Gradient Boosting because they are defensible tree-based ensemble models that can capture nonlinear relationships while keeping the tuning workflow simple and reproducible.


The final tuned Random Forest model achieves a 2025 holdout RMSE of 41,368.60 and an MAE of 30,376.25. Since RMSE is our primary predictive KPI, the main evaluation result is that the model's prediction error on the 2025 holdout set is about $41K under RMSE. The MAE means that, on average, the model's player-level salary prediction is off by about $30K.

The model also has an R² of 0.634. This means that the model explains a moderate amount of variation in 2025 player salaries compared with a simple mean-prediction baseline. However, R² is only an additional diagnostic metric here. It should not be used as the main reason for selecting the model, because the project prioritizes salary prediction error measured in dollars.

The MAPE is 1.287, or about 128.7%. This value should be interpreted carefully. MAPE divides each absolute error by the player's actual salary, so players with smaller actual salaries can produce very large percentage errors even when the dollar error is not unusually large. Because `model_comparison.ipynb` reports one overall MAPE rather than separating the evaluation by contract size, the high MAPE may partly reflect the sensitivity of the metric to small salary denominators.

Overall, the final model is reasonable under the RMSE-first rule. Given the small dataset and the instability of aggregate MAPE in this setting, RMSE and MAE are more reliable for the current audit. MAPE should still be reported, but it should not be overemphasized as a model-selection metric without a contract-size-aware analysis. CPWS is not evaluated in this notebook and should be left as future work for the final project presentation.

 










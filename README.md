# San Francisco Crime Classification (Kaggle)

This repository contains an end-to-end machine learning pipeline for predicting crime categories in San Francisco using structured, spatiotemporal, and textual data. The project demonstrates advanced feature engineering, model experimentation, and evaluation for tabular data.

# Directory Structure

```
dataset
     train.csv
     test.csv
checkpoints
      mlp.pth
      logistic_regression.pkl
submissions
     mlp_submission.csv
     lightgbm_submission.csv
     xgboost_submission.csv
```

# Feature Enigneering

The preprocessing pipeline performs extensive feature extraction and transformation to capture spatial, temporal, and textual patterns. Key steps include:

* ```Spatial Features``` Latitude and longitude are processed to generate linear combinations (X+Y, X-Y) and radial distance (XY_rad). PCA is applied to capture principal geographic patterns. Gaussian Mixture clustering is used to create geo-clusters representing dense crime areas.

* ```Address Features``` Street addresses are parsed into STREET, BLOCK, and INTERSECTION types. Word2Vec embeddings are trained on addresses to capture semantic similarity and spatial relationships. A binary feature indicates intersections. Frequency encoding is applied to high-cardinality categorical features such as street names, address types, and districts.

* ```Temporal Features``` Date and time are parsed to generate hour, day, month, year, minute, and day-of-week features. Binary indicators are created for night and weekend occurrences.

* ```Numeric Scaling``` All numeric features, including engineered spatial and temporal variables, are standardized using StandardScaler for improved model convergence.

* ```Categorical Encoding``` Categorical features such as day-of-week, district, address type, and geo-cluster are label-encoded for model compatibility. High cardinality categorical variables such as street name police district address type and day of week are encoded using frequency encoding and label encoding. 

# Models
Multiple models are trained and evaluated using a consistent train validation split with multi class log loss as the primary metric. 
* ```Baselines Model```Baseline performance is established using multinomial logistic regression. 
* ```Tree Models ``` These includes RandomForest, XGBoost and LightGBM are trained with carefully tuned hyperparameters using Optuna. 
* ```MLP``` A custom PyTorch based multilayer perceptron is implemented for tabular data with a deep architecture of 2048 1024 and 512 hidden units with dropout regularization and trained using cross entropy loss.

# Evaluation

Model predictions are compared on validation performance and the best configurations are used to generate Kaggle submission files. The final solution achieves a multi class log loss of approximately 2.22804 and ranked 61st against public leaderboard.
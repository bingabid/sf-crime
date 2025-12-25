# San Francisco Crime Classification (Kaggle)

This repo implements an end to end machine learning pipeline for the Kaggle San Francisco Crime Classification. The goal is to predict the crime category given spatiotemporal and contextual information from police reports.

# Feature Enigneering

The preprocessing pipeline performs extensive feature engineering on raw tabular data. 
* Temporal features are extracted from timestamps including hour day month year weekend indicators and night time flags.
* Geospatial information from latitude and longitude is enriched using radial distance features principal component analysis and Gaussian Mixture based geo clustering to capture spatial crime patterns. 
* Address text is modeled using Word2Vec embeddings trained on street level tokens to encode semantic structure in locations. 
* High cardinality categorical variables such as street name police district address type and day of week are encoded using frequency encoding and label encoding. 
* All numerical features are standardized while binary indicators are kept unscaled.

# Models

Multiple models are trained and evaluated using a consistent train validation split with multi class log loss as the primary metric. 
* Baseline performance is established using multinomial logistic regression. 
* Tree based models including XGBoost and LightGBM are trained with carefully tuned hyperparameters using Optuna. 
* A custom PyTorch based multilayer perceptron is implemented for tabular data with a deep architecture of 2048 1024 and 512 hidden units with dropout regularization and trained using cross entropy loss.

# Performance

Model predictions are compared on validation performance and the best configurations are used to generate Kaggle submission files. The final solution achieves a multi class log loss of approximately 2.22804 and ranked 61st against public leaderboard.
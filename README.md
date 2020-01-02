# Movie rating prediction project
The goal of this project was to create a regression machine learning model to predict movie ratings
# Dataset
https://www.kaggle.com/deepak525/investigate-tmdb-movie-dataset/data
# Data preprocessing
Dataset contains records with information about 10000 TMDB movies. The following features from the dataset were used: 'cast', 'director', 'budget', 'genres', 'production_companies', 'runtime', 'release_year' to predict 'vote_average' which is an average TMDB movie rating. For each movie for feature 'cast' only 5 leading actors were taken into account. Features 'cast', 'director', 'genres' and 'production_companies' were preprocessed with one-hot encoder using only 500 most frequent values for 'cast', 'director' and 'production_companies'.
# Model training
The library chosen for this task was xgboost with XGBRegressor model and gpu_hist method, which enables GPU support. After GridSearch the following best parameters were chosen to train the model: 'alpha': 0, 'eta': 0.3, 'lambda': 1, 'max_depth': 4, 'n_estimators': 200.
# Model evaluation
The following results were obtained on the test set: 

MSE: 0.5137510044416175

RMSE: 0.7167642600197205

MAE: 0.5616956795476804

R2: 0.312384126617107

Feature importances extracted from the model show that the 10 most important features chosen by the model during training were: 
1. Runtime
2. Drama ('genres')
3. Science Fiction ('genres')
4. Adventure ('genres')
5. Christopher Lloyd ('cast')
6. Horror ('genres')
7. Charlie Sheen ('cast')
8. Dan Hedaya ('cast')
9. Documentary ('genres')
10. DC Comics ('production_companies')
# Possible improvements
To improve the performance of the model the following changes could be implemented:
- including more values for 'cast', 'director' and 'production_companies' because limiting them to 500 most frequent values could have a very negative impact on the model performance
- trying a different model like random forest or neural network

import os
import sys
import getpass
import bz2
import pickle # Rick!
import copy
import math

import pandas as pd
import numpy as np
from scipy.io import loadmat

from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import scale, MinMaxScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, make_scorer

import random

#Set seed
random.seed(0)
np.random.seed(0)

user = getpass.getuser()
os.chdir("/Users/"+user+"/Documents/D/Niebla/")

#Import aux files
sys.path.append('src/aux/')
import aux_functions

aux_functions.check_folder_structure("MLP_temp_split")

print(" # Loading data from csv files")
train_data = pd.read_csv("Data/csv/temp_split/train_data.csv")
test_data  = pd.read_csv("Data/csv/temp_split/test_data.csv")

n_cols = train_data.shape[1]

train_x = scale(train_data.iloc[:,:(n_cols-1)])
test_x  = scale(test_data.iloc[:,:(n_cols-1)])

################################################################################
# MLP Experiment
################################################################################

#Hacer un grid search con un MLP
params = [{"solver": ["sgd"], "hidden_layer_sizes": [[25],[50],[75],[25,25],[25,50],[50,50], [10,25,10]],
		  "activation": ['logistic', 'tanh', 'relu'],
		  "momentum":[.9,.85], "learning_rate_init": [0.001, 0.0025, 0.005]},
		  {"solver": ["adam"], "hidden_layer_sizes": [[25],[50],[75],[25,25],[25,50],[50,50], [10,25,10]],
		  "activation": ['logistic', 'tanh', 'relu'], "learning_rate_init": [0.001, 0.0025, 0.005]}]

# Choose regressor
mlp = MLPRegressor()
# Create GridSearch to optimze parameters
grid_search = GridSearchCV(mlp, params, scoring="neg_root_mean_squared_error", verbose=3, n_jobs=-1)

# Run grid search
grid_search.fit(train_x, train_data["vs"])

print( "  * RMSE: " + str(grid_search.best_score_))

# ---------------------------------------------------------------

print(" # Train best config with best GridSearch parameters")

# Refit best model on train data
mlp = MLPRegressor(**grid_search.best_params_)
print("  * Training")
mlp.fit(train_x, train_data["vs"])

# Predict test data
print("  * Testing")
y_pred = mlp.predict(test_x)

test_metrics = aux_functions.get_metrics(test_data["vs"], y_pred)

print("  * Metrics")
print(test_metrics)
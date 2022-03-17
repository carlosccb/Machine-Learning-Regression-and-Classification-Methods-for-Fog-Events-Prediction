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
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import scale, MinMaxScaler, RobustScaler
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

aux_functions.check_folder_structure("MLP")


def get_metrics(y_true, y_pred):
	r2 = r2_score(y_true, y_pred)
	print("  * R2: " + str(r2))
	MSE = mean_squared_error(y_true, y_pred)
	print("  * MSE: " + str(MSE))
	RMSE = math.sqrt(MSE)
	print("  * RMSE: " + str(RMSE))
	MAE = mean_absolute_error(y_true, y_pred)
	print("  * MAE: " + str(MAE))

	return {"R2": r2, "MSE": MSE,
			"RMSE": RMSE, "MAE": MAE}

	# with open(ALG_COMP_FILE, 'a') as f:
	# 	pd.DataFrame({clasif_name: [r2, RMSE, MAE]}).T.to_csv(f, header=False)

##########################################
print(" # Loading pickled data")

with bz2.BZ2File("./Data/Pickle/Dataset.pbz2", "rb") as f:
	dataset_dict = pickle.load(f)

print("  * Data loaded successfully")

#######
# Meter los datos en dos variables
data_x_ = np.vstack((dataset_dict["Train"][0]["X"], dataset_dict["Test"][0]["X"]))
data_y = np.concatenate((dataset_dict["Train"][0]["y"], dataset_dict["Test"][0]["y"]))

#######
#Normalizar las variables
scaler = MinMaxScaler()
#scaler = RobustScaler()
data_x = scaler.fit_transform(data_x_)

#Hacer un grid search con un MLP
params = [{"solver": ["sgd"], "hidden_layer_sizes": [[25],[50],[75],[25,25],[25,50],[50,50], [10,25,10]],
		  "activation": ['logistic', 'tanh', 'relu'],
		  "momentum":[.9,.85], "learning_rate_init": [0.001, 0.0025, 0.005]},
		  {"solver": ["adam"], "hidden_layer_sizes": [[25],[50],[75],[25,25],[25,50],[50,50], [10,25,10]],
		  "activation": ['logistic', 'tanh', 'relu'], "learning_rate_init": [0.001, 0.0025, 0.005]}
		  ]

# GridSearch con MLP
mlp = MLPRegressor(early_stopping=True, learning_rate="adaptive")
grid_search = GridSearchCV(mlp, params, n_jobs=-1, scoring="neg_root_mean_squared_error", verbose=3)

print(" # Fitting MLP model with GridSearch")
grid_search.fit(data_x, data_y)

#best_mlp = grid_search.best_estimator_

print()
print(" @ Best parameters: ")
print(grid_search.best_params_)
print()
print(" @ Best score:")
print(grid_search.best_score_)
print()

"""
MLPRegressor(activation='relu', alpha=0.0001, batch_size='auto', beta_1=0.9,
             beta_2=0.999, early_stopping=True, epsilon=1e-08,
             hidden_layer_sizes=[50, 75], learning_rate='adaptive',
             learning_rate_init=0.001, max_fun=15000, max_iter=200,
             momentum=0.8, n_iter_no_change=10, nesterovs_momentum=True,
             power_t=0.5, random_state=None, shuffle=True, solver='adam',
             tol=0.0001, validation_fraction=0.1, verbose=False,
             warm_start=False)
"""

k_fold_metrics = {"R2": [], "MSE": [], "RMSE": [], "MAE": []}
plots_fold = {}

for i in range(5):
	print(" # Running fold " + str(i))
	#Crear modelo con los mejores parámetros del gridsearch
	mlp_fold = MLPRegressor(**grid_search.best_params_)
	#mlp_fold.set_params(**grid_search.best_params_)

	print("  * Training")
	x_tr = scaler.transform(dataset_dict["Train"][i]["X"])
	#Entrenar el modelo con los datos del fold i
	mlp_fold.fit(x_tr, dataset_dict["Train"][i]["y"])
	print("  * Predicting")
	#Obtener las predicciones de test para el fold i
	x_ts = scaler.transform(dataset_dict["Test"][i]["X"])
	y_pred = mlp_fold.predict(x_ts)
	y_pred = trim_preds(y_pred)
	#Obtener las métricas de test para el fold i
	metrics = get_metrics(dataset_dict["Test"][i]["y"], y_pred)

	#Plot predictions vs true
	plots_fold[i] = pd.DataFrame({"y_pred":y_pred, "y_true":dataset_dict["Test"][i]["y"]})
	plots_fold[i].plot()

	for key in k_fold_metrics.keys():
		k_fold_metrics[key].append(metrics[key])

k_fold_metrics_df = pd.DataFrame(k_fold_metrics)
k_fold_metrics_df.mean()
"""
R2           0.519234
MSE     324492.318722
RMSE       568.610429
MAE        395.235247
"""

metrics_df = pd.DataFrame(k_fold_metrics_df.mean(), columns=["MLP"])


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
from sklearn.preprocessing import scale, MinMaxScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, make_scorer

import random

#Set seed
random.seed(0)
np.random.seed(0)


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


user = getpass.getuser()
os.chdir("/Users/"+user+"/Documents/D/Niebla/")

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
data_x = scaler.fit_transform(data_x_)

##########################################
print(" # Train a linear regression model")
print("   ... Do ML stuff ...")

k_fold_metrics = {"R2": [], "MSE": [], "RMSE": [], "MAE": []}

for i in range(5):
	print(" # Fold " + str(i))
	lm = LinearRegression(n_jobs=-1)
	print("  * Training model")
	lm.fit(dataset_dict["Train"][i]["X"], dataset_dict["Train"][i]["y"])
	print("  * Predicting")
	y_pred = lm.predict(dataset_dict["Test"][i]["X"])
	#RMSE = np.sqrt(mean_squared_error(dataset_dict["Test"][i]["y"], y_pred))

	metrics = get_metrics(dataset_dict["Test"][i]["y"], y_pred)

	for key in k_fold_metrics.keys():
		k_fold_metrics[key].append(metrics[key])

print()
print(" # Mean metrics")
k_fold_metrics_df = pd.DataFrame(k_fold_metrics)
print(k_fold_metrics_df.mean())

#TODO: Save metrics to file
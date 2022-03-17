import random
import os, sys
import getpass

from itertools import product

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import scale, MinMaxScaler, StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, make_scorer

from hpelm import ELM

#For RBF act fn
from sklearn.cluster import KMeans
from scipy.spatial import distance
from numpy import linalg

user = getpass.getuser()
os.chdir("/Users/"+user+"/Documents/D/Niebla/")

#Import aux files
sys.path.append('src/aux/')
import aux_functions


train_df = pd.read_csv("./Data/csv/temp_split/train_data.csv")
test_df = pd.read_csv("./Data/csv/temp_split/test_data.csv")

ncols=train_df.shape[1]

random.seed(1234)
np.random.seed(1234)

scaler = StandardScaler()

train_x = scaler.fit_transform(train_df.iloc[:,:(ncols-1)])
test_x  = scaler.transform(test_df.iloc[:,:(ncols-1)])

"""
Documentación:
	* https://hpelm.readthedocs.io/en/latest/hpelm.html
Se pueden meter tres tipo de neuronas:
	* sigm
	* tanh
	* rbf: Hay que meter W y B.
		https://hpelm.readthedocs.io/en/latest/hpelm.html#hpelm.elm.ELM.add_neurons
"""

#n_neu = np.linspace(10,60,11, dtype=int)
n_neu = np.arange(5,100,5)
fn_trns = ["sigm", "tanh"]

#Dictionaries to store RMSE for each configuration
loo_rmse = {"tanh":[], "sigm": []}#, "rbf_l2": []}
cv_rmse  = {"tanh":[], "sigm": []}#, "rbf_l2": []}

best_model = {"config": {"n_neu":0,"fn_trns":"", "cv_method":""}, "y_pred": [], "RMSE": np.inf, "Metrics": {}}

for i in product(n_neu, fn_trns):
	print("\n -> ELM with {} {}".format(i[0], i[1]))

	# -- -- --  -- -- --   -- -- --   -- -- --   -- -- --   -- -- --  

	# Entrenar con LOO
	print("  # Training with LOO")
	elm = ELM(ncols-1, 1)
	elm.add_neurons(int(i[0]), i[1])
	elm.train(train_x, train_df["vs"].values, "LOO", "r")

	y_pred = elm.predict(test_x)
	y_pred[y_pred>2000]=2000; y_pred[y_pred<0]=0

	#Sacar el error
	metrics = aux_functions.get_metrics(test_df["vs"].values, y_pred)
	loo_rmse[i[1]].append(metrics["RMSE"])
	#print(metrics)

	if metrics["RMSE"]<best_model["RMSE"]:
		best_model["RMSE"] = metrics["RMSE"]
		best_model["config"]["n_neu"] = int(i[0])
		best_model["config"]["fn_trns"] = i[1]
		best_model["config"]["cv_method"] = "LOO"
		best_model["y_pred"] = y_pred
		best_model["Metrics"] = metrics



	# -- -- --  -- -- --   -- -- --   -- -- --   -- -- --   -- -- --  

	# Entrenar con 3-fold
	print("  # Training with 3-fold")
	elm = ELM(ncols-1, 1)
	elm.add_neurons(int(i[0]), i[1])
	elm.train(train_x, train_df["vs"].values, "CV", "r", k=3)

	y_pred = elm.predict(test_x)
	y_pred[y_pred>2000]=2000; y_pred[y_pred<0]=0

	#Sacar el error
	metrics = aux_functions.get_metrics(test_df["vs"].values, y_pred)
	cv_rmse[i[1]].append(metrics["RMSE"])
	#print(metrics)

	if metrics["RMSE"]<best_model["RMSE"]:
		best_model["RMSE"] = metrics["RMSE"]
		best_model["config"]["n_neu"] = int(i[0])
		best_model["config"]["fn_trns"] = i[1]
		best_model["config"]["cv_method"] = "3fold"
		best_model["y_pred"] = y_pred
		best_model["Metrics"] = metrics

# #Como RBF necesita más parámetros meter en un bucle diferente
# for i in n_neu:
# 	print("\n\n -> ELM with {} rbf_l2".format(i))

# 	# -- -- --  -- -- --   -- -- --   -- -- --   -- -- --   -- -- --  
# 	#b = np.zeros(i)
# 	kmedias = KMeans(n_clusters=i, init="random", n_init=1, max_iter=500)
# 	kmedias.fit(train_x)
# 	distancias = distance.cdist(train_x, kmedias.cluster_centers_, metric='euclidean')
# 	# dists = distance.cdist(kmedias.cluster_centers_, kmedias.cluster_centers_, metric='euclidean')
# 	# radios = dists.sum(axis=1)* (1.0/(2.0*(i-1.0)))

# 	# Entrenar con LOO
# 	print("  # Training with LOO")
# 	elm = ELM(ncols-1, 1)
# 	elm.add_neurons(int(i), "rbf_l2", W=distancias, B=np.ones(i))
# 	elm.train(train_x, train_df["vs"].values, "LOO", "r")

# 	y_pred = elm.predict(test_x)

# 	#Sacar el error
# 	metrics = aux_functions.get_metrics(test_df["vs"].values, y_pred)
# 	loo_rmse["rbf_l2"].append(metrics["RMSE"])
# 	#print(metrics)

# 	# -- -- --  -- -- --   -- -- --   -- -- --   -- -- --   -- -- --  

# 	# Entrenar con 3-fold
# 	print("  # Training with 3-fold")
# 	elm = ELM(ncols-1, 1)
# 	elm.add_neurons(int(i), "rbf_l2")#, B=b)
# 	elm.train(train_x, train_df["vs"].values, "CV", "r", k=3)

# 	y_pred = elm.predict(test_x)

# 	#Sacar el error
# 	metrics = aux_functions.get_metrics(test_df["vs"].values, y_pred)
# 	cv_rmse["rbf_l2"].append(metrics["RMSE"])
# 	#print(metrics)



##################################################################################################################
# Plot Stuff
##################################################################################################################


# def plot_(y, title):
# 	sns.scatterplot(x=n_neu, y=y)
# 	sns.lineplot(x=n_neu, y=y)
# 	plt.xlabel("N neuronas")
# 	plt.ylabel("RMSE")
# 	plt.title(title)

# # Plot RMSE one by one
# plot_(loo_rmse["tanh"], "tanh - LOO RMSE")
# plot_(cv_rmse["tanh"], "tanh - 3-Fold RMSE")

# plot_(loo_rmse["sigm"], "simg - LOO RMSE")
# plot_(cv_rmse["sigm"], "simg - 3-Fold RMSE")

def plot(data, title):
	#sns.scatterplot(data=data)
	#Plot logarithmic y axis
	#_, ax = plt.subplots()
	#ax.set(yscale="log")
	#Plot data
	sns.lineplot(data=data)
	plt.xlabel("N neuronas")
	plt.ylabel("RMSE")
	plt.title(title)

#Plot in comparison
loo_df = pd.DataFrame(loo_rmse, index=n_neu)
# plot(loo_df, "LOO comp")

cv_df = pd.DataFrame(cv_rmse, index=n_neu)
# plot(cv_df, "3-fold comp")

# Plot general comparison
df = loo_df.join(cv_df, lsuffix="_loo", rsuffix="_cv")
plot(data=df, title="General comparison")
plt.savefig("./plots/ELM_comparison_.png")


#--------------------------------------------------
# Guardar los mejores resultados en los ficheros con el resto de experimentos
#

# TODO: Save best model config and predictions by comparing current RMSE to best until now
print(" # Saving predictions")
if os.path.isfile(aux_functions.PREDICTIONS_FILE):
	df = pd.read_csv(aux_functions.PREDICTIONS_FILE)
	df["ELM_best-temp_split"] = best_model["y_pred"]
	df.to_csv(aux_functions.PREDICTIONS_FILE, index=False)
	print("  -> Predictions Saved")
else:
	print("  -> File doesn't exist!! " + aux_functions.PREDICTIONS_FILE)

print(" # Saving best RMSE")
if os.path.isfile(aux_functions.ALG_COMP_FILE):
	df = pd.read_csv(aux_functions.ALG_COMP_FILE)
	df["ELM_best-temp_split"] = best_model["Metrics"].values()
	df.to_csv(aux_functions.ALG_COMP_FILE, index=False)
	print("  -> Saved")
else:
	print("  -> File doesn't exist!! " + aux_functions.ALG_COMP_FILE)

print(" # Saving best configuration")
pd.DataFrame.from_dict(best_model["config"], orient="index").to_csv(aux_functions.BEST_CONFIG+"temp_split/ELM.csv")

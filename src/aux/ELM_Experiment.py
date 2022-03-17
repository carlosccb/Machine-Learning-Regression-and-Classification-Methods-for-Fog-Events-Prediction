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
from Experiment import Experiment

random.seed(1234)
np.random.seed(1234)

"""
Documentación:
	* https://hpelm.readthedocs.io/en/latest/hpelm.html
Se pueden meter tres tipo de neuronas:
	* sigm
	* tanh
	* rbf: Hay que meter W y B.
		https://hpelm.readthedocs.io/en/latest/hpelm.html#hpelm.elm.ELM.add_neurons
"""

class ELM_Experiment(Experiment):
	def __init__(self, train_x, train_y, test_x, test_y, model_name, exp_name, ncols, regression=True):
		self.best_model = {"config": {"n_neu":0,"fn_trns":"", "cv_method":""},
						   "y_pred": [], "RMSE": np.inf, "Metrics": {}}
		self.train_x, self.train_y = train_x, train_y
		self.test_x,  self.test_y  = test_x,  test_y

		self.ncols = ncols

		self.model_name = model_name
		self.exp_name = exp_name

		self.exp_mod_name = self.model_name+"-"+self.exp_name

		# ---------------------
		#Things to make inheritance work

		self.regression = regression

		#Grid search is done manually, set to None so it works with inheritance of Experiment
		self.grid_search = None

		self.ALG_COMP_FILE    = aux_functions.ALG_COMP+"alg_comp-"+exp_name+".csv"
		self.PREDICTIONS_FILE = aux_functions.PREDICTIONS_FOLDER+"preds-"+exp_name+".csv"
		self.DURATION_FILE    = aux_functions.DURATION_FOLDER+exp_name+".csv"

		self.timming = False

		# ---------------------

	def run_grid_search_exp(self):
		#n_neu = np.linspace(10,60,11, dtype=int)
		n_neu = np.arange(5,100,5)
		fn_trns = ["sigm", "tanh"]

		#Dictionaries to store RMSE for each configuration
		loo_rmse = {"tanh":[], "sigm": []}#, "rbf_l2": []}
		cv_rmse  = {"tanh":[], "sigm": []}#, "rbf_l2": []}

		self.time_exp()
		for i in product(n_neu, fn_trns):
			print("\n -> ELM with {} {}".format(i[0], i[1]))

			# -- -- --  -- -- --   -- -- --   -- -- --   -- -- --   -- -- --  

			# Entrenar con LOO
			print("  # Training with LOO")
			elm = ELM(self.ncols-1, 1)
			elm.add_neurons(int(i[0]), i[1])
			elm.train(self.train_x, self.train_y, "LOO", "r")

			self.y_pred = elm.predict(self.test_x)
			self.trim_preds(2000)

			#Sacar el error
			metrics = aux_functions.get_metrics(self.test_y, self.y_pred)
			loo_rmse[i[1]].append(metrics["RMSE"])
			#print(metrics)

			if metrics["RMSE"]<self.best_model["RMSE"]:
				self.get_metrics()
				self.best_model["RMSE"] = metrics["RMSE"]
				self.best_model["config"]["n_neu"] = int(i[0])
				self.best_model["config"]["fn_trns"] = i[1]
				self.best_model["config"]["cv_method"] = "LOO"
				self.best_model["y_pred"] = self.y_pred
				self.best_model["Metrics"] = metrics



			# -- -- --  -- -- --   -- -- --   -- -- --   -- -- --   -- -- --  

			# Entrenar con 3-fold
			print("  # Training with 3-fold")
			elm = ELM(self.ncols-1, 1)
			elm.add_neurons(int(i[0]), i[1])
			elm.train(self.train_x, self.train_y, "CV", "r", k=3)

			self.y_pred = elm.predict(self.test_x)
			self.trim_preds(2000)

			#Sacar el error
			metrics = aux_functions.get_metrics(self.test_y, self.y_pred)
			cv_rmse[i[1]].append(metrics["RMSE"])
			#print(metrics)

			if metrics["RMSE"]<self.best_model["RMSE"]:
				self.get_metrics()
				self.best_model["RMSE"] = metrics["RMSE"]
				self.best_model["config"]["n_neu"] = int(i[0])
				self.best_model["config"]["fn_trns"] = i[1]
				self.best_model["config"]["cv_method"] = "3fold"
				self.best_model["y_pred"] = self.y_pred
				self.best_model["Metrics"] = metrics
		self.time_exp()

		#Refit and get metrics

		# #Como RBF necesita más parámetros meter en un bucle diferente
		# for i in n_neu:
		# 	print("\n\n -> ELM with {} rbf_l2".format(i))

		# 	# -- -- --  -- -- --   -- -- --   -- -- --   -- -- --   -- -- --  
		# 	#b = np.zeros(i)
		# 	kmedias = KMeans(n_clusters=i, init="random", n_init=1, max_iter=500)
		# 	kmedias.fit(self.train_x)
		# 	distancias = distance.cdist(self.train_x, kmedias.cluster_centers_, metric='euclidean')
		# 	# dists = distance.cdist(kmedias.cluster_centers_, kmedias.cluster_centers_, metric='euclidean')
		# 	# radios = dists.sum(axis=1)* (1.0/(2.0*(i-1.0)))

		# 	# Entrenar con LOO
		# 	print("  # Training with LOO")
		# 	elm = ELM(ncols-1, 1)
		# 	elm.add_neurons(int(i), "rbf_l2", W=distancias, B=np.ones(i))
		# 	elm.train(self.train_x, self.train_y, "LOO", "r")

		# 	y_pred = elm.predict(self.test_x)

		# 	#Sacar el error
		# 	metrics = aux_functions.get_metrics(self.test_y, y_pred)
		# 	loo_rmse["rbf_l2"].append(metrics["RMSE"])
		# 	#print(metrics)

		# 	# -- -- --  -- -- --   -- -- --   -- -- --   -- -- --   -- -- --  

		# 	# Entrenar con 3-fold
		# 	print("  # Training with 3-fold")
		# 	elm = ELM(ncols-1, 1)
		# 	elm.add_neurons(int(i), "rbf_l2")#, B=b)
		# 	elm.train(self.train_x, self.train_y, "CV", "r", k=3)

		# 	y_pred = elm.predict(self.test_x)

		# 	#Sacar el error
		# 	metrics = aux_functions.get_metrics(self.test_y, y_pred)
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

	def _plot(data, title):
		#sns.scatterplot(data=data)
		#Plot logarithmic y axis
		#_, ax = plt.subplots()
		#ax.set(yscale="log")
		#Plot data
		sns.lineplot(data=data)
		plt.xlabel("N neuronas")
		plt.ylabel("RMSE")
		plt.title(title)

	def plot_comp(self):
		#Plot in comparison
		loo_df = pd.DataFrame(loo_rmse, index=n_neu)
		# self._plot(loo_df, "LOO comp")

		cv_df = pd.DataFrame(cv_rmse, index=n_neu)
		# self._plot(cv_df, "3-fold comp")

		# Plot general comparison
		df = loo_df.join(cv_df, lsuffix="_loo", rsuffix="_cv")
		self._plot(data=df, title="General comparison")
		plt.savefig("./plots/ELM_comparison_.png")

	#--------------------------------------------------
	# Guardar los mejores resultados en los ficheros con el resto de experimentos
	#
	# def save_logs(self):
	# 	"""
	# 	TODO: Revisar que esto esté bien
	# 	# TODO: Save best model config and predictions by comparing current RMSE to best until now
	# 	"""
	# 	print(" # Saving predictions")
	# 	if os.path.isfile(aux_functions.PREDICTIONS_FILE):
	# 		df = pd.read_csv(aux_functions.PREDICTIONS_FILE)
	# 		df[self.exp_mod_name] = self.best_model["y_pred"]
	# 		df.to_csv(aux_functions.PREDICTIONS_FILE, index=False)
	# 		print("  -> Predictions Saved")
	# 	else:
	# 		print("  -> File not found -> " + aux_functions.PREDICTIONS_FILE)

	# 	print(" # Saving best RMSE")
	# 	if os.path.isfile(aux_functions.ALG_COMP_FILE):
	# 		df = pd.read_csv(aux_functions.ALG_COMP_FILE)
	# 		df[self.exp_mod_name] = self.best_model["Metrics"].values()
	# 		df.to_csv(aux_functions.ALG_COMP_FILE, index=False)
	# 		print("  -> Saved")
	# 	else:
	# 		print("  -> File not found -> " + aux_functions.ALG_COMP_FILE)

	# 	print(" # Saving best configuration")
	# 	pd.DataFrame.from_dict(self.best_model["config"], orient="index").to_csv(aux_functions.BEST_CONFIG+"temp_split/ELM.csv")

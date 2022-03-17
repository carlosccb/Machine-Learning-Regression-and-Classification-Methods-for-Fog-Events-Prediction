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

#Optimization
from sklearn.model_selection import GridSearchCV
#Regression Models
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LinearRegression, ElasticNet
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import BaggingRegressor, AdaBoostRegressor, GradientBoostingRegressor
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel, RBF
#Data preprocesing
from sklearn.preprocessing import scale, StandardScaler, MinMaxScaler, RobustScaler
#Metrics
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, make_scorer

import random

#Set seed
random.seed(1234)
np.random.seed(1234)

user = getpass.getuser()
os.chdir("/Users/"+user+"/Documents/D/Niebla/")

#Import aux files
sys.path.append('src/aux/')
import aux_functions
from Experiment import Experiment

print(" # Loading data from csv files")
train_data = pd.read_csv("Data/csv/temp_split/train_data.csv")
test_data  = pd.read_csv("Data/csv/temp_split/test_data.csv")

n_cols = train_data.shape[1]
PCA_b = False

"""
	Probar varios tipos de escala a ver si se consigue mejora
"""
#Validate input
if len(sys.argv) == 1:
	print()
	print(" @  No experiment selected!!!")
	print(" @ Running default experiment! -> 0: StandardScaler")
	print()
	exp_n = 0
else:
	exp_n = int(sys.argv[1])
	if exp_n not in [0,1,2,9]:
		print()
		print(" # Wrong experiment number!!!")
		sys.exit(1)

# Choose scaler
# Set global variable of experiment name
if exp_n==0:
	scaler = StandardScaler()
	EXP_NAME = "temp_split-std"
elif exp_n == 1:
	scaler = MinMaxScaler()
	EXP_NAME = "temp_split-min_max"
elif exp_n == 2:
	scaler = RobustScaler()
	EXP_NAME = "temp_split-robust_scaler"
elif exp_n == 9:
	PCA_b = True
	EXP_NAME = "temp_split-PCA"

# Para PCA no hay escalado previo
if exp_n != 9:
	#Hacer el escalado de las datos
	train_x = scaler.fit_transform(train_data.iloc[:,:(n_cols-1)])
	test_x  = scaler.transform(test_data.iloc[:,:(n_cols-1)])
	#test_x  = scaler.fit_transform(test_data.iloc[:,:(n_cols-1)])
else:
	train_x = train_data.iloc[:,:(n_cols-1)]
	test_x  = test_data.iloc[:,:(n_cols-1)]


print()
print(" -------------------------------------------")
print()
print("   Running experiment -> " + EXP_NAME)
print()
print(" -------------------------------------------")
print()


################################################################################
# MLP Experiment
################################################################################

def mlp_exp():
	# # Choose regressor
	print("\n\n @ Running Experiment with MLP with GridSearch")
	mlp = MLPRegressor(learning_rate='adaptive', early_stopping=True, max_iter=1000)

	# #Hacer un grid search con un MLP
	params = [{"solver": ["sgd"], "hidden_layer_sizes": [[50],[75],[100],[125],[25,25],[50,50]], #[10],[25],[25,50],[50,50], [10,25,10]
			  "activation": ['logistic', 'tanh',],
			  "momentum":[.9,.85,.7], "learning_rate_init": np.arange(0.0001,0.005,.00075)},
			  {"solver": ["adam"], "hidden_layer_sizes": [[50],[75],[100],[125],[25,25],[50,50]], #[10],[25],[25,50],[50,50], [10,25,10]
			  "activation": ['logistic', 'tanh',], "learning_rate_init": np.arange(0.0001,0.005,.00075)}]
	# params = [{"solver": ["sgd"], "hidden_layer_sizes": [25],"activation": ['logistic'], "momentum":[.9], "learning_rate_init": [0.001]},
	# 		  {"solver": ["adam"], "hidden_layer_sizes": [25], "activation": ['logistic'], "learning_rate_init": [0.001]}]

	# Create GridSearch to optimze parameters
	grid_search = GridSearchCV(mlp, params, scoring="neg_root_mean_squared_error", verbose=1, n_jobs=-1, cv=3)

	exp = Experiment(mlp, train_x, train_data["vs"], test_x, test_data["vs"],
					 exp_name=EXP_NAME, model_name="MLP", grid_search=grid_search)

	if PCA_b: exp.pca_transform()
	exp.run_grid_search_exp(MLPRegressor,refit=False)

	print()
	print( "  * MLP GridSearch -> RMSE: " + str(exp.grid_search.best_score_))
	print(exp.metrics)

# --- --- --- --- 

def mlp_simp_exp():
	# # Choose regressor
	print("\n\n @ Running Experiment with simple MLP")
	mlp = MLPRegressor(learning_rate='adaptive', early_stopping=True, max_iter=1000)

	exp = Experiment(mlp, train_x, train_data["vs"], test_x, test_data["vs"],
					 exp_name=EXP_NAME, model_name="MLP_simp")

	exp.run_exp()

	print()
	print( "  * MLP simple -> RMSE: " + str(exp.metrics["RMSE"]))
	print(exp.metrics)

################################################################################
# SVR Experiment
################################################################################

def svr_exp():
	# Choose regressor
	print("\n\n @ Running Experiment with SVR with GridSearch")
	svr = SVR()

	params = {"kernel": ['rbf', 'sigmoid'], "gamma" : ['scale', 'auto'],
			  "C": np.linspace(1e-4,1,10), "epsilon": np.linspace(1e-2,.25,10)} #np.logspace(-4,0, 10)

	# Create GridSearch to optimze parameters
	grid_search = GridSearchCV(svr, params, scoring="neg_root_mean_squared_error", verbose=1, n_jobs=-1, cv=3)

	exp = Experiment(svr, train_x, train_data["vs"], test_x, test_data["vs"],
					 exp_name=EXP_NAME, model_name="SVR", grid_search=grid_search)

	if PCA_b: exp.pca_transform()
	exp.run_grid_search_exp(SVR, refit=False)

	print()
	print( "  * SVR GridSearch -> RMSE: " + str(exp.grid_search.best_score_))
	print(exp.metrics)

# --- --- --- --- 

def svr_simp_exp():
	# Choose regressor
	print("\n\n @ Running Experiment with SVR")
	svr = SVR()

	exp = Experiment(svr, train_x, train_data["vs"], test_x, test_data["vs"],
					 exp_name=EXP_NAME, model_name="SVR_simp")

	exp.run_exp()

	print()
	print( "  * SVR simple -> RMSE: " + str(exp.metrics["RMSE"]))
	print(exp.metrics)

################################################################################
# Linear Regresion Experiment
################################################################################

def lm_exp():
	# # Choose regressor
	print("\n\n @ Running Experiment with Linear Regresion")
	lm = LinearRegression()

	exp = Experiment(lm, train_x, train_data["vs"], test_x, test_data["vs"], exp_name=EXP_NAME, model_name="LinearReg")

	exp.run_exp()

	print()
	print( "  * LM RMSE: " + str(exp.metrics["RMSE"]))
	print(exp.metrics)

################################################################################
# ElasticNet Experiment
################################################################################

def elnet_exp():
	# # Choose regressor
	print("\n\n @ Running Experiment with ElasticNet")
	el_net = ElasticNet()

	params = {"alpha": np.linspace(0,1,11), "l1_ratio": np.linspace(0,1,11)}

	# Create GridSearch to optimze parameters
	grid_search = GridSearchCV(el_net, params, scoring="neg_root_mean_squared_error", verbose=1, n_jobs=-1)

	# model, train_x, train_y, test_x, test_y, grid_search=None, regression=True, scaler=None)
	exp = Experiment(el_net, train_x, train_data["vs"], test_x, test_data["vs"],
					 exp_name=EXP_NAME, model_name="ElasticNet", grid_search=grid_search)

	if PCA_b: exp.pca_transform()
	exp.run_grid_search_exp(ElasticNet)

	print()
	print( "  * ElNet GridSearch -> RMSE: " + str(exp.grid_search.best_score_))
	print(exp.metrics)

################################################################################
# Gaussian Process Experiment
################################################################################

def gp_exp():
	# Choose regressor
	print("\n\n @ Running Experiment with GP with GridSearch")
	gp = GaussianProcessRegressor()

	params = {"kernel": [DotProduct(), DotProduct()+WhiteKernel(), RBF(), RBF()+WhiteKernel()], "alpha": np.arange(1e-11, 1e-9, 1e-10)}

	# Create GridSearch to optimze parameters
	grid_search = GridSearchCV(gp, params, scoring="neg_root_mean_squared_error", verbose=1, n_jobs=-1, cv=3)

	exp = Experiment(gp, train_x, train_data["vs"], test_x, test_data["vs"],
					 exp_name=EXP_NAME, model_name="GP", grid_search=grid_search)

	if PCA_b: exp.pca_transform()
	exp.run_grid_search_exp(SVR, refit=False)

	print()
	print( "  * GP GridSearch -> RMSE: " + str(exp.grid_search.best_score_))
	print(exp.metrics)

def gp_sim_exp():
	# Choose regressor
	print("\n\n @ Running Experiment with GP")
	gp = GaussianProcessRegressor()

	exp = Experiment(gp, train_x, train_data["vs"], test_x, test_data["vs"],
					 exp_name=EXP_NAME, model_name="GP_simp")

	exp.run_exp()

	print()
	print( "  * GP -> RMSE: " + str(exp.metrics["RMSE"]))
	print(exp.metrics)

################################################################################
# Random Forest Experiment
################################################################################

def rf_exp():
	# Choose regressor
	print("\n\n @ Running Experiment with RF")
	rf = RandomForestRegressor(n_jobs=-1)

	exp = Experiment(rf, train_x, train_data["vs"], test_x, test_data["vs"],
					 exp_name=EXP_NAME, model_name="RF")

	if PCA_b: exp.pca_transform()
	exp.run_exp()

	print()
	print( "  * RF -> RMSE: " + str(exp.metrics["RMSE"]))
	print(exp.metrics)

################################################################################
# Boosting and Bagging Regressor
################################################################################

def adaboost_mlp_exp():
	# Choose regressor
	print("\n\n @ Running Experiment with AdaBoost (MLP)")
	adb = AdaBoostRegressor(MLPRegressor(), n_estimators=10)

	exp = Experiment(adb, train_x, train_data["vs"], test_x, test_data["vs"],
					 exp_name=EXP_NAME, model_name="ADB_MLP")

	exp.run_exp()

	print()
	print( "  * ADB_MLP -> RMSE: " + str(exp.metrics["RMSE"]))
	print(exp.metrics)

def gboost_exp():
	# Choose regressor
	print("\n\n @ Running Experiment with GradientBoosting")
	gb = GradientBoostingRegressor()

	exp = Experiment(gb, train_x, train_data["vs"], test_x, test_data["vs"],
				 exp_name=EXP_NAME, model_name="GB")

	exp.run_exp()

	print()
	print( "  * GB -> RMSE: " + str(exp.metrics["RMSE"]))
	print(exp.metrics)

#Otros metodos:
	# BaggingRegressor(MLPRegressor(), n_estimators=15, , n_jobs=-1,...)
	# AdaBoostRegressor(MLPRegressor(), n_estimators=15, )


################################################################################
# Extreme Learning Machine Experiment
################################################################################

# ELM -> https://pypi.org/project/hpelm/
#	Documentation: https://hpelm.readthedocs.io/en/latest/index.html

################################################################################
# Generic Experiment
################################################################################

# # Choose regressor
# print("\n\n @ Running Experiment with ?")
# mlp = MLPRegressor(early_stopping=True, max_iter=1000)

# #Hacer un grid search con un MLP
# # params = [{"solver": ["sgd"], "hidden_layer_sizes": [[25],[50],[75],[25,25],[25,50],[50,50], [10,25,10]],
# # 		  "activation": ['logistic', 'tanh'],
# # 		  "momentum":[.9,.85,.7], "learning_rate_init": [0.001, 0.0025, 0.005]},
# # 		  {"solver": ["adam"], "hidden_layer_sizes": [[25],[50],[75],[25,25],[25,50],[50,50], [10,25,10]],
# # 		  "activation": ['logistic', 'tanh'], "learning_rate_init": [0.001, 0.0025, 0.005]}]


# # Create GridSearch to optimze parameters
# grid_search = GridSearchCV(mlp, params, scoring="neg_root_mean_squared_error", verbose=1, n_jobs=-1)

# # model, train_x, train_y, test_x, test_y, grid_search=None, regression=True, scaler=None)
# exp = Experiment(mlp, train_x, train_data["vs"], test_x, test_data["vs"],
# 				 exp_name=EXP_NAME, model_name="MLP", grid_search=grid_search)

# exp.run_grid_search_exp(MLPRegressor)

# print()
# print( "  * GridSearch -> RMSE: " + str(exp.grid_search.best_score_))
# print(exp.metrics)

# ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ----

# Lanzar primero los experimentos más simples
lm_exp()
elnet_exp()
mlp_simp_exp()
svr_simp_exp()
gp_sim_exp()
rf_exp()
adaboost_mlp_exp()
gboost_exp()


# #Exoerimentos con GridSearch
mlp_exp()
svr_exp()
#gp_exp() # Tarda demasiado !!

import random
import os, sys
import getpass

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

#Data preprocesing
from sklearn.preprocessing import scale, StandardScaler, MinMaxScaler, RobustScaler
#Metrics
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, make_scorer

user = getpass.getuser()
os.chdir("/Users/"+user+"/Documents/D/Niebla/")

#Import aux files
sys.path.append('src/aux/')
import aux_functions
from ELM_Experiment import ELM_Experiment

random.seed(1234)
np.random.seed(1234)

# ---------------- ---------------- ---------------- ---------------- ---------------- ---------------- ----------------
# Main

print(" # Loading data from csv files")
train_df = pd.read_csv("./Data/csv/temp_split/train_data.csv")
test_df = pd.read_csv("./Data/csv/temp_split/test_data.csv")

n_cols = train_df.shape[1]
PCA_b = False

ncols=train_df.shape[1]

train_y, test_y = train_df["vs"].values, test_df["vs"].values

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
	EXP_NAME = "temp_split"
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
	train_x = scaler.fit_transform(train_df.iloc[:,:(n_cols-1)])
	test_x  = scaler.transform(test_df.iloc[:,:(n_cols-1)])
	#test_x  = scaler.fit_transform(test_df.iloc[:,:(n_cols-1)])
else:
	train_x = train_df.iloc[:,:(n_cols-1)]
	test_x  = test_df.iloc[:,:(n_cols-1)]


print()
print(" -------------------------------------------")
print()
print("   Running experiment -> " + EXP_NAME)
print()
print(" -------------------------------------------")
print()


elm = ELM_Experiment(train_x, train_y, test_x, test_y, ncols=ncols, model_name="ELM_GS", exp_name=EXP_NAME)

elm.run_grid_search_exp()
#TODO: No se está guardando la mejor configuración
elm.save_logs()
import os, sys, shutil
import getpass
import bz2, pickle # Rick!
import copy
import math

import pandas as pd
import numpy as np
from scipy.io import loadmat

user = getpass.getuser()
os.chdir("/Users/"+user+"/Documents/D/Niebla/")

#Import aux files
sys.path.append('src/aux/')
import aux_functions

COMP_PATH="./log_files/alg_comp/"
RESULTS_PATH="./log_files/results/"

os.makedirs(RESULTS_PATH, exist_ok=True)

if not os.path.isdir(COMP_PATH):
	print(" # Can't find metrics folder!")
	sys.exit()

#####################################################################
# Sorting all metrics from all epxeriments
#####################################################################
print(" # Sorting all metrics from all epxeriments")

get_exp_name = lambda x: '-'.join(x[:-4].split("-")[1:])

files = []

print("  # Reading metrics files")
for f in os.listdir(COMP_PATH):
	if f.endswith(".csv"):
		exp_name = get_exp_name(f)
		print("  * Processing " + f)
		df = pd.read_csv(COMP_PATH+f, index_col="Metric")
		files.append(df)

#Dataframes concatenated by Metrics
df = pd.concat(files,axis=1)

df_sorted = df.T.sort_values("RMSE")

#Save all models rated by RMSE
print("  # Saving all metrics from all models sorted by RMSE")
df_sorted.to_csv(RESULTS_PATH+"alg_comp.csv")

#####################################################################
# Collecting best models' plots
#####################################################################
print()
print(" # Collecting best models' plots")

#Getting k best models
k=5
df_k_best = df_sorted.iloc[:k,:]

os.makedirs("./log_files/results/plots/", exist_ok=True)

get_plot_fnm = lambda x: "./plots/"+x["exp"]+"/"+x["model"]+".eps"
get_dest_fnm = lambda x,y: "./log_files/results/plots/"+str(y)+"_"+x["exp"]+"-"+x["model"]+".eps"

print("  * Copying best models plots to results folder: \n   * ", end="")
for idx, val in enumerate(df_k_best.index):
	ret = aux_functions.get_exp_n_model_name(val)
	fn_ = get_plot_fnm(ret)
	if os.path.isfile(fn_):
		print(ret["exp"]+"-"+ret["model"], end=", ")
		shutil.copy(fn_, get_dest_fnm(ret,idx+1))
	else:
		print("  -> Eror: file not found! " + fn_ + "<- ", end="")
print()
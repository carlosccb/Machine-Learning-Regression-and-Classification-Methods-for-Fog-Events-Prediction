import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

import os, sys
import getpass
user = getpass.getuser()
#TODO: Esto hay que cambiarlo para cada proyecto
os.chdir("/Users/"+user+"/Documents/D/Niebla/")

sys.path.append('src/aux/')
import aux_functions

sys.path.append('src/plot/')
import plot

############# ############# ############# ############# ############# ############# 

PREDS_PATH="./log_files/predictions/"

get_exp_name = lambda x: '-'.join(x[:-4].split("-")[1:])

for f in os.listdir(PREDS_PATH):
	if f.endswith(".csv"):
		exp_name = get_exp_name(f)
		print(" # Loading data of " + exp_name + " exp")
		os.makedirs("./plots/"+exp_name, exist_ok=True)

		#Read data
		df = pd.read_csv(PREDS_PATH+f)
		y_true = df["y_true"]

		#Get models used (remove y_true from columns)
		models_used = df.columns[1:]

		print("  * Plotting data")
		for m in models_used:
			ret = aux_functions.get_exp_n_model_name(m)
			print("   · " + ret["exp"] + " - " + ret["model"] )
			plot.plot_preds(y_true, df[m], ret["model"], exp_name, save_plot=True)
			#plot.plot_preds_doc_sancho(y_true, df[m], ret["model"], ret["exp"], save_plot=True)
		print()


# print(" # Plotting data for Sancho's Doc")

# #Para el documento de sancho
# # De 0-300 y de 300-600 va bastante bien el MLP_simp
# print(" test samples 0-300")
# m = models_used[2]
# ret = aux_functions.get_exp_n_model_name(m)
# y_true_ = y_true[0:300]
# y_pred_ = df[m][0:300]
# plot.plot_preds_doc_sancho(y_true_, y_pred_, ret["model"]+"-[0-300]", ret["exp"], save_plot=True)

# # m = models_used[2]
# # ret = aux_functions.get_exp_n_model_name(m)
# print(" test samples 300-600")
# y_true_ = y_true[300:600]
# y_pred_ = df[m][300:600]
# plot.plot_preds_doc_sancho(y_true_, y_pred_, ret["model"]+"-[300-600]", ret["exp"], save_plot=True)
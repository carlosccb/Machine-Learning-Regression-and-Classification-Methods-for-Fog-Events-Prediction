import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

import os, sys
import getpass
user = getpass.getuser()

#TODO: Esto hay que cambiarlo para cada proyecto
os.chdir("/Users/"+user+"/Documents/D/Niebla/")


def plot_preds(y_true, y_pred, model_name="", exp_name="", save_plot=False):
	"""Function to plot y_true vs y_pred
		- y_true: ground truth values, 
		- y_pred: predicted values
		- model_name: model used to predict
		- exp_name: experiment name
		- save_plot: Whether to save the plot or no
	"""
	x=np.arange(y_true.shape[0])
	sns.lineplot(x=x, y=y_true, color="green")
	sns.lineplot(x=x, y=y_pred, color="blue")

	if save_plot:
		plt.savefig("plots/"+exp_name+"/"+model_name+".eps")
	plt.close()

def plot_preds_doc_sancho(y_true, y_pred, model_name="", exp_name="", save_plot=False):
	"""Function to plot y_true vs y_pred
		- y_true: ground truth values, 
		- y_pred: predicted values
		- model_name: model used to predict
		- exp_name: experiment name
		- save_plot: Whether to save the plot or no
	"""
	x=np.arange(y_true.shape[0])

	sns.scatterplot(x=x, y=y_true, color="blue")
	sns.lineplot(x=x, y=y_true, color="blue")

	sns.scatterplot(x=x, y=y_pred, color="red", marker="+")
	sns.lineplot(x=x, y=y_pred, color="red")

	if save_plot:
		plt.savefig("plots/"+exp_name+"/"+model_name+"-doc_sancho"+".eps")
	plt.close()

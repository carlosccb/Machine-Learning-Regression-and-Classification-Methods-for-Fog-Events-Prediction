import numpy as np
import pandas as pd

from sklearn.metrics import cohen_kappa_score, accuracy_score, \
							multilabel_confusion_matrix, precision_recall_fscore_support#, top_k_accuracy_score

import os, sys
from fnmatch import fnmatch

mm_metrics=["precision", "recall", "fbeta_score", "support"]

# Path operations
os.chdir("/home/carlos/Documents/D/Niebla/log_files/alg_comp/classification/ordinal")
PREDS_PATH = "/home/carlos/Documents/D/Niebla/log_files/predictions/classification/ordinal/"
PROBS_PATH =  PREDS_PATH + "predicted_probabilities/"
DATA_PATH =  "/home/carlos/Documents/D/Niebla/Data/Treated_Data/Classification/HoldOut/"

# Aux functions
get_dataset_nm = lambda x: x.split("-")[2]
get_lambda_val = lambda x: x.split("-")[3].split("=")[1]

# Aux variables
results_df = pd.DataFrame()

# Load data
test_data = pd.read_csv(DATA_PATH+"Test.csv")
y_ts = test_data["class"]

# Dictionary to create the pd.DF with
dict_df = {}

for f in os.listdir(PREDS_PATH):
	if fnmatch(f, "GLM_elnet-Ordinal-*"):
		# Get info descripting the dataset to load
		if (nm := get_dataset_nm(f)[1:]) == "":
			nm = "OG"
		info = {"Dataset":nm, "alpha":get_lambda_val(f)}

		print(info)

		# Load the dataset
		y_pred = pd.read_csv(PREDS_PATH+f, squeeze=True)
		y_pred -= 1
		info["acc"] = accuracy_score(y_ts, y_pred)
		info["qwk"] = cohen_kappa_score(y_ts, y_pred, weights="quadratic")

		#  Get all data from preds on a dict with the corresponding class
		# to easily create the pd.DF
		mm = precision_recall_fscore_support(y_ts, y_pred, zero_division=0)
		# TODO: se podría incluir aquí también las métricas medias de la función anterior
		for i in range(5):
			info[f"precision_{i}"] = mm[0][i]
			info[f"recall_{i}"] = mm[1][i]
			info[f"fbeta_score_{i}"] = mm[2][i]

		# Store all information in the dict
		dict_df[f'{info["Dataset"]}-{info["alpha"]}'] = info

pd.DataFrame(dict_df).T.to_csv("Complete_by_class_Ordinal_comparison.csv")

# -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
#  TODO: Se podría hacer una seleción de para cada configuración de experimento escoger sólo
# un valor de alpha: el valor que mejor qwk (o lo que sea) tenga
import numpy as np
import pandas as pd

from sklearn.metrics import cohen_kappa_score, accuracy_score, \
							multilabel_confusion_matrix, precision_recall_fscore_support#, top_k_accuracy_score

import os, sys
from fnmatch import fnmatch
import string

############################################################################################################
#
# Warning: For this script to work all the pred_probs files must have been stored in a separate folder 
#
############################################################################################################

mm_metrics=["precision", "recall", "fbeta_score", "support"]

# Path operations
os.chdir("/home/carlos/Documents/D/Niebla/log_files/alg_comp/classification/nominal")

PREDS_PATH = "/home/carlos/Documents/D/Niebla/log_files/predictions/classification/nominal/"
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
glm_dict_df = {}

# Short names for classifiers
shrt_nms = {"KNeighborsClassifier": "KNN", "GaussianNB": "GNB",
			"DecisionTreeClassifier": "DT","AdaBoostClassifier": "AB",
			"GradientBoostingClassifier": "GB", "BaggingClassifier": "Bagg",
			"RandomForestClassifier": "RF"}

# Definir función para obtener las métricas
def get_metrics(info, y_pred):
	info["acc"] = accuracy_score(y_ts, y_pred)
	info["qwk"] = cohen_kappa_score(y_ts, y_pred, weights="quadratic")

	mm = precision_recall_fscore_support(y_ts, y_pred, zero_division=0)
	# TODO: se podría incluir aquí también las métricas medias de la función anterior
	for i in range(5):
		info[f"precision_{i}"] = mm[0][i]
		info[f"recall_{i}"] = mm[1][i]
		info[f"fbeta_score_{i}"] = mm[2][i]


# ------------------------------------------------------------------------------------------
# Primero cargar los ficheros GLM_elnet y sacar para cada fichero de train el mejor alpha
# ------------------------------------------------------------------------------------------
def glm_match(f):
	return fnmatch(f, "GLM_elnet-Nominal-*")

# Aquí se saca un dict/pd.DF con todas las métricas para todas las combinaciones de GLM (alpha+dataset)
# TODO: ver qué interesa más sacar, si el mejor para cada dataset, o los _n_ mejores (se repetirían datasets)
#   => Creo que interesa más los _n_ mejores
for f in filter(glm_match, os.listdir(PREDS_PATH)):
	# Get info descripting the dataset to load
	if (nm := get_dataset_nm(f)[1:]) == "":
		nm = "OG"
	info = {"Model": "GLM", "Dataset":nm, "alpha":get_lambda_val(f)}

	print(info)

	# Load the dataset # Hay que añadir header=None porque en este caso el fichero no contiene nombre de columna
	y_pred = pd.read_csv(PREDS_PATH+f, squeeze=True, header=None)
	y_pred -= 1
	get_metrics(info, y_pred)

	# Store all information in the dict
	# #  Get all data from preds on a dict with the corresponding class
	# # to easily create the pd.DF
	glm_dict_df[f'{info["Dataset"]}-{info["alpha"]}'] = info

glm_res_df = pd.DataFrame(glm_dict_df).T

glm_res_df.to_csv("GLM_nominal_complete_comparison.csv")



# ------------------------------------------------------------------------------------------
# Cargar los ficheros de clasif nominal y sacar las métricas
#------------------------------------------------------------------------------------------
def nom_match(f):
	return fnmatch(f, "*pred_class.csv") and not fnmatch(f, "GLM*") and not fnmatch(f, "Persistence*")

for f in filter(nom_match, os.listdir(PREDS_PATH)):
	print(f" Loading file {f}")
	df = pd.read_csv(PREDS_PATH+f)

	mdl_nm = f.split("_")[0]
	mdl_nm_s = shrt_nms[mdl_nm]

	for (col, data) in df.iteritems():
		print(f"  Reading column {col}")
		info = {"Model": mdl_nm, "ShortName": mdl_nm_s, "Dataset": col}
		get_metrics(info, data)

		dict_df[f"{mdl_nm_s}-{col}"] = info

# ------------------------------------------------------------------------------------------
# Cargar el fichero de persistencia y sacar las métricas
#------------------------------------------------------------------------------------------
df = pd.read_csv(PREDS_PATH+"Persistence_pred_class.csv")

info = {"Model": "Persistence", "ShortName": "PSTC", "Dataset": "OG"}
get_metrics(info, df)

# Unir los clasificadores nominales de sklearn con la persistencia
dict_df["PSTC-OG"] = info
nom_res_df = pd.DataFrame(dict_df).T

# ------------------------------------------------------------------------------------------
# Unir las métricas de los ficheros disponibles y crear un solo fichero para hacer la comparación
# Habría que elegir cuántas combinaciones métodos-fichero para poner en la comparación
#  Se podrían utilizar varias estrategias de elección: 
#	* que salga al menos para cada caso un representante de cada método (el mejor de cada uno)
#	  y luego los x mejores de todos
#	* o se podrían hacer dos comparaciones: el mejor de cada, y los mejores de entre todos

# TODO: 
# # Cargar el fichero de ordinal glm y poner junto con estos
# pd.read_csv("")

# Ordenar los valores por qwk de los clasificadores de sklearn
nom_res_df.sort_values(["qwk"], ascending=False)#, inplace=True)
# Ordenar los valores por qwk de los clasificadores de glm
glm_res_df.sort_values(["qwk"], ascending=False)#, inplace=True)


# -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --

# ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ##
# TODO: juntar todas las métricas de los clasificadores en el mismo fichero
# Ahora se tienen los datos de la clasificación nominal en dict_df, queda por juntar
# los elegidos (mejores n o mejor de cada) de GLM en este dataset

# TODO: También se podrían cargar y meter las métricas del GLM ordinal aquí ya que la mayoría de los métodos
# usados aquí no tienen correspondiente ordinal

# Para unirlos sólo hace falta hacer
# pd.concat([df,glm_df])

#pd.DataFrame(dict_df).T.to_csv("Complete_by_class_Nominal_comparison.csv")



# -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
#  TODO: Se podría hacer una seleción de para cada configuración de experimento escoger sólo
# un valor de alpha: el valor que mejor qwk (o lo que sea) tenga
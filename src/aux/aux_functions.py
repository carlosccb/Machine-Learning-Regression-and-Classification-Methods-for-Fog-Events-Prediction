import os, sys, getpass, socket, platform
import bz2
import pickle # Rick!

import numpy as np
import pandas as pd
import math

#Regression metrics
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, make_scorer
#Classification metrics
from sklearn.metrics import accuracy_score, auc, cohen_kappa_score, f1_score, confusion_matrix

# Get variables about where the code is running and do neccesary stuff to run code succesfully
def set_environment_location():
	user, hostname = getpass.getuser(), socket.gethostname()
	if platform.system() == "Darwin":
	    os.chdir("/Users/"+user+"/Documents/D/Niebla/")
	elif platform.system() == "Linux":
		if user+"-DesktopPC" == hostname:
			os.chdir("/home/"+user+"/Documents/D/Niebla/")
		else:
			os.chdir("/home/"+user+"/Documentos/D/Niebla/")
	else:
		print(" ERROR: Platform not supported!")
		sys.exit()

set_environment_location()

"""
check_folder_structure:
	Comprueba si existe la estructura de carpetas
	Carpetas necesarias:
		* PREDICTIONS_FILE: Carpeta con un fichero con las predicciones de los modelos y el y_true
		* BEST_CONFIG: Guardar en ficheros la mejor configuración de los modelos
		* PLOTS_FOLDER: Carpeta para guardar los plots
		* ALG_COMP: Comparación de métricas para todos los algoritmos utilizados
"""
PREDICTIONS_FOLDER = "log_files/predictions/"
BEST_CONFIG      = "log_files/best_model_config/"
PLOTS_FOLDER     = "plots/"
ALG_COMP         = "log_files/alg_comp/"
DURATION_FOLDER  = "log_files/time_duration/"

ALG_COMP_FILE = ALG_COMP + "alg_comp.csv"
PREDICTIONS_FILE = PREDICTIONS_FOLDER+"preds.csv"

def check_folder_structure(exp_name, data_prep, model_name):
	print(" check_folder_structure ->  exp_name: " + str(exp_name))
	#Crear carpeta para guardar las predicciones de cada modelo (en un mismo fichero)
	os.makedirs(PREDICTIONS_FOLDER+"/"+exp_name+"/"+data_prep+"/", exist_ok=True)
	#Carpeta para guardar los ficheros con las mejores configuraciones de cada modelo
	os.makedirs(BEST_CONFIG+"/"+exp_name+"/"+data_prep+"/", exist_ok=True)
	#Crear carpeta para guardar las gráficas de los modelos
	# TODO WARNING: esto puede ser que provoque errores, se ha añadido el data_prep sin pensar si corresponde
	os.makedirs(PLOTS_FOLDER+"/"+exp_name+"/"+data_prep+"/", exist_ok=True)
	#Carpeta para almacenar la comparación de algoritmos (en un mismo fichero)
	os.makedirs(ALG_COMP+"/"+exp_name+"/", exist_ok=True)
	#Carpeta para almacenar los tiempos de duración de cada experimento
	os.makedirs(DURATION_FOLDER+"/"+exp_name+"/", exist_ok=True)

############################################
# Funciones para guardar información en ficheros

# TODO: From here on

#Save and append metrics to file
def save_model_metrics(metrics):
	#Cargar fichero
	df = pd.read_csv(ALG_COMP_FILE)
	#Añadir metricas al fichero


def save_predictions(y_true, y_pred, model_name):
	#Comprobar si ya existe el fichero
	if not os.path.isfile(PREDICTIONS_FILE):
		#Crear el fichero
		#Guardar los datos de y_true e y_pred
		pass
	else:
		#Abrir el fichero
		df = pd.read_csv(PREDICTIONS_FILE)

		#Comprobar si el fichero ya tiene los valores de y_true
		if "y_true" not in df:
			#Guardar los valores de y_true e y_pred en el fichero
			df["y_true"] = y_true
			pass
		#Guardar los valores de y_pred en el fichero
		df[model_name] = y_pred
		pass

def save_best_config():
	pass

#Save metrics to text file
def save_metrics(y_true, y_pred, clasif_name, task="regression", verbose=False):
	"""
		 I think this function is deprecated; the same functionality is included
		in Experiment.save_logs
	"""
	if task == "regression":
		print(" # Getting scores on test data")
		r2 = r2_score(y_true, y_pred)
		if verbose: print("  * R2: " + str(r2))
		MSE = mean_squared_error(y_true, y_pred)
		if verbose: print("  * MSE: " + str(MSE))
		RMSE = math.sqrt(MSE)
		if verbose: print("  * RMSE: " + str(RMSE))
		MAE = mean_absolute_error(y_true, y_pred)
		if verbose: print("  * MAE: " + str(MAE))

		with open(ALG_COMP_FILE, 'a') as f:
			pd.DataFrame({clasif_name: [r2, RMSE, MAE]}).T.to_csv(f, header=False)

	elif task == "classification":
		#TODO
		pass

	elif task == "ordinal_classification":
		#TODO
		pass

	else:
		print("\n # Error: task (" + task + ") not found!\n")

#
############################################



#Get metrics and print
def get_metrics(y_true, y_pred, task="regression", verbose=False):
	"""
		 I think this function is deprecated; the same functionality is included
		in Experiment.get_metrics
	"""
	if task == "regression":
		r2 = r2_score(y_true, y_pred)
		if verbose: print("  * R2: " + str(r2))
		MSE = mean_squared_error(y_true, y_pred)
		if verbose: print("  * MSE: " + str(MSE))
		RMSE = math.sqrt(MSE)
		if verbose: print("  * RMSE: " + str(RMSE))
		MAE = mean_absolute_error(y_true, y_pred)
		if verbose: print("  * MAE: " + str(MAE))

		return {"R2": r2, "MSE": MSE,
				"RMSE": RMSE, "MAE": MAE}
	elif task == "classification":
		#TODO
		pass

	elif task == "ordinal_classification":
		#TODO
		pass

	else:
		print("\n # Error: task (" + task + ") not found!\n")


############################################
# Funciones auxiliares

#Trim predictions
def trim_preds(y_pred, upper_limit, lower_limit=0):
	"""Trim the values of the predictions so that none are above a given value
	or below another given value
	"""
	y_pred[y_pred<lower_limit]=lower_limit
	y_pred[y_pred>upper_limit]=upper_limit
	return y_pred

def binarize_rvr(y_true, thres=1950):
	"""Binarize values of visibility with a given threshold
	"""
	y_bin = (y_true < thres)
	return y_bin.astype(int)

def get_exp_n_model_name(name):
	"""Split a string containing Model-experiment_whatev into Model and experiment_whatev
	"""
	ret = name.split("-")
	return {"model": ret[0], "exp": '-'.join(ret[1:])}

#
############################################

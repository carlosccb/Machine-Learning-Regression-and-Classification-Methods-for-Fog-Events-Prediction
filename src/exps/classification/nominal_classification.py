import os, sys
import glob

import datetime
import time

import pandas as pd
import numpy as np

from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier, \
 GradientBoostingClassifier, BaggingClassifier, RandomForestClassifier

from sklearn.preprocessing import MinMaxScaler

from sklearn.metrics import accuracy_score

# -----------------
# Variables
# -----------------

ROOT_PATH = "~/Documents/D/" # Para ejecutar en mis PC
# ROOT_PATH = "/home/ccastillo/" # Para lanzar en el cluster

DATA_PATH = ROOT_PATH+"Niebla/Data/Treated_Data/Classification/HoldOut/"
LOG_PATH  = ROOT_PATH+"Niebla/log_files/alg_comp/classification/nominal/"
PRED_PATH = ROOT_PATH+"Niebla/log_files/predictions/classification/nominal/"

np.random.seed(0)

clasifiers = [KNeighborsClassifier, GaussianNB, DecisionTreeClassifier, AdaBoostClassifier,
 			  GradientBoostingClassifier, BaggingClassifier, RandomForestClassifier]

# ------------------------------------------------------------------------------------------------------------------------
# Cargar los datos, estimar valores para escalado y transformar datos originales
# ------------------------------------------------------------------------------------------------------------------------

test_df = pd.read_csv(DATA_PATH+"Test.csv")
y_ts = test_df["class"]

print(" # Loading train data and setting scaling method")
# Se cargan los datos de Train para calcular los valores que se usarán para el estandarizado
train_df = pd.read_csv(DATA_PATH+"Train.csv")
scl = MinMaxScaler()
scl.fit(train_df.drop("class",axis=1))

print("  * Transforming data")
# Otra opción sería hacer esta transformación en el bucle con los valores para cada uno de los 
# ficheros de train disponibles (aunque esto sería más relevante con un normalizado estadístico)
x_tr = scl.transform(train_df.drop("class",axis=1))
x_ts = scl.transform(test_df.drop( "class",axis=1))

# Crear diccionarios y almacenar los datos originales transformados
# dict para todos los ficheros pasa la X
tr_x_dataset_df = {}
tr_x_dataset_df["OG"] = x_tr
# dict para todos los ficheros pasa la y
tr_y_dataset_df = {}
tr_y_dataset_df["OG"] = train_df["class"]

# ------------------------------------------------------------------------------------------------------------------------
# Cargar todos los conjuntos de datos disponibles y meterlos en un dict
#
# Para hacer esto más eficiente a la hora de ejecutar se deberían primero cargar y guardar todos los ficheros en un dict
# e ir iterando con los clasificadores sobre ellos
# ------------------------------------------------------------------------------------------------------------------------
print(" # Loading all datasets")
# Load file and normalize values
for file in glob.glob(os.path.expanduser(DATA_PATH)+"Train*"):
	# Get dataset name
	dataset = file.split("/")[-1][6:-4]
	dataset = dataset if dataset != "" else "OG"
	print(f"   · Loading and processing dataset {dataset}")

	# Load dataset
	df = pd.read_csv(file)
	# Transform dataset
	df_data = scl.transform(df.drop("class",axis=1))
	# Store dataset
	tr_x_dataset_df[dataset] = df_data
	tr_y_dataset_df[dataset] = df["class"]


# -----------------------
# Hacer las predicciones
# -----------------------
preds_df = {}
#preds_df["y"] = y_ts.values
time_df = {}
print(" # Running training loop")
for i in clasifiers:
	clsfr = i()
	clsfr_nm = clsfr.__class__.__name__
	clsfr_dict = {}

	time_df[clsfr_nm] = {}

	print(f"  @ Using clasifier {clsfr_nm}")

	for j in tr_x_dataset_df.keys():
		print(f"   * Using dataset {j}")

		# Train the model
		start=datetime.datetime.now()
		clsfr.fit(tr_x_dataset_df[j], tr_y_dataset_df[j])
		end=datetime.datetime.now()
		# Predict test data with trained model
		y_pred = clsfr.predict(x_ts)

		# Put preds for current dataset in a dict
		clsfr_dict[j] = y_pred

		# Get fit time
		time_df[clsfr_nm][j] = str(end - start)
		#print(f" =>  fit time: {time_df[clsfr_nm][j]}")

	print(f"   Saving predictions for model {clsfr_nm}")
	# Save all preds for current clasifiers
	pd.DataFrame(clsfr_dict).to_csv(f"{PRED_PATH}{clsfr_nm}_pred_class.csv", index=False)

print(" # Saving fit times")
# Save fit times
pd.DataFrame(time_df).to_csv(f"{LOG_PATH}nominal_elapsed_times.csv")

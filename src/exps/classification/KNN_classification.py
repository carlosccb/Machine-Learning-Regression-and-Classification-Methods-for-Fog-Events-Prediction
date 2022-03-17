import os, sys
import glob

import datetime

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

# -----------------
# Cargar los datos
# -----------------

test_df = pd.read_csv(DATA_PATH+"Test.csv")
y_ts = test_df["class"]

print(" # Loading train data and setting scaling method")
# Se cargan los datos de Train para calcular los valores que se usarán para el estandarizado
tr_df = pd.read_csv(DATA_PATH+"Train.csv")
scl = MinMaxScaler()
scl.fit(tr_df.drop("class",axis=1))

print("  * Transforming test data")
# Otra opción sería hacer esta transformación en el bucle con los valores para cada uno de los 
# ficheros de train disponibles (aunque esto sería más relevante con un normalizado estadístico)
x_ts = scl.transform(test_df.drop("class",axis=1))

#  TODO: Para hacer esto más eficiente a la hora de ejecutar se deberían primero cargar y guardar todos los ficheros en un dict
# e ir iterando con los clasificadores sobre ellos

# -----------------------
# Hacer las predicciones
# -----------------------
preds_df = {}
preds_df["y"] = y_ts.values
time_df = {}
print(" # Running training loop")
# for clsf in clasifiers:
for file in glob.glob(os.path.expanduser(DATA_PATH)+"Train*"):
	#print(file)
	dataset = file.split("/")[-1][6:-4]
	print(f"  * Working with dataset {dataset}")

	# Load file and normalize values
	print("   · Loading and processing")
	train_df = pd.read_csv(file)
	x_tr = scl.transform(train_df.drop("class",axis=1))
	y_tr = train_df["class"]

	# Classify with KNN
	knn = KNeighborsClassifier()
	print("   @ Training")
	start=datetime.datetime.now()
	knn.fit(x_tr, y_tr)
	end=datetime.datetime.now()
	print("   @ Predicting")
	y_pred = knn.predict(x_ts)
	# Save predictions
	preds_df[dataset] = y_pred
	# Get running times and store'em
	time_df[dataset] = str(end - start)

	# TODO: Si se fueran a añadir más clasificadores habría que cambiar la estructura de este bucle
	# también habría que añadir el modelo que se está utilizando. También habría que guardar los csv
	# -> to get the name of the object ClassName().__class__.__name__
	# # Classify with NaiveBayes
	# naiveB = GaussianNB()
	# naiveB.fit(x_tr, y_tr)
	# y_pred = naiveB.predict(x_ts)

print(" # Saving predictions and times to a file on disk")
pd.DataFrame(preds_df).to_csv(PRED_PATH+"KNN_pred_class.csv", index=False)

# Da un fallo y tampoco es tan importante:
#pd.DataFrame(time_df)
import os
import sys
import getpass
import bz2
import pickle # Rick!

import pandas as pd
import numpy as np
from scipy.io import loadmat

user = getpass.getuser()
os.chdir("/Users/"+user+"/Documents/D/Niebla/")

sys.path.append('src/aux/')
import aux_functions

##########################################
print(" @ Loading matlab files")

#Test_A8_Todas_h: datos test horarios de las 11 variables cada hora
ts_data_dict = loadmat("./Data/Matlab/Test_A8_Todas_h.mat")
#Train_A8_Todas_h: datos train horarios de las 11 variables cada hora
tr_data_dict = loadmat("./Data/Matlab/Train_A8_Todas_h.mat")

#Visib_A8_Todas: datos con las 11 variables en continua sin NaN y 2018-2019 seguidos cada 5 min
data_5m = loadmat("./Data/Matlab/Visib_A8_Todas.mat")
#Visib_A8_Todas_h: datos con las 11 variables en continua sin NaN y 2018-2019 seguidos cada hora
data_h = loadmat("./Data/Matlab/Visib_A8_Todas_h.mat")

##########################################################################################
#
##########################################################################################

print()
print(" # Dataset con variables de regresión: train y test")

#Sacar los datos de train y test
tr_data = tr_data_dict["Train"]
ts_data = ts_data_dict["Test"]

##########################################
print(" # Adapting content for new files")
print("  * Storing variables in dictionary")
print("  * Binarizing values of dependent variable")
print("  * Storing values in dictionary")

#Create dictionary to store data
dataset_dict = {"Train": {},
				"Test": {}}

cols_x = [0,1,3,4,5,6,7,8,9,10]

# TODO: Aquí la binarización se hace con el valor 1950!

#Sacar los fold, poner con instancias en filas y meter folds en dicts
# Hacer "lag" en la variable a predecir (que el primer X prediga el segundo y)
for i in range(tr_data.shape[1]):
	dataset_dict["Train"][i] = {"X": [], "y": []}
	dataset_dict["Train"][i]["X"] = tr_data[0,i].T[:-1,cols_x]
	dataset_dict["Train"][i]["y"] = tr_data[0,i].T[1:,2]
	dataset_dict["Train"][i]["y_bin"] = aux_functions.binarize_rvr(dataset_dict["Train"][i]["y"])

	dataset_dict["Test"][i] = {"X": [], "y": []}
	dataset_dict["Test"][i]["X"]  = ts_data[0,i].T[:-1,cols_x]
	dataset_dict["Test"][i]["y"]  = ts_data[0,i].T[1:,2]
	dataset_dict["Test"][i]["y_bin"] = aux_functions.binarize_rvr(dataset_dict["Test"][i]["y"])

print(" # Saving pickled dataset to file")

os.makedirs("./Data/Pickle/", exist_ok=True)

#Utilizar el protocolo pickle.DEFAULT_PROTOCOL (4) para que no ocupe tanto el fichero
#Guardar con compresión bz2 para que ocupe menos
with bz2.BZ2File("./Data/Pickle/Dataset.pbz2", "wb") as f:
	pickle.dump(dataset_dict, f, protocol=pickle.DEFAULT_PROTOCOL)


##########################################################################################
#
##########################################################################################

print()
print(" # Dataset con variables de ventana hasta 3h")

# File Struct:
#  dB_ventana3  = exgonenas(t-2)|exgonenas(t-1)|exgonenas| Minimo |Duracion
# Las exogenas a su vez (como te dije en los txt) van así:
#  qp_ra 	sa 	vs 	hr 	at 	st 	td 	gr 	ws 	wd 	ap

#Variables para crear cabecera
header=[]
var=["qp_ra","sa","vs","hr","at","st","td","gr","ws","wd","ap"]

#Indicar temporalidad en el nombre de la variable
for i in range(2,-1,-1):
	header.append([x+"_t-"+str(i) for x in var])
#Añadir las dos últimas variables, que son a predecir
header.append(["min", "dur"])
#Dejar en una sola lista para poder utilizar como cabecera
cols = [y for x in header for y in x]

os.makedirs("./Data/csv/dB_eventos/", exist_ok=True)

print('  * Guardando datasets con todas las "divisiones" temporales de dB_ventana3')
#Para todos las "divisiones" temporales que hay, pasar el fichero a .csv
for i in [5,15,30,45,60,75,90,120]:
	print("   * Procesando fichero {}m".format(i))
	#Cargar los datos del .mat
	dB_ventana3_df = loadmat("./Data/Matlab/dB_eventos/dB_ventana3_"+str(i)+".mat")
	#Guardar los datos con las columnas en un DataFrame
	db3_df = pd.DataFrame(dB_ventana3_df["dB"], columns=cols)
	#Guardar el DF en csv
	db3_df.to_csv("./Data/csv/dB_eventos/dB_ventana3_"+str(i)+"m.csv", index=False)

##########################################################################################
#
##########################################################################################

print()
print(" # Dataset con división temporal en train(80) y test(20)")

#Cargar los datos del fichero
df = loadmat("./Data/Matlab/Visib_A8_Todas_h.mat")
#Sacar los datos
data = df["Visib_A8_Todas_h"]

rows = data.shape[0]
train_idx = int(rows*.8)

col_names =["qp_ra","sa","hr","at","st","td","gr","ws","wd","ap","vs"]

print("  # Extracting data")
#Sacar datos de train
train_X = data[:train_idx,cols_x]
train_y = data[:train_idx,2].reshape(-1,1)
#Sacar datos de test
test_X  = data[train_idx:,cols_x]
test_y  = data[train_idx:,2].reshape(-1,1)

#Crear los dos datasets
train_data = pd.DataFrame(np.hstack((train_X, train_y)), columns=col_names)
test_data  = pd.DataFrame(np.hstack((test_X, test_y)), columns=col_names)

os.makedirs("./Data/csv/temp_split/", exist_ok=True)

print("  # Saving data")
train_data.to_csv("./Data/csv/temp_split/train_data.csv", index=False)
test_data.to_csv("./Data/csv/temp_split/test_data.csv", index=False)

##########################################################################################
#
##########################################################################################

print()
print(" # Dataset con división temporal y regresivo (x_n = (x_n,y_{n-1})  )")

train_data = pd.read_csv("./Data/csv/temp_split/train_data.csv")
test_data  = pd.read_csv("./Data/csv/temp_split/test_data.csv")

#Meter variable vs-1 en train
tr_df = train_data.drop("vs",1).iloc[1:,:]
#Meter el valor de niebla anterior
tr_df["vs-1"] = train_data["vs"].iloc[:-1].values
#Meter variable a predecir
tr_df["vs"] = train_data["vs"].iloc[1:].values

#Meter todas las columnas menos la de visibilidad
ts_df = test_data.drop("vs",1)
#Meter el valor de niebla anterior
ts_df["vs-1"] = np.zeros(ts_df.shape[0])
#Meter último valor de y correspondiente a train
ts_df["vs-1"].iloc[0]  = train_data["vs"].iloc[-1]
#Meter primero al penúltimo valor de y correspondiente a test
ts_df["vs-1"].iloc[1:] = test_data["vs"].values[:-1]
#Mete valores a predecir
ts_df["vs"] = test_data["vs"].values

print("  # Saving data")
tr_df.to_csv("./Data/csv/temp_split/train_data-regr.csv", index=False)
ts_df.to_csv("./Data/csv/temp_split/test_data-regr.csv", index=False)

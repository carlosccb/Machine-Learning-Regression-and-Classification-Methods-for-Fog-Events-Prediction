#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  8 14:01:01 2020

@author: carloscb
"""

import os
import sys
import re
import platform
import getpass
import bz2
import pickle # Rick! 

import pandas as pd
import numpy as np
from scipy.io import loadmat
import seaborn as sns

from sklearn.model_selection import StratifiedKFold, train_test_split

from imblearn.under_sampling import CondensedNearestNeighbour, NeighbourhoodCleaningRule, TomekLinks
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling  import SMOTE

#Set seeds
import random
random.seed(0)
np.random.seed(0)

"""
    TODO:
        * Se podría probar a entrenar con un % de clase 4 similar a la mayor de las minoritarias y dejar el resto de instancias de
        la clase 4 para test
        * Al calcular las métriccas de clasificación, esta función de imblearn creo que lo hace más facil: 
        imblearn.metrics.classification_report_imbalanced
        * Otra idea sería hacer SMOTE sobre los datos originales y después limpiar fronteras entre clases con TomekLinks


        * Métodos clasificación:
        * Nominal: MLP, SVM (SVM1Vs1, SVM1VsAll)
        * Ordinal (ORCA): NNPOM, REDVSVM, SVOREX

    DONE:
        * Se podría dejar la visibilidad de los 30m anteriores vs_{t-1} en x_{t-1} para predecir y_t.
        Habría que pensar si dejarlo como valor de visibilidad o como la clase a la que pertenece
        * Habría que hacer las particiones de train/test en ¿este u otro fichero? Al haber clases con pocos datos,
        lógicamente, tiene que ser particionamiento estratificado
        * Al separar en train y test, no puede haber elementos generados artificialmente (SMOTE) en test, por lo que
        el proceso de generación de prototipos tiene que ser sólo para train. Duda: Al haber pocos datos de las clases
        minoritarias, ¿se podrían generar los prototipos con todos los datos de esas clases (v.g. incluir los que estarán en test)?

    NVM:
        * Si me pasan los datos que tenían valores perdidos, se podría hacer aprendizaje semi-supervisado para darle
        el valor de clase en vez de imputar valores
        * Hay algo raro al ver las distribuciones de las variables para cada clase. En la clase 0 los valores 
        mínimos de algunas variables son mayores a las clases 1 y 2. => Tiene sentido, al hacer más frio hay menos niebla
"""


user = getpass.getuser()

if platform.system() == "Darwin":
    os.chdir("/Users/"+user+"/Documents/D/Niebla/")
elif platform.system() == "Linux":
    os.chdir("/home/"+user+"/Documentos/D/Niebla/")
else:
    print(" ERROR: Platform not supported!")
    sys.exit()

#sys.path.append('src/aux/')
#import aux_functions

DATA_PATH = "./Data/Datos_A8/"

# Los datos a tratar son con el intervalo de tiempo de TIME_INTERVAL
TIME_INTERVAL = 30

# Si no existe, crear el path para los nuevos datos
NEW_DATA_PATH = "./Data/Treated_Data/Classification/"
os.makedirs(NEW_DATA_PATH, exist_ok=True)


print(" # Opening file: " + str(TIME_INTERVAL) + "m")
FILENAME = "Visib_A8_Todas_"+str(TIME_INTERVAL)
#Fichero con los datos en 30 minutos
data = loadmat(DATA_PATH+FILENAME+".mat")[FILENAME]

"""
    Manipular datos para poder utilizarlos más facilmente
        * Hacer desplazamiento de la y
        * Poner los nombres a las columnas
        * Crear vector para clases
"""
data_ = data.copy()

print(" # Shifting y variable to predict y_t with x_{t-1}")
# Hacer desplazamiento
vis = data[:,2]
#Quitar el primer valor de visibilidad
vis = vis[1:]
#Quitar el ultimo valor del dataset
data = data[:-1,:]

#Nombre de las columnas
cols = ["qpRa","sa","vs","hr","at","st","td","gr","ws","wd","ap"]

#Pasar a dataframe
df = pd.DataFrame(data, columns=cols)

#Guardar los datos de visibilidad desplazados en el dataset
df["y_t+1"] = vis

#Ver la distribución de la visibilidad
#sns.distplot(df["vs"])

"""
    Organizar los valores de visibilidad en 5 clases
"""

print(" # Converting visibility values to classes")

#Crear vectir para las clases
classes = np.zeros(df["vs"].shape, dtype=np.int)

# TODO:
# Cambiado intervalos:
#  Original: 0-40, 40-200, 200-1000, 10000-2000, >=2000
#  Cambiado: 0-40, 40-200, 200-500, 500-2000, >=2000

#classes[(df["vs"] < 40)] = 0
classes[((df["y_t+1"] >= 40) & (df["y_t+1"] < 200))] = 1
classes[((df["y_t+1"] >= 200) & (df["y_t+1"] < 500))] = 2
classes[((df["y_t+1"] >= 500) & (df["y_t+1"] < 2000))] = 3
classes[(df["y_t+1"] >= 2000)] = 4


# TODO:
# Meter la vs como variable categórica

#Ver la distribución por clases
#sns.distplot(classes)

#Insertar las clases en el DF
df["class"] = classes

"""
    Guardar los datos con las clases en un fichero de texto
"""

print(" # Saving pd.DF to csv with class values", flush=True)

#Quitar la columna de visibilidad con la que se ha obtenido la clase antes de guardar
df.drop("y_t+1", axis=1, inplace=True)

#Guardar el fichero con las clases para los datos esccogidos
df.to_csv(NEW_DATA_PATH+"Visib_A8_30m_clasf.csv", index=False)


"""
    Hacer selección de instancias (quitar de la clase mayoritaria, 4, por similaridad)

    REMUESTREO de los DATOS:
         Como los procesos de Under- y Over-sampling son independientes (undersampling se hace sólo
        sobre la clase mayoritaria y posteriormente se hace oversampling sobre el resto, que sólo usa los datos de la misma clase)
        se pueden guardar los ficheros/hacer las operaciones por separado y luego juntar los datos creados con los dataset donde
        se han eliminado de la clase mayoritaria
"""

######################
# Split in train/test

# ---------
# K-Fold

# Número de Folds
K = 3

print(" # Partitioning & saving data with a K-Fold ("+str(K)+")", flush=True)

skf = StratifiedKFold(n_splits=K, random_state=None)

# Carpeta xa guardar los fold
KFold_path = NEW_DATA_PATH+"KFold_"+str(K)+"/"

df_KFold_vector = []
i=0
for train_idx, test_idx in skf.split(X=df.drop(["class"],1), y=df["class"]):
    print("  · Fold " + str(i))
    #Crear carpeta para el fold actual
    fld = KFold_path+str(i)+"/"
    os.makedirs(fld, exist_ok=True)
    #Guardar los fold creados
    df_KFold_vector.append({"train":df.iloc[train_idx], 
                            "test":df.iloc[test_idx]})
    df.iloc[train_idx].to_csv(fld+"Fold_"+str(i)+"_train.csv", index=False)
    df.iloc[test_idx].to_csv(fld+"Fold_"+str(i)+"_test.csv",  index=False)
    i+=1

# ---------
# Hold Out

print(" # Partitioning & saving data with a HoldOut (80/20)", flush=True)
X_train, X_test, y_train, y_test = train_test_split(df.drop(["class"],1), df["class"], test_size=0.2, random_state=0, stratify=df["class"])

X_train["class"] = y_train
X_test["class"]  = y_test

# Carpeta xa guardar las particiones
HoldOut_path = NEW_DATA_PATH+"HoldOut/"
os.makedirs(HoldOut_path, exist_ok=True)

X_train.to_csv(HoldOut_path+"Train.csv", index=False)
X_test.to_csv(HoldOut_path+"Test.csv", index=False)

# sys.exit()

#
######################


print(" # Performing operations on imbalanced data")

######################
#
# Undersampling
print("  @ Undersampling")

# ------------------------
# __Random Undersampling__
# Por ahora, como no hay capacidad computacional, hacer un RandomUndersampling
print("   * Random Undersampling", flush=True)


print("   -> Hold Out", flush=True)
#Calcular los ratios de cada clase
samp_stratg = {i: (sum(y_train == i) if i != 4 else int(sum(y_train == 4)*(3/4))) for i in range(5)}
# Hacer el remuestreo
rus = RandomUnderSampler(sampling_strategy=samp_stratg, random_state=0) #sampling_strategy='majority', random_state=0)
X_resampled, y_resampled = rus.fit_resample(X_train.drop(["class"],1), X_train["class"])

print("   · Saving file", flush=True)
#Añadir columna clase a dataset generado
X_resampled["class"] = y_resampled
#Hacer operaciones sobre path
# RUS_path = HoldOut_path+"Resampled/RUS/"
# os.makedirs(RUS_path, exist_ok=True)
# Guardar los datos generados
X_resampled.to_csv(HoldOut_path+"Train_RUS.csv", index=False)

print("   -> K-Fold", flush=True)
for i in range(K):
    #Calcular los ratios de cada clase
    cls_kfold_rus = df_KFold_vector[i]["train"]["class"]
    samp_stratg = {i: (sum(cls_kfold_rus == i) if i != 4 else int(sum(cls_kfold_rus == 4)*(3/4))) for i in range(5)}
    # Crear objeto de la clase RUS
    rus = RandomUnderSampler(sampling_strategy=samp_stratg, random_state=0) #sampling_strategy='majority', random_state=0)
    # Assign value to path variable
    fld = KFold_path+str(i)+"/"
    # Perform resampling
    print("   · Resampling: fold " + str(i), flush=True)
    X_resampled, y_resampled = rus.fit_resample(df_KFold_vector[i]["train"].drop(["class"],1), df_KFold_vector[i]["train"]["class"])
    X_resampled["class"] = y_resampled
    print("   · Saving file", flush=True)
    X_resampled.to_csv(fld+"Fold_"+str(i)+"_train_RUS.csv")

# ------------------------
# __Condensed Nearest Neighbour__
# Descript
print("\n   * Condensed Nearest Neighbour", flush=True)

print("   -> Hold Out", flush=True)
# Hacer el remuestreo
cnn = CondensedNearestNeighbour(sampling_strategy='majority', random_state=0, n_jobs=-1)
X_resampled, y_resampled = cnn.fit_resample(X_train.drop(["class"],1), X_train["class"])

print("   · Saving file", flush=True)
#Añadir columna clase a dataset generado
X_resampled["class"] = y_resampled
#Hacer operaciones sobre path
# CNN_path = HoldOut_path+"Resampled/CNN/"
# os.makedirs(CNN_path, exist_ok=True)
# Guardar los datos generados
X_resampled.to_csv(HoldOut_path+"Train_CNN.csv", index=False)

print("   -> K-Fold", flush=True)
cnn = CondensedNearestNeighbour(sampling_strategy='majority', random_state=0, n_jobs=-1)
for i in range(K):
    # Assign value to path variable
    fld = KFold_path+str(i)+"/"
    # Perform resampling
    print("   · Resampling: fold " + str(i), flush=True)
    X_resampled, y_resampled = cnn.fit_resample(df_KFold_vector[i]["train"].drop(["class"],1), df_KFold_vector[i]["train"]["class"])
    X_resampled["class"] = y_resampled
    print("   · Saving file", flush=True)
    X_resampled.to_csv(fld+"Fold_"+str(i)+"_train_CNN.csv")


# ------------------------
# __Neighbourhood Cleaning Rule__
# Descript
print("\n   * Neighbourhood Cleaning Rule", flush=True)

print("   -> Hold Out", flush=True)
# Hacer el remuestreo
ncr = NeighbourhoodCleaningRule(sampling_strategy='majority', n_jobs=-1)
X_resampled, y_resampled = ncr.fit_resample(X_train.drop(["class"],1), X_train["class"])

print("   · Saving file", flush=True)
#Añadir columna clase a dataset generado
X_resampled["class"] = y_resampled
#Hacer operaciones sobre path
# NCR_path = HoldOut_path+"Resampled/NCR/"
# os.makedirs(NCR_path, exist_ok=True)
# Guardar los datos generados
X_resampled.to_csv(HoldOut_path+"Train_NCR.csv", index=False)

print("   -> K-Fold", flush=True)
ncr = NeighbourhoodCleaningRule(sampling_strategy='majority', n_jobs=-1)
for i in range(K):
    # Assign value to path variable
    fld = KFold_path+str(i)+"/"
    # Perform resampling
    print("   · Resampling: fold " + str(i), flush=True)
    X_resampled, y_resampled = ncr.fit_resample(df_KFold_vector[i]["train"].drop(["class"],1), df_KFold_vector[i]["train"]["class"])
    X_resampled["class"] = y_resampled
    print("   · Saving file", flush=True)
    X_resampled.to_csv(fld+"Fold_"+str(i)+"_train_NCR.csv")

# ------------------------
# __Tomek Links__
# Descript
print("\n   * Tomek Links", flush=True)

print("   -> Hold Out", flush=True)
# Hacer el remuestreo
tl = TomekLinks(sampling_strategy='majority', n_jobs=-1)
X_resampled, y_resampled = tl.fit_resample(X_train.drop(["class"],1), X_train["class"])

print("   · Saving file", flush=True)
#Añadir columna clase a dataset generado
X_resampled["class"] = y_resampled
#Hacer operaciones sobre path
# TL_path = HoldOut_path+"Resampled/TL/"
# os.makedirs(TL_path, exist_ok=True)
# Guardar los datos generados
X_resampled.to_csv(HoldOut_path+"Train_TL.csv", index=False)

print("   -> K-Fold", flush=True)
tl = TomekLinks(sampling_strategy='majority', n_jobs=-1)
for i in range(K):
    # Assign value to path variable
    fld = KFold_path+str(i)+"/"
    # Perform resampling
    print("   · Resampling: fold " + str(i), flush=True)
    X_resampled, y_resampled = tl.fit_resample(df_KFold_vector[i]["train"].drop(["class"],1), df_KFold_vector[i]["train"]["class"])
    X_resampled["class"] = y_resampled
    print("   · Saving file", flush=True)
    X_resampled.to_csv(fld+"Fold_"+str(i)+"_train_TL.csv")


#
######################


print("\n\n @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ ")
print("    To oversample data run ./oversample_script.py ")
print(" @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ \n\n")


# ######################
# #
# # Oversampling
# print("  @ Oversampling", flush=True)

# # ------------------------
# # __SMOTE__
# # Descript
# print("\n   * SMOTE", flush=True)

# print("     @ Oversampling Only original data", flush=True)

# print("   -> Hold Out", flush=True)
# # Hacer el remuestreo
# # Dar a las clases minoritarias 5/4 el número de instancias de la 2a clase mayoritaria
# strategy = X_train["class"].value_counts().to_dict()
# strategy = {i: int(strategy[1]*(5/4)) if i!=4 else strategy[4]  for i in range(len(strategy))}
# smt = SMOTE(sampling_strategy=strategy, random_state=0, n_jobs=-1)
# X_resampled, y_resampled = smt.fit_resample(X_train.drop(["class"],1), X_train["class"])
# #Añadir columna clase a dataset generado
# X_resampled["class"] = y_resampled

# print("   · Saving file", flush=True)
# #Hacer operaciones sobre path
# # SMT_path = HoldOut_path+"Resampled/SMT/"
# # os.makedirs(SMT_path, exist_ok=True)
# # Guardar los datos generados
# X_resampled.to_csv(HoldOut_path+"Train_SMT.csv", index=False)


# print("   -> K-Fold", flush=True)
# for i in range(K):
#     # Dar a las clases minoritarias 5/4 el número de instancias de la 2a clase mayoritaria
#     strategy = X_train["class"].value_counts().to_dict()
#     strategy = {i: int(strategy[1]*(5/4)) if i!=4 else strategy[4]  for i in range(len(strategy))}
#     smt = SMOTE(sampling_strategy=strategy, random_state=0, n_jobs=-1)
#     # Assign value to path variable
#     fld = KFold_path+str(i)+"/"
#     # Perform resampling
#     print("   · Resampling: fold " + str(i), flush=True)
#     X_resampled, y_resampled = smt.fit_resample(df_KFold_vector[i]["train"].drop(["class"],1), df_KFold_vector[i]["train"]["class"])
#     X_resampled["class"] = y_resampled
#     print("   · Saving file", flush=True)
#     X_resampled.to_csv(fld+"Fold_"+str(i)+"_train_SMT.csv", index=False)


# print("\n     @ Oversampling undersampled data", flush=True)
# """
#      Como el método CNN deja demasiadas pocas instancias de la clase mayoritaria, se utilizarán el resto de los métodos para hacer
#     oversampling de las clases minoritarias
# """

# # El proceso de Oversampling se va a hacer cargando los datos de los ficheros ya procesados y guardados en disco

# print("   -> Hold Out", flush=True)
# HoldOut_files = os.listdir(HoldOut_path)
# r = re.compile("(Train_)(?!CNN)")

# for i in filter(r.match, HoldOut_files):
#     print("   * Oversampling " + str(i[6:-4]) + " data", flush=True)
#     print("   · Reading file", flush=True)
#     # Cargar fichero csv
#     i_df = pd.read_csv(HoldOut_path+i)

#     #print(i_df["class"].value_counts().to_dict(), flush=True)
#     strategy = i_df["class"].value_counts().to_dict()
#     strategy = {j: int(strategy[1]*(5/4)) if j!=4 else strategy[4]  for j in range(len(strategy))}
#     #print(strategy, flush=True)
#     print("   · Oversampling", flush=True)
#     smt = SMOTE(sampling_strategy=strategy, random_state=0, n_jobs=-1)
#     X_resampled, y_resampled = smt.fit_resample(i_df.drop(["class"],1), i_df["class"])
#     X_resampled["class"] = y_resampled
#     print("   · Saving file", flush=True)
#     X_resampled.to_csv(HoldOut_path+'_SMT.'.join(i.split(".")), index=False)

# print("   -> K-Fold", flush=True)
# r = re.compile(".*(train_)(?!CNN)")
# for i in range(K):
#     print("\n   # Oversampling Fold "+str(i), flush=True)
#     fld = KFold_path+str(i)+"/"
#     KFold_files = os.listdir(fld)
#     for j in filter(r.match, KFold_files):
#         print("    * Oversampling " + str(j[13:-4]) + " data", flush=True)
#         print("    · Reading file", flush=True)
#         # Cargar fichero csv
#         j_df = pd.read_csv(fld+j)
#         strategy = j_df["class"].value_counts().to_dict()
#         strategy = {k: int(strategy[1]*(5/4)) if k!=4 else strategy[4]  for k in range(len(strategy))}
#         print("    · Oversampling", flush=True)
#         smt = SMOTE(sampling_strategy=strategy, random_state=0, n_jobs=-1)
#         X_resampled, y_resampled = smt.fit_resample(j_df.drop(["class"],1), j_df["class"])
#         X_resampled["class"] = y_resampled
#         print("    · Saving file", flush=True)
#         X_resampled.to_csv(fld+'_SMT.'.join(j.split(".")), index=False)


# #
# ######################

print(" # Done")

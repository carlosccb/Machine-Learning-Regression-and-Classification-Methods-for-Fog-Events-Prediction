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
import pickl

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
        THIS SCRIPT HAS TO BE RUN AFTER data_wrangling_oversample.py

"""

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
"""

user = getpass.getuser()

if platform.system() == "Darwin":
    os.chdir("/Users/"+user+"/Documents/D/Niebla/")
elif platform.system() == "Linux":
    os.chdir("/home/"+user+"/Documentos/D/Niebla/")
else:
    print(" ERROR: Platform not supported!")
    sys.exit()

DATA_PATH = "./Data/Datos_A8/"


# Los datos a tratar son con el intervalo de tiempo de TIME_INTERVAL
TIME_INTERVAL = 30

# Si no existe, crear el path para los nuevos datos
NEW_DATA_PATH = "./Data/Treated_Data/Classification/"

"""
    Cargar los datos con las clases del fichero de texto
"""

print(" # Loading pd.DF from", flush=True)
#Guardar el fichero con las clases para los datos esccogidos
df = pd.read_csv(NEW_DATA_PATH+"Visib_A8_30m_clasf.csv")


######################
# Load in train/test

# ---------
# K-Fold

# Número de Folds
K = 3

# Carpeta xa guardar los fold
KFold_path = NEW_DATA_PATH+"KFold_"+str(K)+"/"

print(" # Loading K-Fold data ("+str(K)+")", flush=True)

df_KFold_vector = []
for i in range(K):
    print("  · loading Fold " + str(i))
    #Crear carpeta para el fold actual
    fld = KFold_path+str(i)+"/"

    df_tr = pd.read_csv(fld+"Fold_"+str(i)+"_train.csv")
    df_ts = pd.read_csv(fld+"Fold_"+str(i)+"_test.csv")

    #Guardar los fold creados
    df_KFold_vector.append({"train":df_tr, 
                            "test":df_ts})

# ---------
# Hold Out

HoldOut_path = NEW_DATA_PATH+"HoldOut/"

X_train = pd.read_csv(HoldOut_path+"Train.csv")
X_test = pd.read_csv(HoldOut_path+"Test.csv")

y_train = X_train["class"]
y_test  = X_test["class"]
#
######################


######################
#
# Oversampling
print("  @ Oversampling", flush=True)

# ------------------------
# __SMOTE__
# Descript
print("\n   * SMOTE", flush=True)

print("     @ Oversampling Only original data", flush=True)

print("   -> Hold Out", flush=True)
# Hacer el remuestreo
# Dar a las clases minoritarias 5/4 el número de instancias de la 2a clase mayoritaria
strategy = X_train["class"].value_counts().to_dict()
strategy = {i: int(strategy[1]*(5/4)) if i!=4 else strategy[4]  for i in range(len(strategy))}
smt = SMOTE(sampling_strategy=strategy, random_state=0, n_jobs=-1)
X_resampled, y_resampled = smt.fit_resample(X_train.drop(["class"],1), X_train["class"])
#Añadir columna clase a dataset generado
X_resampled["class"] = y_resampled

print("   · Saving file", flush=True)
#Hacer operaciones sobre path
# SMT_path = HoldOut_path+"Resampled/SMT/"
# os.makedirs(SMT_path, exist_ok=True)
# Guardar los datos generados
X_resampled.to_csv(HoldOut_path+"Train_SMT.csv", index=False)


print("   -> K-Fold", flush=True)
for i in range(K):
    # Dar a las clases minoritarias 5/4 el número de instancias de la 2a clase mayoritaria
    strategy = df_KFold_vector[i]["train"]["class"].value_counts().to_dict()
    strategy = {i: int(strategy[1]*(5/4)) if i!=4 else strategy[4]  for i in range(len(strategy))}
    smt = SMOTE(sampling_strategy=strategy, random_state=0, n_jobs=-1)
    # Assign value to path variable
    fld = KFold_path+str(i)+"/"
    # Perform resampling
    print("   · Resampling: fold " + str(i), flush=True)
    X_resampled, y_resampled = smt.fit_resample(df_KFold_vector[i]["train"].drop(["class"],1), df_KFold_vector[i]["train"]["class"])
    X_resampled["class"] = y_resampled
    print("   · Saving file", flush=True)
    X_resampled.to_csv(fld+"Fold_"+str(i)+"_train_SMT.csv", index=False)


print("\n     @ Oversampling undersampled data", flush=True)
"""
     Como el método CNN deja demasiadas pocas instancias de la clase mayoritaria, se utilizarán el resto de los métodos para hacer
    oversampling de las clases minoritarias
"""

# El proceso de Oversampling se va a hacer cargando los datos de los ficheros ya procesados y guardados en disco

print("   -> Hold Out", flush=True)
HoldOut_files = os.listdir(HoldOut_path)
r = re.compile("(Train_)(?!CNN)(?!SMT)")

for i in filter(r.match, HoldOut_files):
    print("   * Oversampling " + str(i[6:-4]) + " data", flush=True)
    print("   · Reading file", flush=True)
    # Cargar fichero csv
    i_df = pd.read_csv(HoldOut_path+i)

    #print(i_df["class"].value_counts().to_dict(), flush=True)
    strategy = i_df["class"].value_counts().to_dict()
    strategy = {j: int(strategy[1]*(5/4)) if j!=4 else strategy[4]  for j in range(len(strategy))}
    #print(strategy, flush=True)
    print("   · Oversampling", flush=True)
    smt = SMOTE(sampling_strategy=strategy, random_state=0, n_jobs=-1)
    X_resampled, y_resampled = smt.fit_resample(i_df.drop(["class"],1), i_df["class"])
    X_resampled["class"] = y_resampled
    print("   · Saving file", flush=True)
    X_resampled.to_csv(HoldOut_path+'_SMT.'.join(i.split(".")), index=False)

print("   -> K-Fold", flush=True)
r = re.compile(".*(train_)(?!CNN)(?!SMT)")
for i in range(K):
    print("\n   # Oversampling Fold "+str(i), flush=True)
    fld = KFold_path+str(i)+"/"
    KFold_files = os.listdir(fld)
    for j in filter(r.match, KFold_files):
        print("    * Oversampling " + str(j[13:-4]) + " data", flush=True)
        print("    · Reading file", flush=True)
        # Cargar fichero csv
        j_df = pd.read_csv(fld+j)
        strategy = j_df["class"].value_counts().to_dict()
        strategy = {k: int(strategy[1]*(5/4)) if k!=4 else strategy[4]  for k in range(len(strategy))}
        print("    · Oversampling", flush=True)
        smt = SMOTE(sampling_strategy=strategy, random_state=0, n_jobs=-1)
        X_resampled, y_resampled = smt.fit_resample(j_df.drop(["class"],1), j_df["class"])
        X_resampled["class"] = y_resampled
        print("    · Saving file", flush=True)
        X_resampled.to_csv(fld+'_SMT.'.join(j.split(".")), index=False)
#
######################

print(" # Done")

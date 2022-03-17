import os, sys
import glob

import datetime

import pandas as pd
import numpy as np

##############################
# Definir el modelo de persistencia
def model_persistence(vs):
    y_pred = np.zeros(vs.shape[0],dtype=np.dtype(int))
    y_pred[(vs >= 2000)] = 4
    y_pred[(vs >= 1000) & (vs < 2000)] = 3
    y_pred[(vs >= 200)  & (vs < 1000)] = 2
    y_pred[(vs >= 40)   & (vs < 200)]  = 1
    return y_pred
#
##############################


# -----------------
# Variables
# -----------------

ROOT_PATH = "~/Documents/D/" # Para ejecutar en mis PC
# ROOT_PATH = "/home/ccastillo/" # Para lanzar en el cluster

DATA_PATH = ROOT_PATH+"Niebla/Data/Treated_Data/Classification/HoldOut/"
LOG_PATH  = ROOT_PATH+"Niebla/log_files/alg_comp/classification/nominal/"
PRED_PATH = ROOT_PATH+"Niebla/log_files/predictions/classification/nominal/"

print(" # Loading file")
# Cargar los datos de test
test_df = pd.read_csv(DATA_PATH+"Test.csv")
y_ts = test_df["class"]
vs_ts = test_df["vs"]

print(" # Predicting")
# Hacer la predicción con la persistencia
start=datetime.datetime.now()
y_pred = model_persistence(vs_ts)
end=datetime.datetime.now()

print(f"  => Elapsed time: {str(end - start)}")

print(" # Saving predictions")
pd.Series(y_pred, dtype=np.dtype(int)).to_csv(PRED_PATH+"Persistence_pred_class.csv", index=False)

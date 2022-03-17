library(tidyverse)

# ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ##
#
# Script creado para comparar los resultados entre la clasificación ordinal y la nominal
#
# ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ##

setwd("/home/carlos/Documents/D/Niebla/log_files/alg_comp/classification/")

#  Ficheros que contienen todas las métricas deseadas para cada combinación de Dataset+alpha
# Para sacar estos ficheros hay que utilizar el script alculate_metrics_sort_compare.py
# que se encuentra en cada una de las carpetas (ordinal, nominal)
df_ordinal <- read_csv("ordinal/Complete_by_class_Ordinal_comparison.csv")
# TODO: Hay que ejecutar el script correspondiente para sacar estos datos
df_nominal <- read_csv("nominal/Complete_by_class_Nominal_comparison.csv")

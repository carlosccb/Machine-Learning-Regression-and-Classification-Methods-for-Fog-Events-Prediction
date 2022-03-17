# ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ##
# ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- -----
#
# ATTENTION: This file is old and deprecated, use calculate_metrics_sort_compare.py
# This file uses R's metrics which is problematic. The other file calculates the metrics
# from the predictions and is much better overall.
#
# ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- -----
# ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ##

import numpy as np
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt

import os, sys
from fnmatch import fnmatch

os.chdir("/home/carlos/Documents/D/Niebla/log_files/alg_comp/classification/nominal")

get_dataset_nm = lambda x: x.split("-")[1]

results_df = pd.DataFrame()

for f in os.listdir("."):
	#if fnmatch(f, "*scaled-metrics_by_class.csv") and (f != "Complete_results-metrics_by_class.csv"):
	if fnmatch(f, "*metrics_by_class.csv") and (f != "Complete_results-metrics_by_class.csv"):
		print(f" # File {f}")
		df_ = pd.read_csv(f)
		if (nm := get_dataset_nm(f)[1:]) == "":
			nm = "OG"
		df_["Dataset"] = nm
		results_df = results_df.append(df_)

results_df.reset_index(inplace=True, drop=True)

#Poner el nombre de las columnas en Title
results_df.columns = [i.title() for i in results_df.columns]

#Poner el dataset como primera columna
cols = results_df.columns.tolist()
cols = cols[-1:] + cols[:-1]
results_df = results_df[cols]

#Sacar medias por modelo/dataset/metrica
results_df["Mean"] = results_df.iloc[:,-5:].mean(axis=1)

#Save dataset
#results_df.to_csv("Complete_results-scaled-metrics_by_class.csv", index=False)
results_df.to_csv("Complete_results-metrics_by_class.csv", index=False)

sys.exit()

# ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ##
# ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- -----
# Visualization

#Sorted from best to worst (value)
best = results_df["Mean"].sort_values(ascending=False).index
results_df.loc[best,["Dataset","Model","Metric","Mean"]].dropna()
plot_data = results_df.loc[best,["Dataset","Model","Metric","Mean"]].dropna()

plot_data["Configuration"] = (plot_data[["Dataset","Model","Metric"]]
                              .agg('-'.join, axis=1)
                              .str.replace("alpha_","")
                              .str.replace("0\.",".")
                              .str.replace("precision","P")
                              .str.replace("recall","Rc")
                              .str.replace("b_acc","bAc"))

plot_data_Acc = plot_data[plot_data["Metric"] == "b_acc"]
plot_data_Prc = plot_data[plot_data["Metric"] == "precision"]
plot_data_Rcl = plot_data[plot_data["Metric"] == "recall"]
plot_data_F1  = plot_data[plot_data["Metric"] == "F1"]

# -----------------------------------------------------------------
sns.barplot(x=plot_data_Prc["Configuration"], y=plot_data_Prc["Mean"])
plt.xticks(rotation=75);

# ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- -----
# ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ##

import pandas as pd
import numpy as np

import os, sys, glob

for f in glob.glob("./elapsed*"):
	print(" # Reading file " + f)
	data = pd.read_csv(f)

	print("  * Treating file " + f)
	data["end_time"] = pd.to_datetime(data["end_time"])
	data["start_time"] = pd.to_datetime(data["start_time"])

	data["elapsed_time"] = data["end_time"]-data["start_time"]

	print("  * Saving file " + f)
	data.to_csv(f, index=False)

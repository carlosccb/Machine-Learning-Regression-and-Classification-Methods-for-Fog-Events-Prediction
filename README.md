# Repository for the publication "Machine Learning Regression and Classification Methods for Fog Events Prediction"

This repository contains the code for the paper "Machine Learning Regression and Classification Methods for Fog Events Prediction''. 

## Source code

The code for this publication are _Python_ and _R_ scripts included in [/src](/src). Under that directory, more folders are included. Each one contains the code related to the name of the folder. It's pretty self explanatory.

The scripts used to run the experiments and obtain the final predictions and metrics are in [/src/exps](/src/exps/). To clean and treat the data [/src/Data](/src/Data).

Jupyter notebooks were used for the EDA stage and for some comparisons during the research process, and are a minimal part of the code (even if the percentage bar on the right says otherwise, but that's intrinsic to jupyter notebook).

## README: Important notice

As an advisory notice, here is some __IMPORTANT INFORMATION YOU MUST KNOW__ before viewing the source code and/or reaching any conclusion from the files included in this repository:

* Only the code is contained. The data can not be shared publicly here as the rights belong to a national governmental organization.

* The code is research code so it is not: organized, optimized, pretty, etc... It also does not adhere to any software engineering practices as it wasn't meant to be published or used after the results had been obtained.

* The comments are mostly in Spanish. The code is not thoroughly commented. Some comments indicate that a file is wrong, that something is provisional or it wasn't used in the final results. Even if it is not said, not all source code files have been used, and/or its results have not been used in the final publication.

* As the data can't be published and it needs quite a lot of pre-processing and treatment, the bare bones structure of the Data folder has been kept as a demo of the structure of the project.

* I guarantee that all final results obtained and published are reproducible with the correct data and pre-processing, and running the right scripts for processing and for training and inference, which are all included.

* Not all included scripts have been used in the final results. Following the naming scheme explained in the paper the right processing and experiment script combination can be followed, and if all is equal to the process followed during research, the final results will be the same.

* I keep in a private repository all the data, original and transformed, and all the logs with: predictions, metrics, running times for experiments, sorted models and methods, etc.

* I keep all rights over the work published here.

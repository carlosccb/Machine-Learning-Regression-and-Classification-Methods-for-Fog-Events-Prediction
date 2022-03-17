import os
import sys
import getpass
import copy
import math

from datetime import datetime

import pandas as pd
import numpy as np
from scipy.io import loadmat

from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import scale, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, make_scorer
from sklearn.metrics import accuracy_score, roc_auc_score, cohen_kappa_score, f1_score, confusion_matrix

import random

#Set seed
random.seed(0)
np.random.seed(0)

#user = getpass.getuser()
##TODO: Esto hay que cambiarlo para cada proyecto
#os.chdir("/Users/"+user+"/Documents/D/Niebla/")

sys.path.append('src/aux/')
import aux_functions

aux_functions.set_environment_location()

"""
	Quizás convendría hacer una clase genérica y que heredaran las clases para implementar KFold & HoldOut, en Kfold se llamaría tantas veces como fold hay a
	las funciones y se haría la media (pensar esto que es tarde y no se si podría ser así exactamente)
"""
class Experiment:
	def __init__(self, model, train_x, train_y, test_x, test_y, kfold, exp_name,
				 model_name, task, grid_search=None, scaler=None, verbose=False,
				 threshold_up=False, threshold_lo=False):
		"""Recieves parameters to run an experiment
			* model: Objeto creado para ser entrenado en el caso de no hacer GridSearchCV.
				En el caso de sí hacer GridSearchCV, el objeto será reemplazado por best_estimator_
				Este mismo objeto es el modelo pasado al objeto GridSearchCV en su constructor.
			* kfold: Whether the data is in a HoldOut partition (kfold=False) or in a KFold (kfold=k) 
			* train_x: Variables independientes de los  de train
			* train_y: Variable a predecir en entrenamiento
			* test_x: Variables independientes de los datos de test
			* test_y: Variable a predecir en test
			* exp_name: Nombre del experimento que se está realizando. Utilizado para guardar logs
				y configuraciones y poder distinguir entre distintos experimentos
			* model_name: Nombre del modelo que se está utilizando. Se utiliza de manera conjunta con
				la varible exp_name para identificar ficheros de log y configuración, etc
			* grid_search: Objeto de la clase GridSearchCV previamente configurado con los valores
				para los hiperperámettros.
				TODO: En próximas versiones posiblemente se creará el objeto en la clase con un fichero
					de configuración que contenga los valores de hiperparámetros a probar.
			* task: Si el problema es de regresión, clasificación, clasif. ordinal. Utilizado para las métricas a
				obtener a partir de las predicciones
			* scaler: Objeto para hacer el escalado de las variables (de momento se hace en el main)
				TODO: Si se modifica esta clase para aceptar ficheros de configuración, esto se podría volver a utilizar
					para realizar el escalado aquí
			* verbose: si se quiere obtener salida por stdout durante el proceso de ejecución de la clase

		TODO: 
			- Seguir adaptando lo que queda para que funcione con KFold
				Quizás habría que hacer funciones "sobrecargadas", o en su defecto, hacer una función genérica que llamara en cada caso a la función
				que ejecuta la funcionalidad deseada
			- Añadir funcionalidad para utilizar ficheros de configuración
				- Hiperparámetros de GridSearchCV
				- Métricas
				- Experimentos a lanzar
				- ....
			- Crear carpetas para los experimentos generales y subcarpetas para las modificaciones de los datos, por ej:
				- temp_split (partición train/test temporal 80/20)
					- PCA
					- min_max
					-robust
				- temp_split_regr (partición train/test temporal 80/20 con valores regresivos de la y)
					- PCA
					- min_max
					-robust
				- temp_split_regr_clasif (partición train/test temporal 80/20 con valores regresivos de la y, clasificación)
					- ....
			- Añadir funcionalidad para crear carpetas
			- Añadir función para guardar logs
			- Añadir función para hacer plots
			- Añadir para guardar en pikle los objetos (entrenados) de GridSearchCV
				Igual conviene más utilizar joblib
			- Añadir para hacer el log de cómo van de avanzadas las cosas la librería logging (https://docs.python.org/3/library/logging.html)
				- Añadir las opciones para que se pueda hacer por stdout, log o ambas

		DONE
			- Añadir documentación
		"""
		self.model = model

		#Enable use of K-Fold
		#if (train_y.values.ndim > 1) and (train_x.values.ndim > 2) and (train_y.values.shape[1] == train_x.values.shape[1]):
		#if kfold:
		#self.__k_fold__ = kfold if kfold == len(train_y) else kfold #train_y.values.shape[0]
		if kfold == False:
			self.__k_fold__ = kfold
		elif kfold > 1:
			if kfold == len(train_y):
				self.__k_fold__ = kfold
			else:
				print(" Error: Mismatch in kfold size!")
				sys.exit(1)
		else:
			print(" Warning: Taking number of elements in list train as K-Fold partitions")
			self.__k_fold__ = len(train_y)
		#else:
		#	self.__k_fold__ == False

		self.train_y = train_y
		self.test_y  = test_y
		self.grid_search = grid_search

		self.model_name = model_name
		self.exp_name = exp_name

		if threshold_up:
			self.threshold_up = threshold_up
		elif threshold_lo: 
			self.threshold_lo = threshold_lo

		assert task in ["regression", "classification", "ordinal_classification"]
		self.task = task

		#Guardar en una carpeta que sea exp_name[0]/exp_name[1].....csv
		# exp_data_type_ = exp_name.split("-")[0]
		# exp_data_prep_ = exp_name.split("-")[1]
		# print(" in -> " + str(exp_name))
		self.exp_type_name = exp_name.split("-")[0]
		self.exp_data_prep_name = exp_name.split("-")[1]
		self.exp_logs_path = self.exp_type_name + "/" + self.exp_data_prep_name

		# En una carpeta para cada experimento y tipo de preproc, se guarda _UN SOLO_ fichero para comparar _TODOS_ los métodos
		self.ALG_COMP_FILE    = aux_functions.ALG_COMP+self.exp_logs_path+".csv"
		# En una carpeta para cada experimento y tipo de preproc, se guarda _UN SOLO_ fichero con las pred de todos los metodos
		self.PREDICTIONS_FILE = aux_functions.PREDICTIONS_FOLDER+self.exp_logs_path+"/preds-"+self.exp_data_prep_name+".csv"
		# En una carpeta para cada experimento y tipo de preproc, se guarda _UN SOLO_ fichero para comparar la duración de _TODOS_ los métodos
		self.DURATION_FILE    = aux_functions.DURATION_FOLDER+self.exp_logs_path+".csv"
		# En una carpeta para cada experimento y tipo de preproc, se guarda un fichero individual para cada modelo
		self.BEST_CONFIG_FILE = aux_functions.BEST_CONFIG+self.exp_logs_path+"/"+self.model_name+".csv"

		aux_functions.check_folder_structure(self.exp_type_name, self.exp_data_prep_name, self.model_name)

		if scaler != None:
			self.scaler = scaler
			if self.__k_fold__:
				self.train_x, self.test_x = [], []
				for i in range(self.__k_fold__):
					self.train_x.append(self.scaler(train_x[i]))
					self.test_x.append(self.scaler(test_x[i]))
			else:
				self.train_x = self.scaler(train_x)
				self.test_x  = self.scaler(test_x)
		else:
			self.train_x = train_x
			self.test_x  = test_x

		self.verbose = verbose

		#Private & Internal variables
		#Keeps track of wether the time is being kept during an exp 
		self.__is_timming__ = False
		#Variable that holds the number of classes
		self.__num_classes__ = len(np.unique(self.train_y)) if not self.__k_fold__ else len(np.unique(np.concatenate((self.train_y))))

		#TODO: No estaría mal imprimir aquí un resumen de la configuración elegida para que compruebe el usuario que es correcta


	def get_metrics(self):
		"""Function to get metrics of predictions. Uses the variable regression passed in the constructor.
		By default regression.
		"""
		# Regression metrics
		if self.__k_fold__:
			print(" Error: Currently the functionality to use K-Fold data hasn't been added. Sorry!!")
			return

		if self.task == "regression":
			r2 = r2_score(self.test_y, self.y_pred)
			if self.verbose: print("  * R2: " + str(r2))
			MSE = mean_squared_error(self.test_y, self.y_pred)
			if self.verbose: print("  * MSE: " + str(MSE))
			RMSE = math.sqrt(MSE)
			if self.verbose: print("  * RMSE: " + str(RMSE))
			MAE = mean_absolute_error(self.test_y, self.y_pred)
			if self.verbose: print("  * MAE: " + str(MAE))
			#Save classification metrics
			self.metrics = {"R2": r2, "MSE": MSE, "RMSE": RMSE, "MAE": MAE}
		# Classification metrics
		elif self.task == "classification":
			acc = accuracy_score(self.test_y, self.y_pred)
			if self.verbose: print("  * ACC:" + str(acc))
			if self.__num_classes__ == 2:
				auc = roc_auc_score(self.test_y, self.y_pred)
				if self.verbose: print("  * AUC:" + str(auc))
				f1  = f1_score(self.test_y, self.y_pred_prob)
				if self.verbose: print("  * F1:" + str(f1))
			else:
				auc = roc_auc_score(self.test_y, self.y_pred_prob, multi_class="ovo")
				if self.verbose: print("  * AUC:" + str(auc))
				f1  = f1_score(self.test_y, self.y_pred, average="weighted")
				if self.verbose: print("  * F1:" + str(f1))
			kap = cohen_kappa_score(self.test_y, self.y_pred)
			if self.verbose: print("  * KAP:" + str(kap))
			cm = confusion_matrix(self.test_y, self.y_pred)
			if self.verbose: print("  * CM:\n" + str(cm))
			#Save classification metrics
			self.metrics = {"ACC": acc,"AUC": auc,
							"KAP": kap,"F1": f1,"CM": cm}

		elif self.task == "ordinal_classification":
			#TODO
			pass


	def time_exp(self):
		"""Function to time an experiment
		"""
		tm = datetime.now()
		#If not counting, start
		if not self.__is_timming__:
			self.__is_timming__ = True
			self.tm_begin = datetime.now()
		#Otherwise stop
		else:
			self.__is_timming__ = False
			self.tm_end = tm
			#Get elapsed time
			self.tm_elapsed = self.tm_end - self.tm_begin

	def pca_transform(self,pc_num=.90):
		"""Transform data with PCA and get the pc_num Principal Components that amount to pc_num variance
			* pc_num: Percentage of variance explained of Principal Components to keep
		"""
		if self.__k_fold__:
			print(" Error: Currently the functionality to use K-Fold data hasn't been added. Sorry!!")
			return

		#Crear objeto PCA para que se cojan los PC que sean el pc_nu de la var original
		pca = PCA(pc_num, svd_solver="full")
		# Hacer PCA sobre datos de train
		self.train_x = pca.fit_transform(self.train_x)
		#Guardar estos valores de train
		self.explained_variance_ = pca.explained_variance_
		self.explained_variance_ratio_ = pca.explained_variance_ratio_
		#  Asegurarse de que se tienen el mismo número de PC: Crear un nuevo objeto y forzar a tener 
		# el mismo número de PC
		n_pc = pca.components_.shape[0]
		pca = PCA(n_pc)
		#Transformar los datos de test
		self.test_x  = pca.fit_transform(self.test_x)


	def run_exp(self):
		"""Run an experiment with GridSearch. With a configured model, train with data and extract predictions, trim 'em
		and get metrics. Also used by run_grid_search_exp when the optimun parameters have been found, a new object is created with the
		best parameters, trained and used to predict test data.
		"""
		if self.__k_fold__:
			print(" Error: Currently the functionality to use K-Fold data hasn't been added. Sorry!!")
			return

		self.time_exp()
		# Fit model to train data
		self.model.fit(self.train_x, self.train_y)
		self.time_exp()

		# Predict test data
		self.y_pred = self.model.predict(self.test_x)
		self.y_pred = pd.Series(self.y_pred)

		if self.task != "regression":
			if hasattr(self.model, 'predict_proba'):
				self.y_pred_prob = self.model.predict_proba(self.test_x)
			else:
				self.fake_prob()

		#Trim the predictions to thresholds
		if self.task== "regression" and (self.threshold_lo or self.threshold_up ):
			self.trim_preds()

		# Get metrics
		self.get_metrics()

		#Saving logs
		print(" # Saving Logs")
		self.save_logs()

	def run_grid_search_exp(self, ModelClass, refit=True):
		"""Run an experiment with GridSearch. There's an optional refit, meaning that the best parameters found in the
		GridSearch process are used to train a model with all the train data and then the test data is predicted. 
		Otherwise the GridSearch's best_estimator_ is used to predict the test data and obtain the metrics, etc
			* ModelClass: A class to create a new object from. Only used if refit=True
		"""
		if self.__k_fold__:
			print(" Error: Can't run grid search on K-Fold data. sklearn's gridsearch uses an internal K-Fold")
			return

		# Run Grid Search
		self.time_exp()
		self.grid_search.fit(self.train_x, self.train_y)
		self.time_exp()
		if refit:
			# Create model with optimized parameters
			self.model = ModelClass()#**self.grid_search.best_params_)
			self.model.set_params(**self.grid_search.best_estimator_.get_params())
			# Re-Train model with optimized parameters
			self.run_exp()
		else:
			self.model = self.grid_search.best_estimator_

			# Predict test data
			self.y_pred = self.model.predict(self.test_x)
			self.y_pred = pd.Series(self.y_pred)

			if self.task != "regression":
				if hasattr(self.model, 'predict_proba'):
					self.y_pred_prob = self.model.predict_proba(self.test_x)
				else:
					self.fake_prob()

			#Trim the predictions to thresholds
			if self.task== "regression" and (self.threshold_lo or self.threshold_up ):
				self.trim_preds(2000)

			# Get metrics
			self.get_metrics()

			#Saving logs
			print(" # Saving Logs")
			self.save_logs()


	def save_logs(self):
		"""Function to save the logs generated in the process of running an experiment. At the moment it saves:
			- Best parameters encountered in the GridSearch
			- Metrics obtained from predictions to a file with all other models used in the same experiment
			- Save predictions to a file so a comparation can be made with all other models used in the same experiment
			- Duration of the experiment (fit process)
		"""
		if(self.grid_search != None):
			print("   * Saving best params")
			# Save best parameters chosen on GridSearch
			#TODO: FIX
			pd.DataFrame.from_dict(self.grid_search.best_params_, orient="index").to_csv(self.BEST_CONFIG_FILE)#aux_functions.BEST_CONFIG+self.exp_type_name+"/"+self.model_name+".csv")

		print("   * Saving metrics")
		# Save the metrics
		if not os.path.isfile(self.ALG_COMP_FILE):
			#df = pd.DataFrame.from_dict(exp.metrics, orient="index")
			#pd.DataFrame(exp.metrics, columns=[exp.model_name])
			df = pd.DataFrame(list(self.metrics.items()), columns=["Metric",self.model_name+"-"+self.exp_name])
		else:
			df = pd.read_csv(self.ALG_COMP_FILE)#, index_col="Metric")
			#No hecho porque es un coñazo: 
			#  Transponer para que poder insertar facilmente (como columna)
			#TODO: Aquí se podría quitar de la columna -> "-"+self.exp_name <-
			df[self.model_name+"-"+self.exp_name] = self.metrics.values()
			#No hecho porque es un coñazo: 
			#  Transponer para que tenga en las columnas las métricas y en las filas los modelos
		#Save file
		df.to_csv(self.ALG_COMP_FILE, index=False)
		
		print("   * Saving y_pred")
		# Save y_true vs y_pred
		#Comprobar si ya existe el fichero
		if not os.path.isfile(self.PREDICTIONS_FILE):
			df = pd.DataFrame(np.column_stack((self.test_y, self.y_pred)),
							  columns=["y_true", self.model_name+"-"+self.exp_name])
		else:
			df = pd.read_csv(self.PREDICTIONS_FILE)
			#TODO: Aquí se podría quitar de la columna -> "-"+self.exp_name <-
			df[self.model_name+"-"+self.exp_name] = self.y_pred
		df.to_csv(self.PREDICTIONS_FILE, index=False)

		#Save elapsed time
		if not os.path.isfile(self.DURATION_FILE):
			df = pd.DataFrame([self.tm_elapsed], columns=["Elapsed time"], index=[self.model_name])
		else:
			df = pd.read_csv(self.DURATION_FILE, index_col=0)
			df = df.append(pd.DataFrame({"Elapsed time":{self.model_name: self.tm_elapsed}}))
		df.to_csv(self.DURATION_FILE)


	def fake_prob(self):
		"""Fake the probability prediction of a model. Predicted class gets 1 prob. and the rest 0"""
		self.y_pred_prob = np.zeros((self.test_y.shape[0],self.__num_classes__))
		self.y_pred_prob[range(len(self.y_pred)),self.y_pred] = 1


	#Trim predictions
	def trim_preds(self):
		"""Trim the values of the predictions so that none are above a given value
		or below another given value
		"""
		if self.__k_fold__:
			print(" Error: Currently the functionality to use K-Fold data hasn't been added. Sorry!!")
			return

		#TODO: Esto se puede hacer con una función de numpy: np.clip(lower_limit,upper_limit)
		self.y_pred[self.y_pred<self.threshold_lo]=self.threshold_lo
		self.y_pred[self.y_pred>self.threshold_up]=self.threshold_up

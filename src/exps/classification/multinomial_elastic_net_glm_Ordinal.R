library(tidyverse)

#library(glmnet)
# Ver cuál de estas librerías se adapta mejor a lo que hace la otra
library("ordinalNet") # Doc: https://cran.r-project.org/web/packages/ordinalNet/ordinalNet.pdf
  # Theory behing model and libray: https://arxiv.org/pdf/1706.05003.pdf
#library("glmnetcr")   # Doc: https://cran.r-project.org/web/packages/glmnetcr/glmnetcr.pdf

library("ramify")  # argmax
library("Metrics") # metrics (duh)
library("mltest")  # multiclass metrics <- da problemas con algunos dataset que no clasifican algunas clases
library(data.table) # tranpose data.frame
# Script inspired by: https://github.com/StatQuest/ridge_lasso_elastic_net_demo/blob/master/ridge_lass_elastic_net_demo.R
# Vignette on how to use glmnet: https://web.stanford.edu/~hastie/glmnet/glmnet_alpha.html#int
# Documentation of the package: https://cran.r-project.org/web/packages/glmnet/vignettes/glmnet.pdf
#                               https://cran.r-project.org/web/packages/glmnet/glmnet.pdf

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# Para meter kernel (creo) que habría que utilizar el paquete gelnet              #
# source: https://cran.r-project.org/web/packages/gelnet/gelnet.pdf               #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

ROOT_PATH <- "~/Documents/D/" # Para ejecutar en mis PC
# ROOT_PATH <- "/home/ccastillo/" # Para lanzar en el cluster

DATA_PATH <- str_c(ROOT_PATH, "Niebla/Data/Treated_Data/Classification/HoldOut/")
LOG_PATH  <- str_c(ROOT_PATH, "Niebla/log_files/alg_comp/classification/ordinal/")
PRED_PATH <- str_c(ROOT_PATH, "Niebla/log_files/predictions/classification/ordinal/")

args = commandArgs(trailingOnly=TRUE)
if(length(args)>1) {
  print(" @ Error: Too many input arguments. Aborting!")
  stop()
} else if(length(args) == 1) {
  if(! is.na(as.integer(args))) {
    dataset_num <- as.integer(args)
  } else {
    print(" @ Error: Argument can't be converted to integer. Aborting!")
    stop()
  }
} else {
  print(" No argument to select a dataset has been introduced, runing script with all datasets!")
  dataset_num <- "all"
}

# Set seed
set.seed(0)

# Load test data
test_df  <- read_csv(str_c(DATA_PATH,"Test.csv"))
# Split Test data
x.ts <- as.matrix(test_df[,-ncol(test_df)])
y.ts <- as.ordered(test_df$class + 1)

# Function to get metrics
get_metrics <- function(y.true, y.pred, verbose=FALSE) {
  n_classes <- length(unique(y.true))
  # Get metrics
  mean_metrics <- data.frame(acc=accuracy(y.true, y.pred),
                             qwk=ScoreQuadraticWeightedKappa(y.true, y.pred, 1, n_classes))
  
  if(length(unique(y.true)) !=  length(unique(y.pred))) {
    print(" * Some classes were not classified and therefore can't use multiclass metrics at the moment.")
    multclss_metrics <- data.frame(balanced.accuracy=rep(NA,5), F1=NA, precision=NA, recall=NA)
  } else {
    multclss_metrics <- ml_test(y.pred, y.true)
  }
  
  m_df <- data.frame(b_acc=multclss_metrics$balanced.accuracy,
                     F1=multclss_metrics$F1,
                     precision=multclss_metrics$precision,
                     recall=multclss_metrics$recall)
  m_df <- transpose(m_df, keep.names = "metric")
  colnames(m_df) <- c("metric", str_c("class_",1:n_classes))
  
  if(verbose) {
    print(str_c(" * Accuracy: ", m_df$acc))
    print(str_c(" * QWK: ", m_df$qwk))
    print(str_c(" * balanced accuracy: ", m_df$b_acc))
    print(str_c(" * F1: ", m_df$F1))
    print(str_c(" * precision: ", m_df$precision))
    print(str_c(" * recall: ", m_df$recall))
  }
  
  return(list(metrics_avg=mean_metrics, metrics_cls=m_df))
}

# All available datasets
DATASETS <- c("","_CNN","_NCR","_NCR_SMT","_RUS","_RUS_SMT","_SMT","_TL","_TL_SMT")

if(is.numeric(dataset_num)) {
  DATASETS <- DATASETS[dataset_num]
  # print(str_c("Using dataset: ", dataset_num, " - ", DATASETS))
} else {
  print(str_c("Using dataset all DATASETS"))
}

#Crear un dataframe en el que meter los tiempos de ejecución de cada combinación
exec_time_df <- data.frame()

for(DATASET in DATASETS) {
  print("######################################\n\n")
  print(str_c("@ Using dataset ", DATASET))
  # Load data files
  # train_df <- read_csv(str_c(DATA_PATH,"Train.csv"))
  train_df <- read_csv(str_c(DATA_PATH,"Train",DATASET,".csv"))
  
  # Split Train data
  x.tr <- as.matrix(train_df[,-ncol(train_df)])
  y.tr <- as.ordered(train_df$class + 1)
  
  # Standardize data
  #x.tr  <- scale(x.tr)
  #x.ts_ <- scale(x.ts) # TODO: No se si está bien esto: # sweep(sweep(x.tr, 2, colMeans(x.tr)), 2, apply(x.tr, 2, sd))
  
  # Obtener número de clases
  n_classes <- length(unique(y.tr))
  # Crear el dataframe para guardar los resultados
  metrics_cls <- data.frame()
  metrics_avg <- data.frame()
  #Crear una lista en la que guardar los modelos
  list_models <- list()

  # library(doParallel)
  # registerDoParallel(6)
  
  # Probar con varias combinaciones de alpha para la el_net
  for(i in 0:10) {
    alpha <- i/10 # Set alpha value
    model_nm <- str_c("alpha_",alpha) #Get model name
    print(str_c(" # Fitting model ", i, " (lambda=",alpha,")"))
  
    # Fit model on data
    # * class: misclassification error
    # * deviance: actual deviance
    start_time <- Sys.time()
    set.seed(0)
    fitted_model <- ordinalNetTune(x.tr, y.tr, nFolds=10, alpha=alpha,
                                   family="cumulative", link="logit",
                                   parallelTerms=T, nonparallelTerms=T)
    #fitted_model <- cv.glmnet(x.tr, y.tr, type.measure="class", alpha=alpha, family="multinomial") #, parallel = TRUE)
    end_time <- Sys.time()
    print(end_time-start_time)

    # Save elapsed time to dataframe
    exec_time_df <- rbind(exec_time_df, data.frame("DATASET"=DATASETS, "alpha"=alpha,
                                                   "start_time"=start_time, "end_time"=end_time,
                                                   "elapsed_time"=""))
    # Save model to list
    #list_models[[model_nm]] <- fitted_model

    # Get predictions
    #y.pred.probs <- predict(fitted_model, s=fitted_model$lambda.1se, newx=x.ts)
    #y.pred <- argmax(y.pred.probs)
    # Esto es lo mismo que lo anterior pero no hace falta argmax
    #y.pred <- predict(fitted_model, s=fitted_model$lambda.1se, newx=x.ts_, type = "class")

    #y.pred <- predict(fitted_model$fit, criteria="bic", newx=x.ts_, type = "class") # Scaled data
    y.pred <- predict(fitted_model$fit, criteria="bic", newx=x.ts, type="class") # Non-Scaled data
    y.prob <- predict(fitted_model$fit, criteria="bic", newx=x.ts, type="response") # Non-Scaled data

    #TODO: Guardar las predicciones para sacar las métricas en un script a parte (¿en python?)
    write.csv(y.pred, str_c(PRED_PATH,"GLM_elnet-Ordinal-",DATASET,"-alpha=",alpha,"-pred_class.csv"), row.names=F, col.names=F) # Non-Scaled data
    write.csv(y.prob, str_c(PRED_PATH,"GLM_elnet-Ordinal-",DATASET,"-alpha=",alpha,"-pred_probs.csv"), row.names=F, col.names=F) # Non-Scaled data

    # Print metrics for current classifier and dataset
    metrics <- get_metrics(y.ts, y.pred)
    # Guardar las métricas por clase
    metrics_cls <- rbind(metrics_cls, cbind(model=model_nm, metrics$metrics_cls))
    # Guardar las métricas medias  
    metrics_avg <- rbind(metrics_avg, cbind(model_nm=model_nm, metrics$metrics_avg))
  }

  # write.csv(metrics_cls, str_c(LOG_PATH,"GLM_elnet_Ordinal-",DATASET,"-scaled-metrics_by_class.csv"), row.names=F) # Scaled data
  # write.csv(metrics_avg, str_c(LOG_PATH,"GLM_elnet_Ordinal-",DATASET,"-scaled-metrics_avg.csv"),      row.names=F) # Scaled data
  write.csv(metrics_cls, str_c(LOG_PATH,"GLM_elnet-Ordinal-",DATASET,"-metrics_by_class.csv"), row.names=F) # Non-Scaled data
  write.csv(metrics_avg, str_c(LOG_PATH,"GLM_elnet-Ordinal-",DATASET,"-metrics_avg.csv"),      row.names=F) # Non-Scaled data
}

if(is.numeric(dataset_num)) {
  # Guardar con el nombre del dataset ejecutado
  write.csv(exec_time_df, str_c(LOG_PATH, "elapsed_times-Ordinal-", DATASETS, ".csv"), row.names=F)
} else {
  # Guardar con nombre genérico
  write.csv(exec_time_df, str_c(LOG_PATH, "elapsed_times-Ordinal-ALL.csv"), row.names=F)
}

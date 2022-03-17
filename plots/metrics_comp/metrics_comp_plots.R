library(tidyverse)
library(ggplot2)
library(reshape2)

METRICS_PATH <- "Documents/D/Niebla/plots/metrics_comp/"

nominal_df <- read.csv(str_c(METRICS_PATH, "nominal_metrics.csv"), row.names=1)
glm_ord_df <- read.csv(str_c(METRICS_PATH, "GLM_ord_metrics.csv"), row.names=1)
glm_nom_df <- read.csv(str_c(METRICS_PATH, "GLM_metrics.csv"), row.names=1)

# Se va a probar primero a utilizar el dataset de clasificación nominal
nominal_df

# Guardar el nombre de la fila (ID) en el dataset
nominal_df$ID <- rownames(nominal_df)

# Crear un nuevo DF que sea para plotear
nominal_df_plot <- nominal_df
# Guardar los nombres de filas como ID y pasar a factor para que no se cambien de orden
# a la hora de plotear
nominal_df_plot$ID <- rownames(nominal_df)
nominal_df_plot$ID <- factor(nominal_df_plot$ID, levels = nominal_df_plot$ID)


# Crear una gráfica de barras con los valores de qwk, coloreados por Modelo usado
ggplot(nominal_df_plot, aes(x=ID, y=qwk, fill=Model)) +
  geom_bar(stat="identity") +
  theme(axis.text.x = element_text(angle = 45, hjust=1)) +
  coord_cartesian(ylim=c(0.75,0.95))


# TODO: Ordenar
# -> Sacar esta gráfica como la categórica de cada modelo como gráfica general
# Crear una gráfica de barras con los valores de F1_weighted, coloreados por Modelo usado
ggplot(nominal_df_plot, aes(x=ID, y=Fb_w, fill=Model)) +
  geom_bar(stat="identity") +
  theme(axis.text.x = element_text(angle = 45, hjust=1)) +
  coord_cartesian(ylim=c(0.5,0.83))
# TODO: Hacer otras dos gráficas como esta, una para GLM y otra GLM_ord


# Crear una gráfica de barras con los valores de qwk, coloreados por Dataset usado
ggplot(nominal_df_plot, aes(x=ID, y=qwk, fill=Dataset)) +
  geom_bar(stat="identity") +
  theme(axis.text.x = element_text(angle = 45, hjust=1)) +
  coord_cartesian(ylim=c(0.75,0.95))

# TODO: Poner las dos para la conclusión de que tiene más relación con el orden de resultados el modelo que el dataset

#  A simple vista se ve que es más importante el modelo usado que el dataset usado
# (hay más agrupación por colores en Model que en Dataset). También se ve cuales son los mejores modelos.
# Para comprobar fehacientemente esto se podrían hacer test de comparación de medias para ver si hay dif signf.

#  Hacer una comparación stacked con las métricas de precission y recall, para cada Model-Dataset
# de tipo así: https://stackoverflow.com/questions/6644997/showing-data-values-on-stacked-bar-chart-in-ggplot2

# TODO: Igual que se hizo antes, sacar de aquí los mejores de cada modelo, para hacer una gráfica con el mejor ed cada modelo

recall_plot_df <- nominal_df_plot %>%
  select(ID, contains("recall")) %>%
  melt(id.vars=c("ID"))

ggplot(recall_plot_df, aes(x=ID, y=value, fill=variable)) +
  geom_bar(stat = "identity") +
  theme(axis.text.x = element_text(angle = 45, hjust=1))

# El mismo plot, ordenado por la suma de recall
recall_plot_df_ <- recall_plot_df
recall_plot_df_$ID <- with(recall_plot_df, reorder(ID, -value, sum))
ggplot(recall_plot_df_, aes(x=ID, y=value, fill=variable)) +
  geom_bar(stat = "identity") +
  theme(axis.text.x = element_text(angle = 45, hjust=1))


# ---------------  

nominal_df_plot %>%
  select(ID, contains("precision")) %>%
  melt(id.vars=c("ID")) %>%
  ggplot(aes(x=ID, y=value, fill=variable)) +
  geom_bar(stat = "identity") +
  theme(axis.text.x = element_text(angle = 45, hjust=1))

nominal_df_plot %>%
  select(ID, contains("f1_s")) %>%
  melt(id.vars=c("ID")) %>%
  ggplot(aes(x=ID, y=value, fill=variable)) +
  geom_bar(stat = "identity") +
  theme(axis.text.x = element_text(angle = 45, hjust=1))


# -------------------------------------------
# TODO
# nominal_df$ID <- rownames(nominal_df)
# glm_ord_df %>%
  

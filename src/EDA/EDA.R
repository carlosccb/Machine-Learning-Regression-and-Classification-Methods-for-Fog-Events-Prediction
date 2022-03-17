library(tidyverse)
library(reshape2)

PLOT_PATH <- "~/Documents/D/Niebla/plots/"

df_tr <- read_csv("~/Documents/D/Niebla/Data/Treated_Data/Classification/HoldOut/Train.csv")

df_tr_by_class <- df_tr %>% 
  group_by(class)

df_tr_by_class_ <- cbind(df_tr_by_class, ID=1:dim(df_tr_by_class)[1])

df_tr_by_class_melted <- reshape2::melt(df_tr_by_class_, id="ID", values=colnames(df_tr_by_class))

# ggplot(df_tr_by_class_melted, aes(x=variable, y=value, fill=variable)) + 
#   geom_boxplot(data=filter(df_tr_by_class_melted, variable %in% c("hr","qpRa"))) +
#   facet_wrap(~variable,ncol = 4)

# ggplot(df_tr_by_class_melted, aes(x=variable, y=value, fill=variable)) + 
#   geom_boxplot(data=filter(df_tr_by_class_melted, variable %in% c("hr","qpRa"))) +
#   facet_wrap(~variable,ncol = 4)

# No consigo que esté cada variable en un cuadrante distinto
ggplot(df_tr_by_class_melted) + 
  geom_boxplot(aes(x=variable, y=value, fill=variable)) #+ #data=filter(df_tr_by_class_melted, variable %in% c("hr","qpRa"))) +
  #facet_wrap(~variable,ncol = 4)

# # Lo más cercano es esto, pero están todas las variables en las x de cada cuadrante
# ggplot(df_tr_by_class_melted) +
#   geom_boxplot(aes(x=variable, y=value, fill=variable)) + #data=filter(df_tr_by_class_melted, variable %in% c("hr","qpRa"))) +
#   facet_wrap(~variable,ncol = 4)
# 
# # O esto, donde el límite superior de los valores es el más alto de todo el dataset: 2000 de vs
# ggplot(df_tr_by_class_melted) +
#   geom_boxplot(aes(y=value, fill=variable)) + #data=filter(df_tr_by_class_melted, variable %in% c("hr","qpRa"))) +
#   facet_wrap(~variable)

# CONSEGUIDO!!
# Boxplot
ggplot(df_tr_by_class_melted) +
  geom_boxplot(aes(y=value, fill=variable)) + #data=filter(df_tr_by_class_melted, variable %in% c("hr","qpRa"))) +
  facet_wrap(~variable, scales = "free")
ggsave(filename="EDA_boxplot_variables.eps", device="eps", path=str_c(PLOT_PATH, "Dataset"), )

# Violinplot
ggplot(df_tr_by_class_melted) +
  geom_violin(aes(x=variable, y=value, fill=variable)) + #data=filter(df_tr_by_class_melted, variable %in% c("hr","qpRa"))) +
  facet_wrap(~variable, scales = "free")
ggsave(filename="EDA_violinplot_variables.eps", device="eps", path=str_c(PLOT_PATH, "Dataset"), )

# -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --

# Se podría comparar, utilizando los valores de vs, si el hacer under/over-sampling ha surtido efecto a la hora de balancear las clases junto con las variables

# Cargar los tres dataset a comparar, añadir una columna con el nombre del dataset que es y unir los tres dataset en uno nuevo df_comparison_datasets
df_tr_RUS <- read_csv("~/Documents/D/Niebla/Data/Treated_Data/Classification/HoldOut/Train_RUS.csv")
df_tr_RUS <- cbind(df_tr_RUS, dataset="RUS")

df_tr_RUS_SMT <- read_csv("~/Documents/D/Niebla/Data/Treated_Data/Classification/HoldOut/Train_RUS_SMT.csv")
df_tr_RUS_SMT <- cbind(df_tr_RUS_SMT, dataset="RUS_SMT")

df_tr_TL_SMT <- read_csv("~/Documents/D/Niebla/Data/Treated_Data/Classification/HoldOut/Train_TL_SMT.csv")
df_tr_TL_SMT <- cbind(df_tr_TL_SMT, dataset="TL_SMT")

df_tr_best <- read_csv("~/Documents/D/Niebla/Data/Treated_Data/Classification/HoldOut/Train_NCR_SMT.csv")
df_tr_best <- cbind(df_tr_best, dataset="NCR_SMT")

df_tr_og <- cbind(df_tr, dataset="OG")

df_comparison_datasets <- rbind(df_tr_og, df_tr_RUS, df_tr_RUS_SMT,  df_tr_TL_SMT, df_tr_best)

# Para conseguir que se puedan utilizar tanto las variables dataset como variable para elegir la x y el facet, respectivamente, se ponen sólo
# las variables de las que se quiere sacar el violinplot en measure.vars, y automáticamente se escogen como id.vars el resto, creando las columnas
# necesarias en el nuevo dataset
df_comparison_datasets_melted <- reshape2::melt(df_comparison_datasets, measure.vars=colnames(df_tr))

df_comparison_datasets_melted$dataset <- factor(df_comparison_datasets_melted$dataset, level=c("OG", "RUS", "RUS_SMT", "TL_SMT", "NCR_SMT"))

# Se puede comprobar que no hay prácticamente diferencias visibles entre las variables de los tres dataset, y las hay ligeramente en class
ggplot(df_comparison_datasets_melted) +
  geom_violin(aes(x=dataset, y=value, fill=dataset)) +
  facet_wrap(~variable, scales = "free")
ggsave(filename="EDA_violinplot_variables_comparison_datasets.eps", device="eps", path=str_c(PLOT_PATH, "Dataset"), )

# Comprobar la diferencia entre clases en los distintos dataset
foo <- df_comparison_datasets_melted %>% filter(variable=="class")

ggplot(foo) +
  geom_bar(aes(x=value, fill=dataset)) +
  facet_wrap(~dataset, ncol=5) +
  theme_bw() +
  theme(axis.text.x=element_blank(),
        axis.ticks.x=element_blank(),
        axis.title.x=element_blank()
        ) +
  scale_y_continuous(expand = expansion(add=c(0, 100)))
ggsave(filename="EDA_classes_distribution.eps", device="eps", path=str_c(PLOT_PATH, "Dataset"), )

# -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
#   =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  
# -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --s
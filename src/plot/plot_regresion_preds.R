library(tidyverse)
library(reshape2)

PLOT_PATH <- "~/Documents/D/Niebla/plots/"

# ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ##
#
# Cargar los datos de las predicciones
df_regression_pred <- read_csv("~/Documents/D/Niebla/log_files_old_struct/predictions/preds-temp_split_regr.csv")

# Añadir un ID para valor de test predicho
df_regression_pred_plot <- cbind(df_regression_pred, ID=1:dim(df_regression_pred)[1])

# -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
# Escoger los primeros 200 datos a predecir de test y plotear los distintos regresores
df_regression_pred_plot_melted <- reshape2::melt(df_regression_pred_plot[1:200,], id="ID", values=colnames(df_regression_pred))

# ggplot(df_regression_pred_plot_melted, aes(x=ID, y=value, color=variable)) +
#   geom_line()

ggplot(df_regression_pred_plot_melted, aes(x=ID, y=value, color=variable)) +
  geom_line() +
  ggtitle("200 first predictions vs round truth") +
  labs(x="Index", y="Visibility (m)") +
  theme(plot.title = element_text(hjust = 0.5))
ggsave(filename="200_first_predictions.eps", device="eps", path=str_c(PLOT_PATH,"Regression"), )

ggplot(df_regression_pred_plot_melted, aes(x=ID, y=value, color=variable, shape=variable)) +
  geom_line() +
  scale_shape_manual(values=1:nlevels(df_regression_pred_plot_melted$variable)) +
  geom_point() + #aes(shape=variable)) + 
  ggtitle("200 first predictions vs ground truth") +
  labs(x="Index", y="Visibility (m)") +
  theme(plot.title = element_text(hjust = 0.5))
ggsave(filename="200_first_predictions_points.eps", device="eps", path=str_c(PLOT_PATH,"Regression"), )
# -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --

# -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
# Esto son pruebas pero no quedan demasiado bien, así que estas no se guardan

# Por ahora solucionado: No se xk pero ayer funcionaba todo y ahora me da problemas con el melt (al que hay que poner reshape2::melt, y con el objeto que devuelve,
# que parece ser que es una lista en vez de un DF y da problemas al hacer el filter en geom_line

# Ordenar por los valores de y_true y plotear las predicciones
df_regression_pred_plot_sorted <- df_regression_pred %>% arrange(y_true)
df_regression_pred_plot_sorted <- cbind(df_regression_pred_plot_sorted, ID=1:dim(df_regression_pred)[1])  %>%
  select(ID, y_true, "LinearReg-temp_split_regr", "ElasticNet-temp_split_regr", "MLP_big-temp_split_regr")
df_regression_pred_plot_sorted_melted <- reshape2::melt(df_regression_pred_plot_sorted, id="ID", values=colnames(df_regression_pred))
df_regression_pred_plot_sorted_melted <- as_tibble(df_regression_pred_plot_sorted_melted)

# ggplot(df_regression_pred_plot_sorted_melted) +
#   geom_line(aes(x=ID, y=filter(value == "y_true"), color=variable))

ggplot(df_regression_pred_plot_sorted_melted) +
  geom_line(aes(x=ID, y=value, color=variable, alpha=0.1))

# Lo que quiero probar es hacer una gráfica donde: y_true sea una línea y las predicciones sean puntos
# Puede ser que para lo que quiero hacer haya que crear varios dataset
ggplot(df_regression_pred_plot_sorted, aes(x=ID)) +
  geom_line(aes(y=y_true)) + #, color=variable))
  geom_point(aes(x=ID, y=df_regression_pred_plot_sorted[["LinearReg-temp_split_regr"]], color="red")) 

ggplot(df_regression_pred_plot_sorted, aes(x=ID)) +
  geom_line(aes(y=y_true)) + #, color=variable))
  geom_point(aes(x=ID, y=df_regression_pred_plot_sorted[["LinearReg-temp_split_regr"]], color="red")) +
  geom_point(aes(x=ID, y=df_regression_pred_plot_sorted[["MLP_big-temp_split_regr"]], color="blue"))

ggplot(df_regression_pred_plot_sorted, aes(x=ID)) +
  geom_line(aes(y=y_true)) + #, color=variable))
  geom_point(aes(x=ID, y=df_regression_pred_plot_sorted[["MLP_big-temp_split_regr"]], color="blue", fill="blue"))

ggplot(df_regression_pred_plot_sorted, aes(x=ID)) +
  geom_line(aes(y=y_true)) + #, color=variable))
  geom_point(aes(y=`MLP_big-temp_split_regr`, color="red", alpha=0.4))

ggplot(df_regression_pred_plot_sorted_melted, aes(x=ID, y=value, color=variable)) +
  geom_line(data=filter(df_regression_pred_plot_sorted_melted, variable=="y_true")) + #, color=variable))
  geom_point(data=filter(df_regression_pred_plot_sorted_melted, variable!="y_true"), aes(alpha=0.2))


# Hacer la mísma gráfica con los valores umbralizados de clasificación ordenados como los de regresión
# Esta gráfica es con los datos de test
df_classes <- read_csv("~/Documents/D/Niebla/Data/Treated_Data/Classification/HoldOut/Test.csv")
data_classes <- as.data.frame(cbind(ID=1:dim(df_classes)[1],
                                    class=df_classes$class[order(df_classes$class)]))
ggplot(data_classes) +
  geom_line(aes(x=ID, y=class))

# Esta gráfica es con los valores de visibilidad umbralizados con el mismo criterio que se ha hecho en clasificación
# Para poder ver la comparación bien hay que hacer una gráfica con doble eje y (uno vs, otro class)
# Meter los valores de regresión y los correspondientes de clasificación en un DF
df_regr_vs_class <- df_regression_pred %>% select(y_true) %>% arrange(y_true)
colnames(df_regr_vs_class)[1] <- "vs"
df_regr_vs_class$class <- 4
df_regr_vs_class$n <- 1:dim(df_regr_vs_class)[1]

# Esto parece que no está funcionando y no se está haciendo correctamente la umbralización de los valores de visibilidad
df_regr_vs_class$class[df_regr_vs_class$vs < 40] <- 0
df_regr_vs_class$class[df_regr_vs_class$vs >= 40 && df_regr_vs_class$vs < 200 ] <- 1
df_regr_vs_class$class[df_regr_vs_class$vs >= 200 && df_regr_vs_class$vs < 500 ] <- 2
df_regr_vs_class$class[df_regr_vs_class$vs >= 500 && df_regr_vs_class$vs < 2000 ] <- 3

df_regr_vs_class_long <- reshape2::melt(df_regr_vs_class, measure.vars=c("vs","class")) %>% filter(variable %in% c("class", "n"))

# TODO: Aquí hace falta (buscar, saber cómo y) poner otro eje y con valores los discretos de clase que concuerden con los de y
ggplot(df_regr_vs_class_long, aes(x=n, y=value, color=variable)) +
  geom_line()

# -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --

# -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
# TODO: también se podría plotear la distancia (media) de cada predicción respecto al original

#
# -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
#
# ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ##


# ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ##
# TODO: Hacer gráficas con los valores de las métricas de prediccion en vez de con las predicciones en bruto
# Cargar los datos de las predicciones
library(data.table)
library(wesanderson)
# Carga datos
df_regression_metrcs <- read_csv("~/Documents/D/Niebla/log_files_old_struct/alg_comp/alg_comp-temp_split_regr.csv")

# Transponer los datos
df_regression_metrcs <- df_regression_metrcs %>% 
  gather(var, val, 2:ncol(df_regression_metrcs)) %>%
  spread(Metric, val) #%>%
  # TODO: eliminar el "-templ_split_regr" de los nombres
  #mutate(txt = str_replace(var, ".temp_split_regr", " "))

# Reemplazar la parte de la cadena que es siempre igual
df_regression_metrcs <- df_regression_metrcs  %>%
  mutate(var = str_sub(var, 1, nchar(var)-16))


# Para cambiar la paleta de colores: http://www.sthda.com/english/wiki/ggplot2-colors-how-to-change-colors-automatically-and-manually
# Para utilizar paletas que tienen menos colores que objetos a rellenar: https://stackoverflow.com/a/61162982

# Plotear las columnas con los valores de RMSE para cada modelo de predicción
ggplot(df_regression_metrcs, aes(x=factor(var, levels=var[order(RMSE)]), y=RMSE, fill=var)) +
  geom_col() +
  geom_text(aes(label = sprintf("%0.4f", round(RMSE, digits = 4))), vjust = -0.5) +
  ggtitle("RMSE comparison", ) +
  labs(x="Model", y="RMSE") +
  theme(axis.text.x = element_text(angle = 45, vjust = 1, hjust=1),
        plot.title = element_text(hjust = 0.5)) +
  scale_fill_manual(values = wes_palette(12, name = "GrandBudapest1", type = "continuous"), name = "")
  #scale_color_gradient(low="blue", high="green")
  #scale_fill_brewer(palette="Spectral")
ggsave(filename="RMSE_comparison.eps", device="eps", path=str_c(PLOT_PATH,"Regression"), )

# Plotear las columnas con los valores de MSE para cada modelo de predicción
ggplot(df_regression_metrcs, aes(x=factor(var, levels=var[order(MSE)]), y=MSE, fill=var)) +
  geom_col() +
  geom_text(aes(label = sprintf("%0.4f", round(MSE, digits = 4))), vjust = -0.5) +
  ggtitle("MSE comparison", ) +
  labs(x="Model", y="MSE") +
  theme(axis.text.x = element_text(angle = 45, vjust = 1, hjust=1),
        plot.title = element_text(hjust = 0.5))
ggsave(filename="MSE_comparison.eps", device="eps", path=str_c(PLOT_PATH,"Regression"), )

# Plotear las columnas con los valores de MAE para cada modelo de predicción
ggplot(df_regression_metrcs, aes(x=factor(var, levels=var[order(MAE)]), y=MAE, fill=var)) +
  geom_col() +
  geom_text(aes(label = sprintf("%0.4f", round(MAE, digits = 4))), vjust = -0.5) +
  ggtitle("MAE comparison", ) +
  labs(x="Model", y="MAE") +
  theme(axis.text.x = element_text(angle = 45, vjust = 1, hjust=1),
        plot.title = element_text(hjust = 0.5))
ggsave(filename="MAE_comparison.eps", device="eps", path=str_c(PLOT_PATH,"Regression"), )

# Plotear las columnas con los valores de R2 para cada modelo de predicción
ggplot(df_regression_metrcs, aes(x=factor(var, levels=var[order(R2,decreasing=T)]), y=R2, fill=var)) +
  geom_col() +
  geom_text(aes(label = sprintf("%0.4f", round(R2, digits = 4))), vjust = -0.5) +
  ggtitle("R2 comparison", ) +
  labs(x="Model", y="R2") +
  theme(axis.text.x = element_text(angle = 45, vjust = 1, hjust=1),
        plot.title = element_text(hjust = 0.5))
ggsave(filename="R2_comparison.eps", device="eps", path=str_c(PLOT_PATH,"Regression"), )

#
# ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ##

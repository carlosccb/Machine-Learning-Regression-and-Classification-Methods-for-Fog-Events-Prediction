library(tidyverse)

PLOT_PATH <- "~/Documents/D/Niebla/plots/"

# Leer fichero
df <- read_csv("./Documents/D/Niebla/log_files/alg_comp/classification/ordinal/Complete_by_class_Ordinal_comparison.csv")

#Seleccionar variables que interesan, cambiar nombre al ID y comprobar datos
data <- df[c("X1", "Dataset", "alpha", "qwk")]
colnames(data) <- c("ID", "Dataset", "alpha", "qwk")
data

# -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
# Comprobación inicial de los datos: ver cuál es la relación entre valores de alpha y qwk
gg_data <- gather(data, key="ID")

ggplot(data, aes(x=alpha, y=qwk, color=Dataset)) +
  geom_point() +
  geom_line() +
  ggtitle("QWK for each alpha value by dataset") +
  theme(plot.title = element_text(hjust = 0.5))
ggsave(filename="alpha_vs_QWK_by_dataset.eps", device="eps", path=str_c(PLOT_PATH, "Classification"), )

# ggplot(data) +
#   geom_point(aes(x=alpha, y=qwk, color=Dataset)) +
#   geom_line(aes(x=alpha, y=qwk, color=Dataset))

sorted_data <- arrange(data, alpha)

ggplot(sorted_data, aes(alpha, qwk, color=Dataset)) +
  geom_path(aes(group = 1)) +
  geom_point()

ggplot(sorted_data, aes(qwk, alpha, color=Dataset)) +
  geom_path(aes(group = 1)) +
  geom_point()

ggplot(sorted_data, aes(qwk, alpha, color=Dataset)) +
  geom_point()
ggsave(filename="alpha_vs_QWK_by_dataset_scatter.eps", device="eps", path=str_c(PLOT_PATH, "Classification"), )
#
# -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --


# -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
# Para cada dataset probado, el mejor resultado obtenido (siempre es cuando alpha=1)
best_alpha_data <- data %>%
  group_by(Dataset) %>%
  top_n(1, qwk) %>%
  arrange(qwk)

ggplot(best_alpha_data, aes(factor(ID, levels=ID[order(best_alpha_data$qwk)]), qwk, fill=Dataset)) +
  geom_col(width = 0.7) +
  geom_text(aes(label = sprintf("%0.4f", round(qwk, digits = 4))), vjust = -0.5) +
  ggtitle("Best qwk by dataset", ) +
  labs(x="Dataset-alpha", y="QWK") +
  theme(axis.text.x = element_text(angle = 45, vjust = 1, hjust=1),
        plot.title = element_text(hjust = 0.5))
ggsave(filename="Best_QWK_by_dataset.eps", device="eps", path=str_c(PLOT_PATH, "Classification"), )

# Para todas las combinaciones probadas, los 10 mejores resultados
best_data <- data %>%
  #group_by(Dataset) %>%
  top_n(15, qwk) %>%
  arrange(qwk)

# Aquí hay que cambiar el tamaño de la fuente que se imprime encima de las barras porque se apelotona
ggplot(best_data, aes(factor(ID, levels=ID[order(qwk)]), qwk, fill=Dataset)) +
  geom_col(width = 0.7) +
  geom_text(aes(label = sprintf("%0.4f", round(qwk, digits = 4))), vjust = -0.5) +
  ggtitle("Best 15 qwk", ) +
  labs(x="Dataset-alpha", y="QWK") +
  theme(axis.text.x = element_text(angle = 45, vjust = 1, hjust=1),
        plot.title = element_text(hjust = 0.5))
ggsave(filename="Best_15_QWK.eps", device="eps", path=str_c(PLOT_PATH, "Classification"), )

best_data <- data %>%
  #group_by(Dataset) %>%
  top_n(10, qwk) %>%
  arrange(qwk)

ggplot(best_data, aes(factor(ID, levels=ID[order(qwk)]), qwk, fill=Dataset)) +
  geom_col(width = 0.7) +
  geom_text(aes(label = sprintf("%0.4f", round(qwk, digits = 4))), vjust = -0.5) +
  ggtitle("Best 10 qwk", ) +
  labs(x="Dataset-alpha", y="QWK") +
  theme(axis.text.x = element_text(angle = 45, vjust = 1, hjust=1),
        plot.title = element_text(hjust = 0.5))
ggsave(filename="Best_10_QWK.eps", device="eps", path=str_c(PLOT_PATH, "Classification"), )


#
# -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --

# -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
test_data <- best_data[c("ID", "qwk")]

# TODO: Ahora quedaría hacer comparación de medias (test estadísticos) para ver cual (o si hay) datasets que sean mejores a otros
# todos con su mejor valor de qwk
# Aquí  habría que hacer un rank test para comparar si hay diferencias entre los distintos datasets para el mismo algoritmo

?wilcox.test()

# TODO: Posteriormente se podría hacer la comparación entre hacerlo ordinal vs nominal
# Aquí habría de hacer un test de comparación de medias (creo), un t-test, porque se comparan las medias de los dos
# métodos (ordinal vs nominal) para ver si hay diferencias significativas entre ambos métodos de clasificación

#
# -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
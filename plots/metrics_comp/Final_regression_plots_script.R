library(tidyverse)
library(ggplot2)
library(reshape2)

# Set Variables
ROOT = "~/Documents/D/Niebla/"
# PATH donde están las predicciones
PREDS_PATH = str_c(ROOT, "log_files_old_struct/predictions/")
# PATH donde guardar las gráficas
METRICS_PATH <- "~/Documents/D/Niebla/plots/metrics_comp/"

# Load files
df_1 <- read.csv(str_c(PREDS_PATH,"preds-temp_split_regr.csv"))
df_2 <- read.csv(str_c(PREDS_PATH,"preds-temp_split_regr-min_max.csv"))

df <- as_tibble(cbind(x=0:(dim(df_2)[1]-1),
                      GroundTruth=df_1$y_true,
                      LinearRegression=df_1$LinearReg.temp_split_regr,
                      RF=df_2$RF.temp_split_regr.min_max,
                      MLP_big=df_1$MLP_big.temp_split_regr))

# Para comprobar que son las que se muestran en las tablas
RMSE <- function(y_true, y_pred) {sqrt(mean((y_true - y_pred)**2))}
print(str_c("LinReg: ", RMSE(df$GroundTruth, df$LinearRegression)))
print(str_c("RF: ", RMSE(df$GroundTruth, df$RF)))
print(str_c("MLP_big: ", RMSE(df$GroundTruth, df$MLP_big)))

# === === === === === === === === === === === === === === === === === === === === === === === === === === === === === === === === === === ===
# La gráfica general con todos los métodos escogidos
df_ <- df[1:250,] %>%
  gather(key = "variable", value = "value", -x)

# Sacar la gráfica general con todos los modelos
ggplot(df_, aes(x=x, y=value, color=variable)) +
  geom_line(lwd=1) +
  labs(x="", y="Visibility") +
  theme_minimal() +
  scale_x_continuous(expand = c(0, 2)) +
  scale_y_continuous(expand = expansion(add=c(0,25))) +
  guides(color=guide_legend(title="")) +
  theme(legend.position="top",
        legend.text=element_text(size=20),
        axis.title=element_text(size=20),
        axis.text.y=element_text(size=17),
        axis.text.x=element_blank(),
        axis.ticks.x=element_blank())
ggsave(file="Regression-CompleteComparison-thick_l_v4.eps", device="eps", path=METRICS_PATH, scale=1.5, dpi=320, units="px")#, width=1290, height=1080)
# === === === === === === === === === === === === === === === === === === === === === === === === === === === === === === === === === === ===

# === === === === === === === === === === === === === === === === === === === === === === === === === === === === === === === === === === ===
# Sacar las gráficas "particulares" (de cada método)

# Reg Lineal
df[1:250,] %>%
  select(x, GroundTruth, LinearRegression) %>%
  gather(key = "variable", value = "value", -x) %>%
  ggplot(aes(x=x, y=value, color=variable)) +
    geom_line(lwd=1) +
    labs(x="", y="Visibility") +
    theme_bw() +
    scale_x_continuous(expand = c(0, 2)) +
    scale_y_continuous(expand = c(0, 10)) +
    guides(color=guide_legend(title="")) +
    theme(legend.position="top",
          legend.text=element_text(size=12),
          axis.text.x=element_blank(),
          axis.ticks.x=element_blank()) +
  scale_color_brewer(palette="Set1")
ggsave(file="Regression-LinearRegr_v4.eps", device="eps", path=METRICS_PATH, scale=1.5, dpi=320, units="px")#, width=1290, height=1080)


# Random Forest
df[1:250,] %>%
  select(x, GroundTruth, RF) %>%
  gather(key = "variable", value = "value", -x) %>%
  ggplot(aes(x=x, y=value, color=variable)) +
    geom_line(lwd=1) +
    theme_bw() +
    scale_x_continuous(expand = c(0, 2)) +
    scale_y_continuous(expand = c(0, 10)) +
    labs(x="", y="Visibility") +
    guides(color=guide_legend(title="")) +
    theme(legend.position="top",
          legend.text=element_text(size=12),
          axis.text.x=element_blank(),
          axis.ticks.x=element_blank()) +
    scale_color_brewer(palette="Set1")
ggsave(file="Regression-RF_v4.eps", device="eps", path=METRICS_PATH, scale=1.5, dpi=320, units="px")#, width=1290, height=1080)


# MLP big
df[1:250,] %>%
  select(x, GroundTruth, MLP_big) %>%
  gather(key = "variable", value = "value", -x) %>%
  ggplot(aes(x=x, y=value, color=variable)) +
    geom_line(lwd=1) +
    labs(x="", y="Visibility") +
    theme_bw() +
    scale_x_continuous(expand = c(0, 2)) +
    scale_y_continuous(expand = c(0, 10)) +
    guides(color=guide_legend(title="")) +
    theme(legend.position="top",
          legend.text=element_text(size=12),
          axis.text.x=element_blank(),
          axis.ticks.x=element_blank()) +
    scale_color_brewer(palette="Set1")
ggsave(file="Regression-MLP_big_v4.eps", device="eps", path=METRICS_PATH, scale=1.5, dpi=320, units="px")#, width=1290, height=1080)
# === === === === === === === === === === === === === === === === === === === === === === === === === === === === === === === === === === ===
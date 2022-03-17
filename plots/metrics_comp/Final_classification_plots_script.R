library(tidyverse)
library(ggplot2)
library(ggsci)
library(reshape2)

METRICS_PATH <- "~/Documents/D/Niebla/plots/metrics_comp/"

nominal_df <- read.csv(str_c(METRICS_PATH, "nominal_metrics.csv"), row.names=1)
glm_ord_df <- read.csv(str_c(METRICS_PATH, "GLM_ord_metrics.csv"), row.names=1)
glm_nom_df <- read.csv(str_c(METRICS_PATH, "GLM_metrics.csv"), row.names=1)

# Guardar el nombre de la fila (ID) en el dataset
# nominal_df <- nominal_df[order(-nominal_df$F1_w),]
nominal_df$ID <- rownames(nominal_df)
nominal_df$ID <- factor(nominal_df$ID, levels = nominal_df$ID)
# Quitar "Classifier" de los nomobres
nominal_df <- nominal_df %>%
  mutate(Model=str_replace(Model, "Classifier", ""))

# glm_nom_df <- glm_nom_df[order(-glm_nom_df$F1_w),]
glm_nom_df$ID <- rownames(glm_nom_df)
glm_nom_df$ID <- factor(glm_nom_df$ID, levels = glm_nom_df$ID)

# glm_ord_df <- glm_ord_df[order(-glm_ord_df$F1_w),]
glm_ord_df$ID <- rownames(glm_ord_df)
glm_ord_df$ID <- factor(glm_ord_df$ID, levels = glm_ord_df$ID)
# ----------------------------------------------------------------------------------

# ===============================================================================================================
# 1. Hacer las gráficas (separadas) con F1-score de R (colorines) **con todos los modelos** para los csv de:

pub_fig_theme <- function() {
  theme(legend.position="top",
        axis.text.x=element_blank(),
        axis.ticks.x=element_blank(),
        #axis.text.x = element_text(angle = 45, hjust=1, size=6)
        axis.text.y=element_text(size=15),
        axis.title.y=element_text(size=20),
        legend.title=element_text(size=21),
        legend.text=element_text(size=16),
        )
}

#   1. sklearn nominal
summary(nominal_df$F1_w)
# TODO: Ver cómo exportar estas gráficas para que se vean bien (intentar en: eps, svg o png)
ggplot(nominal_df, aes(x=ID, y=F1_w, fill=Model)) +
  geom_bar(stat="identity") +
  theme_classic() +
  coord_cartesian(ylim=c(0.45,0.83)) +
  labs(x="", y="F1") +
  pub_fig_theme() +
  geom_text(aes(label=Dataset, y=0.433), hjust=1, angle=-90, size=3)

  # Poner el nombre del dataset dentro de los bars
  #geom_text(aes(label=Dataset, y=0.433), hjust = 0, angle=90, size=2.5)
  #TODO: ver si se queda mejor este esquema de colores # + scale_fill_nejm("default", 0.8)
ggsave(file="Complete_comp-sk_nom-F1_v4.1.eps", device="eps", path=METRICS_PATH, scale=1.25, dpi=320, units="px")#, width=1290, height=1080)
#ggsave(file="name.eps")

# Plot alpha values in x axis
pub_fig_theme_ <- function() {
  theme(legend.position="top",
        #axis.ticks.x=element_blank(),
        axis.text.x=element_text(angle = -90, hjust=0, vjust = 0.5), #element_blank(), #
        axis.title.y=element_text(size=20, angle = 0, hjust=0, vjust = 0.5),
        #axis.text.x = element_text(angle = 45, hjust=1, size=6),
        axis.text.y=element_text(size=15),
        legend.title=element_text(size=21),
        legend.text=element_text(size=16),
      )
}

# Plot alpha values inside bar
pub_fig_theme_ <- function() {
  theme(legend.position="top",
        axis.ticks.x=element_blank(),
        axis.text.x=element_blank(),
        axis.title.y=element_text(size=20, angle = 0, hjust=0, vjust = 0.5),
        #axis.text.x = element_text(angle = 45, hjust=1, size=6),
        axis.text.y=element_text(size=15),
        legend.title=element_text(size=21),
        legend.text=element_text(size=16),
  )
}

#   2. GLM
summary(glm_nom_df$F1_w)
ggplot(glm_nom_df, aes(x=ID, y=F1_w, fill=Dataset)) +
  geom_bar(stat="identity") +
  theme_classic() +
  coord_cartesian(ylim=c(0.45,0.83)) +
  labs(x="", y="F1") +
  pub_fig_theme_() +
  # Plot alpha values in x axis
  #scale_x_discrete(labels=as.character(glm_nom_df$alpha))
  # Plot alpha values inside bar
  geom_text(aes(label=alpha, y=0.433), hjust = 1, angle=-90, size=2.5)
ggsave(file="Complete_comp-glm_nom-F1_v4.eps", device="eps", path=METRICS_PATH, scale=1.5, dpi=320, units="px")#, width=1290, height=1080)

#   3. GLM ordinal
summary(glm_ord_df$F1_w)
ggplot(glm_ord_df, aes(x=ID, y=F1_w, fill=Dataset)) +
  geom_bar(stat="identity") +
  theme_classic() +
  coord_cartesian(ylim=c(0.45,0.83)) +
  labs(x="", y="F1") +
  pub_fig_theme_() +
  # Plot alpha values in x axis
  #scale_x_discrete(labels=as.character(glm_ord_df$alpha)) +
  # Plot alpha values inside bar
  geom_text(aes(label=alpha, y=0.433), hjust = 1, angle=-90, size=2.5)
ggsave(file="Complete_comp-GLM_ord-F1_v4.eps", device="eps", path=METRICS_PATH, scale=1.5, dpi=320, units="px")#, width=1290, height=1080)
#ggsave(file="name.eps")

# ---
# Intento de mostrar el valor de alpha con una gráfica de dos ejes y. Se queda muy mal

# ggplot(glm_nom_df, aes(x=ID, y=F1_w, fill=Dataset)) +
#   geom_bar(stat="identity", alpha=0.8) +
#   # geom_point(aes(x=ID, y=F1_w, color=Dataset), alpha=0.8) +
#   # geom_line(aes(x=ID, y=F1_w, color=Dataset, group=1), alpha=0.8) +
#   theme_classic() +
#   coord_cartesian(ylim=c(0.45,0.83)) +
#   labs(x="", y="F1") +
#   pub_fig_theme_() +
#   scale_y_continuous(
#     "F1", 
#     sec.axis = sec_axis( ~(.-0.432)/0.37, name = "alpha")
#   ) +
#   geom_point(aes(x=ID, y=((alpha*0.37)+0.432)), size=1) +
#   geom_line(aes(x=ID,  y=((alpha*0.37)+0.432), group = 2), alpha=0.7)
# ---

# ===============================================================================================================

# ===============================================================================================================
best_overall_df <- read.csv(str_c(METRICS_PATH, "best_overall_classif.csv"), row.names=1)
best_overall_df$ID <- rownames(best_overall_df)
best_overall_df$ID <- factor(best_overall_df$ID, levels = best_overall_df$ID)

# Quitar el "Classifier" de Model
best_overall_df <- best_overall_df %>%
  mutate(Model=str_replace(best_overall_df$Model, "Classifier", ""))

# 2. Sacar los mejores de GLM nom y ord, y meter en la misma gráfica que los **mejores por modelo** de nominal 
#  (como la gráfica *nom_res_df*), es decir, añadiendo las dos barras de GLM. Sacar esto para las métricas:

pub_fig_text_size <- function() {
  theme(axis.text.x=element_text(angle = 45, hjust=1, size=15),
        axis.text.y=element_text(size=15),
        axis.title.y=element_text(size=20),
        legend.title=element_text(size=22),
        legend.text=element_text(size=17),)
}

#  1. F1-score
summary(best_overall_df$F1_w)
ggplot(best_overall_df, aes(x=ID, y=F1_w, fill=Model)) + # TODO: Poner los ShortName en la leyenda^
  geom_bar(stat="identity") +
  theme_classic() +
  pub_fig_text_size() +
  coord_cartesian(ylim=c(0.45,0.83)) +
  labs(x="", y="F1") +
  theme(legend.position="top") +
  scale_x_discrete(labels = str_c("[",best_overall_df$Dataset,"]"))
ggsave(file="Best_overall-F1_v4.eps", device="eps", path=METRICS_PATH, scale=1.5, dpi=320, units="px")#, width=1290, height=1080)

#  2. QWK
summary(best_overall_df$qwk)
ggplot(best_overall_df, aes(x=ID, y=qwk, fill=Model)) +
  geom_bar(stat="identity") +
  theme_classic() +
  pub_fig_text_size() +
  coord_cartesian(ylim=c(0.85,0.95)) +
  labs(x="", y="QWK") +
  theme(legend.position="top") +
  scale_x_discrete(labels = str_c("[",best_overall_df$Dataset,"]"))
ggsave(file="Best_overall-QWK_v4.eps", device="eps", path=METRICS_PATH, scale=1.5, dpi=320, units="px")#, width=1290, height=1080)

#  3. Accuracy
summary(best_overall_df$acc)
ggplot(best_overall_df, aes(x=ID, y=acc, fill=Model)) +
  geom_bar(stat="identity") +
  theme_classic() +
  pub_fig_text_size() +
  coord_cartesian(ylim=c(0.70,0.83)) +
  labs(x="", y="Accuracy") +
  theme(legend.position="top") +
  scale_x_discrete(labels = str_c("[",best_overall_df$Dataset,"]"))
ggsave(file="Best_overall-Acc_v4.eps", device="eps", path=METRICS_PATH, scale=1.5, dpi=320, units="px")#, width=1290, height=1080)

# === === === === === === === === === === === === === === === === === === === === === === === === === === === === === === === === === === === ===

pub_fig_text_size_comp_sum <- function() {
  theme(axis.text.x=element_text(angle = 45, hjust=1, size=18),
        axis.text.y=element_text(size=20),
        axis.title.y=element_text(size=28),
        legend.title=element_text(size=28),
        legend.text=element_text(size=25),)
}

#  4. Y la suma del recall por clase (la gráfica stack plot de R) (sacando los mejores de cada clasificador) (con 10 barras que serían los 10 clasificadores) (podrían ser 9 si se quita la persistencia)
# TODO: Ordenar
best_overall_df %>%
  select(ID, contains("recall")) %>%
  melt(id.vars=c("ID")) %>%
  ggplot(aes(x=ID, y=value, fill=variable)) +
  geom_bar(stat = "identity") +
  theme_classic() +
  pub_fig_text_size_comp_sum() +
  labs(x="", y="Sum of Recall") +
  guides(fill=guide_legend(title="Metric")) +
  theme(legend.position="top") +
  scale_y_continuous(expand = c(0, 0)) +
  scale_x_discrete(labels = str_c(str_replace(best_overall_df$Model, "Classifier", ""),
                                  "\n[",
                                  best_overall_df$Dataset,
                                  "]"),)
  
  #scale_x_discrete(labels = str_c(best_overall_df$ShortName,"\n[",best_overall_df$Dataset,"]"),)
ggsave(file="Best_overall-SumRecall_v4.eps", device="eps", path=METRICS_PATH, scale=1.5, dpi=320)#, units="px")#, width=1290, height=1080)

best_overall_df %>% select()

#  5. Poner la misma anterior como F1-score

colnames(best_overall_df) <- gsub("_score_", "_", colnames(best_overall_df))

# TODO: Ordenar
best_overall_df %>%
  select(ID, contains("f1_"), -contains("F1_w")) %>%
  melt(id.vars=c("ID")) %>%
  ggplot(aes(x=ID, y=value, fill=variable)) +
  geom_bar(stat = "identity") +
  theme_classic() +
  pub_fig_text_size_comp_sum() +
  labs(x="", y="Sum of F1") +
  guides(fill=guide_legend(title="Metric")) +
  theme(legend.position="top") +
  scale_y_continuous(expand = c(0, 0)) +
  scale_x_discrete(labels = str_c(str_replace(best_overall_df$Model, "Classifier", ""),
                                  "\n[",
                                  best_overall_df$Dataset,
                                  "]"),)
ggsave(file="Best_overall-SumF1_v4.eps", device="eps", path=METRICS_PATH, scale=1.5, dpi=320)#, units="px")#, width=1290, height=1080)
#ggsave(file="name.eps")
# ===============================================================================================================
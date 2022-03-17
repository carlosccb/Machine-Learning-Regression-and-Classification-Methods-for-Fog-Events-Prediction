library(tidyverse)
#library(ggsci)

DATASETS_PATH <- "~/Documents/D/Niebla/Data/Treated_Data/Classification/HoldOut/"

classes <- c("0","1","2","3","4")

# Para que se cargen bien los tipos de datos por columna y se puedan visualizar, etc... utilizar esto
read_file_ <- function(filename) {
  # Esto funciona en macOS, pero en Linux no
  #read_csv(str_c(DATASETS_PATH, filename), col_types=cols(class=col_factor(levels=0:4)))
  # Esto fuciona en Linux, en macOS no se si funcionará (los niveles tienen que ser string)
  read_csv(str_c(DATASETS_PATH, filename), col_types=cols(class=col_factor(levels=classes)))
}

IMB_PLOT_PATH <- "~/Documents/D/Niebla/plots/EDA"

pub_fig_theme_big <- function() {
  theme(axis.text.y = element_text(size=18),
        #axis.text.x = element_text(size=17),
        axis.title=element_text(size=22),
        legend.title=element_text(size=22),
        legend.text=element_text(size=20),
        # Para quitar todo lo del eje x 
        axis.title.x=element_blank(),
        axis.text.x=element_blank(),
        axis.ticks.x=element_blank(),
        #legend.position="none"
  )
}

pub_fig_theme_small <- function() {
  theme(axis.text.y = element_text(size=18),
        #axis.text.x = element_text(size=17),
        axis.title=element_blank(), #element_text(size=22),
        legend.title=element_blank(), #element_text(size=22),
        legend.text=element_blank(), #element_text(size=20),
        # Para quitar todo lo del eje x 
        axis.title.x=element_blank(),
        axis.text.x=element_blank(),
        axis.ticks.x=element_blank(),
        legend.position="none"
  )
}

# Función para visualizar la distribución de clases para cada fichero
visualize_class_distr_ <- function(df, plot_name, PATH) {
  df %>%
    select(class) %>%
    ggplot(aes(x=class, fill=class)) +
    geom_bar(stat = "count") +  #theme_classic() +
    theme_bw() +
    coord_cartesian(ylim=c(0,16400)) +
    labs(x="", y="Class count") +
    theme(legend.position="top") +
    guides(fill=guide_legend(title="Class")) +
    scale_y_continuous(expand = c(0, 0)) +
    pub_fig_theme_small() #+
    #scale_fill_nejm("default", 0.8)
    #scale_x_continuous(expand = c(0, 0))
  ggsave(file=plot_name, device="eps", path=PATH, scale=1.5, dpi=320)#, units="px")#, width=1290, height=1080)
}

# TODO: WARNING: Lo suyo sería poner en una tabla la distribución de clases para cada fichero
# y luego poner algunas figuras (3 ò 4) que sean ilustrativas

# Comparación general de la distribución de todas las clases para todos los datasets
tr_df <- read_file_("Train.csv")
visualize_class_distr_(tr_df, "imb_plot_OG_v4_small.eps", IMB_PLOT_PATH)

smt_df <- read_file_("Train_SMT.csv")
visualize_class_distr_(smt_df, "imb_plot_SMT_v4.eps", IMB_PLOT_PATH)

ncr_df <- read_file_("Train_NCR.csv")
visualize_class_distr_(ncr_df, "imb_plot_NCR_v4.eps", IMB_PLOT_PATH)

ncr_smt_df <- read_file_("Train_NCR_SMT.csv")
visualize_class_distr_(ncr_smt_df, "imb_plot_NCR_SMT_v4.eps", IMB_PLOT_PATH)

cnn_df <- read_file_("Train_CNN.csv")
visualize_class_distr_(cnn_df, "imb_plot_CNN_v4.eps", IMB_PLOT_PATH)

# cnn_smt_df <- read_file_("Train_CNN_SMT.csv")
# visualize_class_distr_(cnn_smt_df, "imb_plot_CNN_SMT_v4.eps", IMB_PLOT_PATH)

tl_df <- read_file_("Train_TL.csv")
visualize_class_distr_(tl_df, "imb_plot_TL_v4.eps", IMB_PLOT_PATH)

tl_smt_df <- read_file_("Train_TL_SMT.csv")
visualize_class_distr_(tl_smt_df, "imb_plot_TL_SMT_v4.eps", IMB_PLOT_PATH)

rus_df <- read_file_("Train_RUS.csv")
visualize_class_distr_(rus_df, "imb_plot_RUS_v4.eps", IMB_PLOT_PATH)

rus_smt_df <- read_file_("Train_RUS_SMT.csv")
visualize_class_distr_(rus_smt_df, "imb_plot_RUS_SMT_v4.eps", IMB_PLOT_PATH)


# Comparación del Undersampling para la clase mayoritaria en todos los dataset cargados
undersampl_comparison_df <- c("OG" =sum(tr_df$class==4),
                              "RUS"=sum(rus_df$class==4),
                              "NCR"=sum(ncr_df$class==4),
                              "CNN"=sum(cnn_df$class==4),
                              "TL" =sum(tl_df$class==4))
undersampl_comparison_df <- as.data.frame(undersampl_comparison_df)
colnames(undersampl_comparison_df) <- "Class count"
undersampl_comparison_df <- rownames_to_column(undersampl_comparison_df, "Method")

# Poner el nombre de la fila como variable para cuando se hace el gather y ponerlo como fill en el plot
# TODO
undersampl_comparison_df %>%
  gather(key = "Method", value =`Class count`) %>%
  ggplot(aes(x=Method, y=`Class count`, fill=Method)) +
    geom_bar(stat="identity")

# ggplot(undersampl_comparison_df, aes(x=Method, y=`Class count`, fill=Method)) +
#   geom_line()

barplot(undersampl_comparison_df$`Class count`, )

#  ===============  ===============  ===============  ===============  ===============  ===============  ===============  ===============
# ====  =====  ==== ====  =====  ==== ====  =====  ==== ====  =====  ==== ====  =====  ==== ====  =====  ==== ====  =====  ==== ====  =====
#  ===============  ===============  ===============  ===============  ===============  ===============  ===============  ===============
# ====  =====  ==== ====  =====  ==== ====  =====  ==== ====  =====  ==== ====  =====  ==== ====  =====  ==== ====  =====  ==== ====  =====
#  ===============  ===============  ===============  ===============  ===============  ===============  ===============  ===============

NEW_DATA_PATH = "~/Documents/D/Niebla/Data/Treated_Data/Classification/"
new_df <- read_csv(str_c(NEW_DATA_PATH,"OG_data_unmodified.csv"),
                   col_types=cols(class=col_factor(levels=classes))) # en macOS: 0:4

# TODO: hacer esto mismo con el resto de ficheros de clasificación de train

aux <- new_df[order(new_df$vs),]
aux$ID <- 1:nrow(aux)

umbrales_clases <- c("0"=0, "1"=40, "2"=200,
                     "3"=1000, "4"=2000)
umbrales_clases_df <- as.data.frame(umbrales_clases)
umbrales_clases_df <- rownames_to_column(umbrales_clases_df, "Class")

PAPER_STUFF <- "~/Documents/D/Niebla/plots/paper_stuff/"

# Plot con los intervalos
ggplot(aux, aes(x=ID, y=vs, color=class)) +
  #geom_point() +
  geom_line(size = 4) +
  labs(x="Samples", y="Visibility") +
  theme_bw() +
  theme(legend.position="top") +
  guides(color=guide_legend(title="Class")) +
  theme(axis.title.y=element_text(size=20),
        axis.title.x=element_text(size=20),
        axis.text.y=element_text(size=16),
        axis.text.x=element_text(size=16),
        legend.title=element_text(size=22),
        legend.text=element_text(size=20),) +
  scale_x_continuous(expand = c(0,100)) +
  scale_y_continuous(expand = c(0,25))
ggsave(file="Visibility_colored_by_class_v4.eps", device="eps", path=PAPER_STUFF, scale=1.5, dpi=320)#, units="px")#, width=1290, height=1080)
#coord_cartesian(ylim=c(0,16400)) +

# #Este se queda más feo; source: https://stackoverflow.com/a/45026110
# ggplot(aux, aes(x=ID, y=vs, color=class)) +
#   geom_point() +
#   geom_hline(data=umbrales_clases_df, aes(yintercept=umbrales_clases, color=as.factor(Class)))

# tr_df[order(tr_df$vs),] %>%
#   select(vs, class) %>%
#   ggplot(aes(x=vs, y=vs, color=class)) +
#   geom_point()

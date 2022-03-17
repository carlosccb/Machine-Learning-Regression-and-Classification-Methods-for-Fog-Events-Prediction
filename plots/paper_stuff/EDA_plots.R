# Here are included some EDA plots, others are included in other files, because they were thought in that context and were made in that script

library(tidyverse)
library(corrplot)
library(reshape2)

# Variables necesarias para cargar los datos
DATASETS_PATH <- "~/Documents/D/Niebla/Data/Treated_Data/Classification/HoldOut/"
classes <- c("0","1","2","3","4")

PLOT_PATH <- "~/Documents/D/Niebla/plots/EDA/"

# === === === === === === === === === === === === === === === === === === === ===
# Cargar los datos del csv
#
tr_df <- read_csv(str_c(DATASETS_PATH, "Train.csv"),
                  col_types=cols(class=col_factor(levels=classes)))

# Dejar como id.vars el nombre de la columna y la clase
molten_tr_df <- reshape2::melt(tr_df[1:100,], measure.vars=colnames(tr_df)[1:ncol(tr_df)-1])

# Violin plot para cada variable por clase
# ggplot(tr_df, aes(x=class, y=vs, fill=class)) +
#   geom_violin()

# Esto también se puede hacer añaidendo dots inside the violinplot, Hay varias opciones, dejo las fuentes: 
# http://www.sthda.com/english/wiki/ggplot2-violin-plot-quick-start-guide-r-software-and-data-visualization#violin-plot-with-dots
# https://r-charts.com/distribution/violin-plot-points-ggplot2/

ggplot(molten_tr_df, aes(x=variable, y=value, fill=class),) +
  geom_violin(position=position_dodge()) +#, draw_quantiles=c(0.5)) +
  geom_boxplot(aes(x=variable, y=value, fill=class), width=0.1, color="black",position=position_dodge(width =0.9)) +
  facet_wrap(~variable, scales = "free") +
  theme_bw() +
  theme(legend.position="top",
        legend.title=element_text(size=22),
        legend.text=element_text(size=20),
        axis.title.y=element_text(size=18),
        axis.text.y=element_text(size=15),
        axis.text.x=element_blank(),
        axis.ticks.x=element_blank(),
        #legend.title="Class",
        strip.text.x = element_text(size = 18)) +
  scale_fill_discrete(name = "Class") +
  labs(x="", y="Values")
ggsave(file="Violin_plot_by_class_and_variables_v4.eps", device="eps", path=PLOT_PATH, scale=1.5, dpi=320)#, units="px")#, width=1290, height=1080)

# Add "break" in axis
#https://stackoverflow.com/a/56000944

#
#
# === === === === === === === === === === === === === === === === === === === ===


# === === === === === === === === === === === === === === === === === === === ===
# Gráfica de corrplot con los colores por correlación
#

# SOURCE: http://www.sthda.com/english/wiki/visualize-correlation-matrix-using-correlogram

#Sacar la matriz de correlaciones
M_corr <- cor(tr_df[,-ncol(tr_df)])
# Hacer la paleta de colores (de frío a cálido)
col_palette <- colorRampPalette(c('#4477AA', '#77AADD', '#FFFFFF', '#EE9988', '#BB4444'))

setEPS()
postscript(str_c(PLOT_PATH,"corrplot.eps"))

corrplot(M_corr, method="color", col=col_palette(50),
         type="upper", #order="hclust",
         addCoef.col = "black", # Add coefficient of correlation
         tl.col="black", tl.srt=45, #Text label color and rotation
         # Combine with significance
         #p.mat = p.mat, sig.level = 0.01, insig = "blank",
         # hide correlation coefficient on the principal diagonal
         #diag=FALSE
         )
dev.off()

#
#
# === === === === === === === === === === === === === === === === === === === ===
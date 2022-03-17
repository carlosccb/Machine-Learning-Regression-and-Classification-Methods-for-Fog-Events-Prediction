#!/bin/bash

for i in {1..9};
do
	echo " # Ejecutando exps multinomial Nominal ${i}"
	nohup Rscript --vanilla multinomial_elastic_net_glm_Ordinal.R $i &
	#sudo nice -n -12 Rscript --vanilla multinomial_elastic_net_glm_Ordinal.R $i
done

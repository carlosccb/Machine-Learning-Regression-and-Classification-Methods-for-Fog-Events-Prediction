#!/bin/bash

for i in {1..9};
do
	echo " # Ejecutando exps multinomial Nominal ${i}"
	nohup Rscript --vanilla multinomial_elastic_net_glm.R $i &
done

#!/bin/bash

for i in {0,1,2,9};
do
	echo "# Running exp with regr ${i}"
	python exp_regr.py $i
	#mv nohup.out nohup-exp_${i}.out
done

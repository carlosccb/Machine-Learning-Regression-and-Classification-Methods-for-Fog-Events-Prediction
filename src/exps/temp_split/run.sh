#!/bin/bash

for i in {0,1,2,9};
do
	echo "# Running exp ${i}"
	python exp.py $i
	#mv nohup.out nohup-exp_${i}.out
done

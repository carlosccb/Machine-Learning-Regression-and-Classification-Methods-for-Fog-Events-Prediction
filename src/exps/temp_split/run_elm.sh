#!/bin/bash

for i in {0,1,2};
do
	echo "# Running exp ${i}"
	python elm_exp_class.py $i
	#mv nohup.out nohup-exp_${i}.out
done

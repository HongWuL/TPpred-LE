#!/bin/bash

for ((p=1; p<=5; p ++))
do
    echo "partial ${p}"
    for ((i=1; i<=5; i++))
    do
        echo "square iter ${i}"
        python -u main.py -act rt e -seed `expr ${i} + 45` -s square -src "datasets/data_partial/p${p}" -task2 RT_p${p}_sqr${i} -pth trained_models/Patial_p${p}_${i}.pth -pth2 trained_models/RT_p${p}_sqr${i}.pth >logs/RT_p${p}_sqr${i}.txt
    done
done




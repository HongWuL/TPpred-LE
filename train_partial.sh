#!/bin/bash


for ((p=1; p<=5; p++))
do
    echo "partial ${p}"
    for ((i=1; i<=5; i++))
    do
        echo "iter ${i}"
        python -u main.py -act t e -seed `expr ${i} + 45` -src "datasets/data_partial/p${p}" -task Patial_p${p}_${i} -pth trained_models/Patial_p${p}_${i}.pth >logs/Patial_p${p}_${i}.txt 
    done

done

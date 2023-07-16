#!/bin/bash

for ((i=1; i<=5; i++))
do
    echo "Square iter ${i}"
    python -u main.py -seed `expr 45 + ${i}` -act rt e -s square -task2 RTsqr${i} -pth trained_models/TPpred_tw_60_${i}.pth -pth2 trained_models/RTsqr${i}.pth >logs/RTsqr${i}.txt

done


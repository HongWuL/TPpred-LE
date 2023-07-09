#!/bin/bash

for ((i=1; i<=5; i++))

do
    echo "train $i, seed = `expr 45 + ${i}`"
    python -u main.py -act t e -seed `expr 45 + ${i}` -task TPpred_tw_60_${i} -pth trained_models/TPpred_tw_60_${i}_rpad.pth >logs/TPpred_tw_60_${i}_rpad.txt

done
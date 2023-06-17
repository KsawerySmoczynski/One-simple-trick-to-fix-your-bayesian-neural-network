#!/bin/bash

for VAR in "manual" "auto"
do
  for MODEL in "MLEClassify"
  do
    for DATASET in "MNIST" "FashionMNIST"
    do
      sbatch job.sh ${DATASET} ${MODEL} ${VAR}
    done
  done
done

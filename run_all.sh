#!/bin/bash

for VAR in "manual" "auto"
do
  for SEED in {1..5}
  do
    sbatch job.sh ${VAR} ${SEED}
  done
done
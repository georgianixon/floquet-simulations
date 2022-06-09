#!/usr/bin/env bash

start=4;
end=20;
step=0.1;

for val in `seq $start $step $end`; do 
sbatch slurm_submit $val; 
done

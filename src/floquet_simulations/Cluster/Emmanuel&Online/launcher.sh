#!/usr/bin/env bash

start_depth=1.5;
end_depth=11.6;
step_depth=0.25;

for depth in `seq $start_depth $step_depth $end_depth`; do sbatch slurm_submit $depth; done

#!/usr/bin/env bash

startA=0; 
endA=3; 
stepA=1;
startB=5; 
endB=7; 
stepB=1;

for valA in `seq $startA $stepA $endA`; do 
for valB in `seq $startB $stepB $endB`; do 

cp -r DataGenBase.py DataGenNA"$valA"B"$valB".py
cp -r slurm_submit_base slurm_submitNA"$valA"B"$valB"

sed -i -e "s/BASHA2/$valA/g" DataGenNA"$valA"B"$valB".py
sed -i -e "s/BASHA3/$valB/g" DataGenNA"$valA"B"$valB".py
sed -i -e "s/DataGen/DataGenNB$valB/g" slurm_submitNA"$valA"B"$valB"
sed -i -e "s/DataGenN/DataGenNA$valA/g" slurm_submitNA"$valA"B"$valB"


sbatch slurm_submitNA"$valA"B"$valB"

done


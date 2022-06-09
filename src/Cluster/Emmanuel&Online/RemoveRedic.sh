#!/usr/bin/env bash

for ((Num=61355816; Num<61356008; Num=SysSz+1)); do

rm -r "ERR"$Num"_cpu-e-1149.err"
rm -r "ERR"$Num"_cpu-e-1050.err"
rm -r "ERR"$Num"_cpu-e-15.err"
rm -r "ERR"$Num"_cpu-e-16.err"
rm -r "ERR"$Num"_cpu-e-236.err"
rm -r "ERR"$Num"_cpu-e-210.err"
rm -r "ERR"$Num"_cpu-e-231.err"
rm -r "ERR"$Num"_cpu-e-124.err"
rm -r "ERR"$Num"_cpu-e-134.err"
rm -r "ERR"$Num"_cpu-e-244.err"
rm -r "ERR"$Num"_cpu-e-86.err"
rm -r "ERR"$Num"_cpu-e-36.err"
rm -r "ERR"$Num"_cpu-e-141.err"
rm -r "ERR"$Num"_cpu-e-140.err"
rm -r "ERR"$Num"_cpu-e-126.err"
rm -r "ERR"$Num"_cpu-e-1151.err"
rm -r "ERR"$Num"_cpu-e-210.err"

rm "machine.file."$Num

rm "OUT"$Num"_cpu-e-1149.err"
rm "OUT"$Num"_cpu-e-1050.err"
rm "OUT"$Num"_cpu-e-15.err"
rm "OUT"$Num"_cpu-e-16.err"
rm "OUT"$Num"_cpu-e-236.err"
rm "OUT"$Num"_cpu-e-210.err"
rm "OUT"$Num"_cpu-e-231.err"
rm "OUT"$Num"_cpu-e-124.err"
rm "OUT"$Num"_cpu-e-134.err"
rm "OUT"$Num"_cpu-e-244.err"
rm "OUT"$Num"_cpu-e-86.err"
rm "OUT"$Num"_cpu-e-36.err"
rm "OUT"$Num"_cpu-e-141.err"
rm "OUT"$Num"_cpu-e-140.err"
rm "OUT"$Num"_cpu-e-126.err"
rm "OUT"$Num"_cpu-e-1151.err"
rm "OUT"$Num"_cpu-e-210.err"


done


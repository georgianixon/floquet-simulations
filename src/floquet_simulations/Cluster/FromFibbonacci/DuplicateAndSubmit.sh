#!/bin/sh

WT=23
MEM=1

T=400
SampleSize=100

pStart=0.228
pGap=0.00
pNumIterations=50


for ((SysSz=64; SysSz<65; SysSz=SysSz*2 )); do
for ((i=0; i<pNumIterations; i=i+1)); do
for ((MT=0; MT<1; MT=MT+2)); do

cp -r ./"FibCode-IID" ./"FibCodeL="$SysSz"pInt="$i
cd "FibCodeL="$SysSz"pInt="$i

sed -i -e "s/BASHSYSSIZE/$SysSz/g" main.cpp
sed -i -e "s/BASHSAMPLESIZE/$SampleSize/g" main.cpp
sed -i -e "s/BASHpINT/$i/g" main.cpp
sed -i -e "s/BASHpSTART/$pStart/g" main.cpp
sed -i -e "s/BASHpGAP/$pGap/g" main.cpp
sed -i -e "s/BASHMINTRIES/$MT/g" main.cpp


sed -i -e "s/BASHWALLTIME/$WT/g" Submit.pbs
sed -i -e "s/BASHTRIALS/$T/g" Submit.pbs
sed -i -e "s/BASHMEMORY/"$MEM"GB/g" Submit.pbs

sed -i -e "s/BASHFILENAME/FibCodeL="$SysSz"pInt="$i"/g" Submit.pbs
sed -i -e "s/BASHJOB/Fib"$T"L"$SysSz"p"$i"/g" Submit.pbs

qsub Submit.pbs

cd ..

#rm -r "X-cube-X-erL="$SysSz"pInt="$i"cnrs="$CORNER

done
done
done
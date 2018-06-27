#! /bin/bash

if [ $1 -eq 0 ];
then
echo "Long running test! testPlinkMultivariateLinearMixedModelMultiplePhenotypes_Issue58"
./build/faster_lmm_d --bfile=data/HLC --pheno=data/HLC.simu.pheno.txt --kinship=output/testPlinkStandardRelatednessMatrixK.sXX.txt --lmm=1 --maf=0.1 --n=1,2,3,15 --c=HLC_covariates.txt --cmd=mvlmm
fi
#! /bin/bash

echo "Long running test! testPlinkMultivariateLinearMixedModelMultiplePhenotypes_Issue58"
./build/faster_lmm_d --bfile=data/gemma/HLC --pheno=data/gemma/HLC.simu.pheno.txt --kinship=data/gemma/PlinkStandardRelatednessMatrixK.txt --maf=0.1 --n=1,2,3,15 --covar=data/gemma/HLC_covariates.txt --cmd=mvlmm
if [ $? -ne 0 ]; then echo "ERR1 Multivariate Linear Mixed Model test failed" ; exit 1 ; fi

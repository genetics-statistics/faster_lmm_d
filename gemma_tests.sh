#! /bin/bash

if [ $1 -eq 0 ];
then
echo "running kinship test"
./build/faster_lmm_d --geno=./data/gemma/mouse_hs1940.geno.txt.gz --kinship=output.txt  --pheno=./data/gemma/mouse_hs1940.pheno --cmd=gk

echo "testUnivariateLinearMixedModel()"
./build/faster_lmm_d --pheno=data/gemma/mouse_hs1940.pheno --geno=data/gemma/mouse_hs1940.geno.txt.gz --ni_total=1940 --kinship=data/gemma/mouse_hs1940.kinship --indicator_idv=data/gemma/indicator_idv.txt --indicator_snp=data/gemma/indicator_snp.txt --test-kinship=true --test-name=mouse_hs1940 --cmd=run $*
if [ $? -ne 0 ]; then echo "ERR2 testCenteredRelatednessMatrixK()" ; exit 1 ; fi
fi

if [ $1 -eq 1 ];
then
echo "testBXDwithout covar()"
./build/faster_lmm_d --pheno=data/gemma/BXD.pheno --geno=data/gemma/BXD_geno.txt.gz --kinship=data/gemma/BXD.kinship --indicator_idv=data/gemma/BXD_indicator_idv.txt --indicator_snp=data/gemma/BXD_indicator_snp.txt --test-kinship=true --test-name=BXD --cmd=run $*
fi

if [ $1 -eq 2 ];
then
echo "testBXDMultivariateLinearMixedModel()"
./build/faster_lmm_d --pheno=data/gemma/BXD.pheno --geno=data/gemma/BXD_geno.txt.gz --kinship=data/gemma/BXD.kinship --covar=data/gemma/BXD_covariates2.txt --indicator_idv=data/gemma/BXD_indicator_idv.txt --indicator_snp=data/gemma/BXD_indicator_snp.txt --test-kinship=true --test-name=BXD --cmd=run
fi

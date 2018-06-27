#! /bin/bash

echo "testUnivariateLinearMixedModel()"
./build/faster_lmm_d --pheno=data/gemma/mouse_hs1940.pheno --geno=data/gemma/mouse_hs1940.geno.txt.gz --ni_total=1940 --kinship=data/gemma/mouse_hs1940.kinship --indicator_idv=data/gemma/indicator_idv.txt --indicator_snp=data/gemma/indicator_snp.txt --test-kinship=true --test-name=mouse_hs1940 --cmd=run $*
if [ $? -ne 0 ]; then echo "ERR2 testCenteredRelatednessMatrixK()" ; exit 1 ; fi

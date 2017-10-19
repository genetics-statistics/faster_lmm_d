#! /bin/bash

echo "Running mouse_hs1940 test"
./build/faster_lmm_d --pheno=data/gemma/mouse_hs1940.pheno --geno=data/gemma/mouse_hs1940.geno.txt.gz --kinship=data/gemma/mouse_hs1940.kinship --indicator_idv=data/gemma/indicator_idv.txt --indicator_snp=data/gemma/indicator_snp.txt --test-kinship=true --cmd=run $*
if [ $? -ne 0 ]; then echo "ERR1a mouse_hs1940 test failed" ; exit 1 ; fi

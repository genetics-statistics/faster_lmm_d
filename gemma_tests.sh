#! /bin/bash

echo "Running mouse_hs1940 test"
./build/faster_lmm_d --pheno=data/gemma/mouse_hs1940.pheno --geno=data/gemma/mouse_hs1940.geno --kinship=data/gemma/mouse_hs1940.kinship --test-kinship=true --cmd=run $*
if [ $? -ne 0 ]; then echo "ERR1a mouse_hs1940 test failed" ; exit 1 ; fi
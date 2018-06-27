#! /bin/bash

echo "running kinship test"
./build/faster_lmm_d --geno=data/gemma/mouse_hs1940.geno.txt.gz --kinship=output.txt --pheno=./data/gemma/mouse_hs1940.pheno.txt  --cmd=gk

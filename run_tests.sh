#! /bin/bash

echo "Running small test"
./build/faster_lmm_d --control=data/genenetwork/BXD.json --pheno=data/small.pheno --geno=data/small.geno --cmd=run $*
if [ $? -ne 0 ]; then echo "ERR1a small test failed" ; exit 1 ; fi

echo "Running small_na test"
./build/faster_lmm_d --control=data/genenetwork/BXD.json --pheno=data/small_na.pheno --geno=data/small_na.geno --cmd=run $*
if [ $? -ne 0 ]; then echo "ERR1b small_na test failed" ; exit 1 ; fi

echo "Running 104617_at"
./build/faster_lmm_d --control=data/genenetwork/BXD.json --pheno=data/genenetwork/104617_at.json --geno=data/genenetwork/BXD.csv --cmd=rqtl $*
if [ $? -ne 0 ]; then echo "ERR2 104617_at test failed" ; exit 1 ; fi

echo "Running test8000"
time ./build/faster_lmm_d --control=data/genenetwork/BXD.json --pheno=data/test8000.pheno --geno=data/test8000.geno --cmd=run $*
if [ $? -ne 0 ]; then echo "ERR3 test8000 failed" ; fi

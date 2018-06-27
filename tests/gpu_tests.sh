#! /bin/bash
key="$1"

mkdir gpu_test

echo "Pheno file generated"
rdmd ./source/test/pheno_generator.d $key > ./gpu_test/test.pheno

echo "Pheno file generated"
rdmd ./source/test/geno_generator.d  $key > ./gpu_test/test.geno

echo "Running GWAS"
time ./build/faster_lmm_d --control=data/genenetwork/BXD.json --pheno=./gpu_test/test.pheno --geno=./gpu_test/test.geno --cmd=run

rmdir gpu_test

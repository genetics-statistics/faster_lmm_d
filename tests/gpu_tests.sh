#! /bin/bash
key="$1"

echo "Geno file generated"
rdmd source/test/pheno_generator.d $key > source/test/test.pheno

echo "Pheno file generated"
rdmd source/test/geno_generator.d  $key > source/test/test.geno

echo "Running GWAS"
time ./build/faster_lmm_d --control=data/genenetwork/BXD.json --pheno=source/test/test.pheno --geno=source/test/test.geno --cmd=run
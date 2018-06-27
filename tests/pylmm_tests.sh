#! /bin/bash

echo "PyLMM tests"

echo "Running small test"
./build/faster_lmm_d --pheno=data/small.pheno --geno=data/small.geno --test-kinship=true --cmd=run $*
if [ $? -ne 0 ]; then echo "ERR1a small test failed" ; exit 1 ; fi

echo "Running small_na test"
./build/faster_lmm_d --pheno=data/small_na.pheno --geno=data/small_na.geno --test-kinship=true --cmd=run $*
if [ $? -ne 0 ]; then echo "ERR1b small_na test failed" ; exit 1 ; fi

echo "iron"
./build/faster_lmm_d --control=data/rqtl/iron.json --pheno=data/rqtl/iron_pheno.csv --geno=data/rqtl/iron_geno.csv --test-kinship=true --cmd=rqtl $*
if [ $? -ne 0 ]; then echo "ERR2a iron test failed" ; exit 1 ; fi

# echo "recla"
# ./build/faster_lmm_d --control=data/rqtl/recla.json --pheno=data/rqtl/recla_pheno.csv --geno=data/rqtl/recla_geno.csv --test-kinship=true --cmd=rqtl $*
# if [ $? -ne 0 ]; then echo "ERR2b recla test failed" ; exit 1 ; fi

echo "Running 104617_at"
./build/faster_lmm_d --control=data/genenetwork/BXD.json --pheno=data/genenetwork/104617_at.json --geno=data/genenetwork/BXD.csv --test-kinship=true --cmd=rqtl $*
if [ $? -ne 0 ]; then echo "ERR3a 104617_at test failed" ; exit 1 ; fi

echo "Running test8000"
time ./build/faster_lmm_d --pheno=data/test8000.pheno --geno=data/test8000.geno --test-kinship=true --cmd=run $*
if [ $? -ne 0 ]; then echo "ERR3b test8000 failed" ; fi

# echo "iron with covar"
# ./build/faster_lmm_d --control=data/rqtl/iron.json --covar=data/rqtl/iron_covar.csv --pheno=data/rqtl/iron_pheno.csv --geno=data/rqtl/iron_geno.csv --test-kinship=true --cmd=rqtl $*
# if [ $? -ne 0 ]; then echo "ERR4a iron test failed" ; exit 1 ; fi

# echo "recla with covar"
# ./build/faster_lmm_d --control=data/rqtl/recla.json --covar=data/rqtl/recla_covar.csv --pheno=data/rqtl/recla_pheno.csv --geno=data/rqtl/recla_geno.csv --test-kinship=true --cmd=rqtl $*
# if [ $? -ne 0 ]; then echo "ERR4b recla test failed" ; exit 1 ; fi

# echo "recla with kinship"
# ./build/faster_lmm_d --control=data/rqtl/recla.json --covar=data/rqtl/recla_covar.csv --pheno=data/rqtl/recla_pheno.csv --geno=data/rqtl/recla_geno.csv --kinship=data/rqtl/recla_kinship.csv --test-kinship=true --cmd=rqtl $*
# if [ $? -ne 0 ]; then echo "ERR4c recla test failed: Must fail for now" ; exit 1 ; fi

echo "GEMMA tests"

echo "CenteredRelatednessMatrixKFullLOCO1"
$build/faster_lmm_d -g ../example/mouse_hs1940.geno.txt.gz \
                    -p ../example/mouse_hs1940.pheno.txt \
                    -a ../example/mouse_hs1940.anno.txt \
                    -loco 1 \
                    -gk \
                    -o $outn
if [ $? -ne 0 ]; then echo "Error: CenteredRelatednessMatrixKFullLOCO1 test failed" ; exit 1 ; fi

echo "UnivariateLinearMixedModelFullLOCO1"
./build/faster_lmm_d -g ./data/gemma/mouse_hs1940.geno.txt.gz \
                     -p ./data/gemma/mouse_hs1940.pheno.txt \
                     -n 1 \
                     -loco 1 \
                     -a ./data/gemma/mouse_hs1940.anno.txt \
                     -k ./output/mouse_hs1940_full_LOCO1.cXX.txt \
                     -lmm \
                     -o $outn
if [ $? -ne 0 ]; then echo "Error: UnivariateLinearMixedModelFullLOCO1 test failed" ; exit 1 ; fi

echo "CenteredRelatednessMatrixK"
./build/faster_lmm_d -g ./data/gemma/mouse_hs1940.geno.txt.gz \
                     -p ./data/gemma/mouse_hs1940.pheno.txt \
                     -gk -o mouse_hs1940
if [ $? -ne 0 ]; then echo "Error: CenteredRelatednessMatrixK test failed" ; exit 1 ; fi

echo "UnivariateLinearMixedModel"
./build/faster_lmm_d -g ./data/gemma/mouse_hs1940.geno.txt.gz \
                     -p ./data/gemma/mouse_hs1940.pheno.txt \
                     -n 1 \
                     -a ./data/gemma/mouse_hs1940.anno.txt \
                     -k ./output/mouse_hs1940.cXX.txt \
                     -lmm \
                     -o mouse_hs1940_CD8_lmm
if [ $? -ne 0 ]; then echo "Error: UnivariateLinearMixedModel test failed" ; exit 1 ; fi

echo "LinearMixedModelPhenotypes"
./build/faster_lmm_d -g ./data/gemma/mouse_hs1940.geno.txt.gz \
                     -p ./data/gemma/mouse_hs1940.pheno.txt \
                     -n 1 6 \
                     -a ./data/gemma/mouse_hs1940.anno.txt \
                     -k ./output/mouse_hs1940.cXX.txt \
                     -lmm \
                     -o mouse_hs1940_CD8MCH_lmm
if [ $? -ne 0 ]; then echo "Error: LinearMixedModelPhenotypes test failed" ; exit 1 ; fi

echo "PlinkStandardRelatednessMatrixK"
./build/faster_lmm_d -bfile ./data/gemma/HLC \
                     -gk 2
                     -o $testname
if [ $? -ne 0 ]; then echo "Error: PlinkStandardRelatednessMatrixK test failed" ; exit 1 ; fi

# Test for https://github.com/genetics-statistics/GEMMA/issues/58
# fixed GSLv2 NaN's that appeared with covariates.
echo "PlinkLinearMixedModelCovariates"
./build/faster_lmm_d -bfile $datadir/HLC \
                     -k output/testPlinkStandardRelatednessMatrixK.sXX.txt \
                     -lmm 1 \
                     -maf 0.1 \
                     -c $datadir/HLC_covariates.txt \
                     -o $testname
if [ $? -ne 0 ]; then echo "Error: PlinkLinearMixedModelCovariates test failed" ; exit 1 ; fi

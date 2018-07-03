
echo "Running Linear Model test"
./build/faster_lmm_d --geno=data/gemma/BXD_geno.txt.gz --pheno=data/gemma/BXD.pheno --covar=data/gemma/BXD_covariates2.txt --anno=data/gemma/BXD_snps.txt --kinship=data/gemma/BXD.cXX.txt --lm=4 --maf=0.1 --cmd=lm
if [ $? -ne 0 ]; then echo "ERR1 Linear Model test failed" ; exit 1 ; fi

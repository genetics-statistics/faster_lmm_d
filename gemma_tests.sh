echo "testBXDwithout covar()"
./build/faster_lmm_d --pheno=data/gemma/BXD.pheno --geno=data/gemma/BXD_geno.txt.gz --kinship=data/gemma/BXD.kinship --indicator_idv=data/gemma/BXD_indicator_idv.txt --indicator_snp=data/gemma/BXD_indicator_snp.txt --test-kinship=true --test-name=BXD --cmd=run $*

echo "testBXDMultivariateLinearMixedModel()"
./build/faster_lmm_d --pheno=data/gemma/BXD.pheno --geno=data/gemma/BXD_geno.txt.gz --kinship=data/gemma/BXD.kinship --covar=data/gemma/BXD_covariates2.txt --indicator_idv=data/gemma/BXD_indicator_idv.txt --indicator_snp=data/gemma/BXD_indicator_snp.txt --test-kinship=true --test-name=BXD --cmd=run

echo "testBXDMultivariateLinearMixedModel()"
./build/faster_lmm_d -bfile=data/gemma//HLC -pheno=data/gemma/HLC.simu.pheno.txt -kinship=PlinkStandardRelatednessMatrixK.txt --lmm=1 --maf=0.1 --n=12 3 15 --c=data/gemma/HLC_covariates.txt

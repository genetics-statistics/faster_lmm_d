/*
   This code is part of faster_lmm_d and published under the GPLv3
   License (see LICENSE.txt)

   Copyright © 2017 Prasun Anand & Pjotr Prins
*/

module faster_lmm_d.lmm;

import std.exception;
import std.experimental.logger;

import faster_lmm_d.dmatrix;
import faster_lmm_d.gwas;
import faster_lmm_d.helpers;
import faster_lmm_d.kinship;
import faster_lmm_d.optmatrix;
import faster_lmm_d.phenotype;

struct KGstruct{
  dmatrix K, G;

  this(dmatrix K, dmatrix G){
    this.K = K;
    this.G = G;
  }
}

auto run_gwas(int n, int m, dmatrix k, double[] y, dmatrix geno){

  trace("run_gwas");
  log("pheno ", y.length," ", y[0..5]);
  log(geno.shape,m);
  enforce(y.length == n);
  enforce(geno.shape[1] == m);

  phenoStruct pheno = remove_missing(n,y);
  n = pheno.n;
  double[] Y = pheno.Y;
  bool[] keep = pheno.keep;

  geno = removeCols(geno,keep);
  trace("Calculate Kinship");
  KGstruct KG = calculate_kinship_new(geno);
  log("kinship_matrix.shape: ", KG.K.shape);

  log("run_other_new genotype_matrix: ", KG.G.shape);
  return gwas(Y, KG.G, KG.K, true, false, true);
}

KGstruct calculate_kinship_new(dmatrix genotype_matrix){
  trace("call calculate_kinship_new");
  log(genotype_matrix.shape);
  dmatrix G = normalize_along_row(genotype_matrix);
  dmatrix K = kinship_full(G);
  return KGstruct(K, G);
}

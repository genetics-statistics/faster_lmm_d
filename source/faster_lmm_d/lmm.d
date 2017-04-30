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
  dmatrix G = normalize_along_row(geno);
  dmatrix K = kinship_full(G);
  log("kinship_matrix.shape: ", K.shape);

  log("run_other_new genotype_matrix: ", G.shape);
  return gwas(Y, G, K, true, false, true);
}

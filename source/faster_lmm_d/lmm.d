/*
   This code is part of faster_lmm_d and published under the GPLv3
   License (see LICENSE.txt)

   Copyright © 2017 Prasun Anand & Pjotr Prins
*/

module faster_lmm_d.lmm;

import std.experimental.logger;

import faster_lmm_d.dmatrix;
import faster_lmm_d.gwas;
import faster_lmm_d.helpers;
import faster_lmm_d.kinship;
import faster_lmm_d.optmatrix;
import faster_lmm_d.phenotype;

auto run_gwas(ulong n, ulong m, const dmatrix k, double[] y, const dmatrix geno){

  trace("run_gwas");
  log("pheno ", y.length," ", y[0..5]);
  log(geno.shape,m);
  assert(y.length == n);
  assert(geno.shape[1] == m);

  phenoStruct pheno = remove_missing(n,y);

  auto geno2 = removeCols(geno,pheno.keep);
  dmatrix G = normalize_along_row(geno2);
  log("run_other_new genotype_matrix: ", G.shape);
  dmatrix K = kinship_full(G);
  log("kinship_matrix.shape: ", K.shape);

  return gwas(pheno.Y, G, K, true, false, true);
}

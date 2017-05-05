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

auto run_gwas(m_items n, m_items m, const DMatrix k, double[] y, const DMatrix geno) {
  trace("run_gwas");
  trace("pheno ", y.length," ", y[0..4]);
  trace(geno.shape,m);
  assert(y.length == n);
  assert(geno.m_geno == m);

  phenoStruct pheno = remove_missing(n,y);

  auto geno2 = removeCols(geno,pheno.keep);
  DMatrix G = normalize_along_row(geno2);
  trace("run_other_new genotype_matrix: ", G.shape);
  DMatrix K = kinship_full(G);
  trace("kinship_matrix.shape: ", K.shape);

  return gwas(pheno.Y, G, K, true, false, true);
}

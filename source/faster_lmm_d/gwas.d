/*
   This code is part of faster_lmm_d and published under the GPLv3
   License (see LICENSE.txt)

   Copyright © 2017 Prasun Anand & Pjotr Prins
*/

module faster_lmm_d.gwas;

import std.experimental.logger;
import std.typecons;

import faster_lmm_d.dmatrix;
import faster_lmm_d.kinship;
import faster_lmm_d.phenotype;
import faster_lmm_d.lmm2;
import faster_lmm_d.memory;
import faster_lmm_d.optmatrix;
import faster_lmm_d.output;

import core.stdc.stdlib : exit;

auto gwas(immutable double[] Y, const DMatrix G, const DMatrix K, const bool reml = true, const bool refit=false, const bool verbose = true){

  trace("In gwas.gwas");

  auto inds = G.cols();
  auto snps = G.rows();

  infof("%d SNPs",snps);

  if( snps < inds ){
    log("snps should be larger than inds (snps=%d,inds=%d)", snps,inds);
  }

  DMatrix Kva;
  DMatrix Kve;
  DMatrix X0;

  check_memory("Before gwas");

  LMM lmm = LMM(Y, Kva, Kve, K.shape[0], X0, kvakve(K));
  lmm = lmm_transform(lmm);

  check_memory();

  if(!refit){
    trace("Computing fit for null model");
    DMatrix X;
    lmm = lmm_fit(lmm, X);
    log("heritability= ", lmm.opt_H, " sigma= ", lmm.opt_sigma, " LL= ", lmm.opt_LL);
  }

  double[] ps = new double[snps];
  double[] ts = new double[snps];
  double[] lod = new double[snps];

  info(G.shape);

  for(int i=0; i<snps; i++){
    DMatrix x = get_row(G, i);
    x.shape = [inds, 1];
    auto tsps = lmm_association(lmm, x, true,true);
    ps[i]  = tsps[1];
    ts[i]  = tsps[0];
    lod[i] = tsps[2];

    if(i%1000 == 0){
      log(i, " snps processed");
    }
  }

  return Tuple!(double[], double[], double[])(ts, ps, lod);
}


auto run_gwas(immutable m_items n, immutable m_items m, DMatrix k, immutable double[] y, const DMatrix geno) {
  trace("run_gwas");
  trace("pheno ", y.length," ", y[0..4]);
  trace(geno.shape,m);
  check_memory("before run_gwas");
  assert(y.length == n);
  assert(geno.m_geno == m);

  PhenoStruct pheno = remove_missing(n,y);

  auto geno2 = remove_cols(geno,pheno.keep);
  DMatrix G = normalize_along_row(geno2);
  trace("run_other_new genotype_matrix: ", G.shape);
  DMatrix K = kinship_full(G);
  trace("kinship_matrix.shape: ", K.shape);

  return gwas(pheno.Y, G, K, true, false, true);
}

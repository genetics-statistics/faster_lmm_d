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

auto gwas(immutable double[] Y, const DMatrix G, const DMatrix K){

  const bool reml  = true;
  const bool refit = false;
  trace("In gwas.gwas");

  auto inds = G.cols();
  auto snps = G.rows();

  infof("%d SNPs",snps);

  if( snps < inds ){
    log("snps should be larger than inds (snps=%d,inds=%d)", snps,inds);
  }

  check_memory("Before gwas");

  auto N = cast(N_Individuals)K.shape[0];
  auto kvakve = kvakve(K);
  DMatrix Dummy_X0;
  LMM lmm1 = LMM(Y, kvakve.kva, Dummy_X0);
  auto lmm2 = lmm_transform(lmm1,N,Y,kvakve.kve);

  trace("Computing fit for null model");
  DMatrix X;
  auto lmm = lmm_fit(lmm2, N, X);
  log("heritability= ", lmm.opt_H, " sigma= ", lmm.opt_sigma, " LL= ", lmm.opt_LL);

  check_memory();

  double[] ps = new double[snps];
  double[] ts = new double[snps];
  double[] lod = new double[snps];

  info(G.shape);

  DMatrix KveT = kvakve.kve.T; // out of the loop
  for(int i=0; i<snps; i++){
    DMatrix x = get_row(G, i);
    x.shape = [inds, 1];
    auto tsps = lmm_association(lmm, N, x, KveT);
    ps[i]  = tsps[1];
    ts[i]  = tsps[0];
    lod[i] = tsps[2];

    if(i%1000 == 0){
      info(i, " snps processed");
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

  return gwas(pheno.Y, G, K);
}

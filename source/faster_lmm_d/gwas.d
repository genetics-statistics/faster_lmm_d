/*
   This code is part of faster_lmm_d and published under the GPLv3
   License (see LICENSE.txt)

   Copyright © 2017 Prasun Anand & Pjotr Prins
*/

module faster_lmm_d.gwas;

import std.experimental.logger;
import std.parallelism;
import std.range;
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

  println("Compute GWAS");
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
  info(G.shape);

  auto task_pool = new TaskPool(8);
  scope(exit) task_pool.finish();

  DMatrix KveT = kvakve.kve.T; // compute out of loop
  version(PARALLEL) {
    auto tsps = new TStat[snps];
    auto items = iota(0,snps).array;

    foreach (ref snp; taskPool.parallel(items,10)) {
      print(".");
      tsps[snp] = lmm_association(snp, lmm, N, G, KveT);
    }
  } else {
    TStat[] tsps;
    foreach(snp; 0..snps) {
      tsps ~= lmm_association(snp, lmm, N, G, KveT);
      if(snp % 1000 == 0){
        info(snp, " snps processed");
      }
    }
  }

  return tsps;
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

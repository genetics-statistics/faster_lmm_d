/*
   This code is part of faster_lmm_d and published under the GPLv3
   License (see LICENSE.txt)

   Copyright © 2017 Prasun Anand & Pjotr Prins
*/

module faster_lmm_d.gwas;

import core.stdc.stdlib : exit;

import std.experimental.logger;
import std.exception;
import std.parallelism;
import std.range;
import std.typecons;

import faster_lmm_d.dmatrix;
import faster_lmm_d.kinship;
import faster_lmm_d.lmm2;
import faster_lmm_d.memory;
import faster_lmm_d.optmatrix;
import faster_lmm_d.output;
import faster_lmm_d.phenotype;
import faster_lmm_d.helpers : sum;

import test.fit;
import test.geno_matrix;
import test.kinship;

auto gwas(immutable double[] Y, const DMatrix G, const DMatrix K, const DMatrix covar_matrix, string geno_fn){

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

  LMM lmm1 = LMM(Y, kvakve.kva, covar_matrix);
  auto lmm2 = lmm_transform(lmm1,N,Y,kvakve.kve);

  check_X0t_matrix(lmm2.X0t, geno_fn);

  trace("Computing fit for null model");

  DMatrix X; // FIXME;
  auto lmm = lmm_fit(lmm2, N, X);

  trace(
    "\nheritability = ", lmm.opt_H,
    "\nsum = ", sum(lmm.opt_beta.elements),
    "\nbeta = ", lmm.opt_beta,
    "\nsigma = ", lmm.opt_sigma,
    "\nsigmasq_g = ", lmm.opt_H * lmm.opt_sigma,
    "\nsigmasq_e = ", (1 - lmm.opt_H) * lmm.opt_sigma,
    "\nlog-likelihood = ", lmm.opt_LL,
  );
  check_lmm_fit(lmm, geno_fn);

  check_memory();
  info(G.shape);

  auto task_pool = new TaskPool(8);
  scope(exit) task_pool.finish();

  DMatrix KveT = kvakve.kve.T; // compute out of loop
  trace("Call data offload");
  offload_cache(KveT);             // send this to the cache

  version(PARALLEL) {
    auto tsps = new TStat[snps];
    auto items = iota(0,snps).array;

    println("Parallel");
    foreach (ref snp; taskPool.parallel(items,100)) {
      tsps[snp] = lmm_association(snp, lmm, N, G, KveT);
      if((snp+1) % 1000 == 0){
        println(snp+1, " snps processed");
      }
    }
  } else {
    TStat[] tsps;
    foreach(snp; 0..snps) {
      tsps ~= lmm_association(snp, lmm, N, G, KveT);
      if((snp+1) % 1000 == 0){
        println(snp+1, " snps processed");
      }
    }
  }

  return tsps;
}

auto run_gwas(immutable m_items n, immutable m_items m, immutable double[] y,
                const DMatrix geno, const DMatrix covar_matrix,
                const string geno_fn, bool test_kinship = false) {

  trace("run_gwas");
  trace("pheno ", y.length," ", y[0..4]);
  trace(geno.shape,m);
  check_memory("before run_gwas");
  assert(y.length == n);
  assert(geno.n_pheno == m);

  PhenoStruct pheno = remove_missing(n,y);

  auto geno2 = remove_cols(geno,pheno.keep);
  DMatrix G = normalize_along_row(geno2);
  trace("run_other_new genotype_matrix: ", G.shape);
  DMatrix K = kinship_full(G);
  trace("kinship_matrix.shape: ", K.shape);

  if(test_kinship){check_kinship(K, geno_fn);}

  return gwas(pheno.Y, G, K, covar_matrix, geno_fn);
}

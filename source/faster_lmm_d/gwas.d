/*
   This code is part of faster_lmm_d and published under the GPLv3
   License (see LICENSE.txt)

   Copyright © 2017 Prasun Anand & Pjotr Prins
*/

module faster_lmm_d.gwas;

import std.experimental.logger;
import std.typecons;

import dstats.distrib;
import faster_lmm_d.dmatrix;
import faster_lmm_d.lmm2;
import faster_lmm_d.optmatrix;


auto gwas(double[] Y, dmatrix G, dmatrix K, bool restricted_max_likelihood = true, bool refit=false, bool verbose = true){

  trace("In gwas.gwas");

  int inds = G.shape[1];
  int snps = G.shape[0];

  infof("%d SNPs",snps);

  if( snps < inds ){
    log("snps should be larger than inds (snps=%d,inds=%d)", snps,inds);
  }

  dmatrix Kva;
  dmatrix Kve;
  dmatrix X0;

  LMM2 lmm2 = LMM2(Y,K,Kva,Kve,X0, true);
  lmm2 = lmm2transform(lmm2);
  dmatrix X;

  if(!refit){
    trace("Computing fit for null model");
    double fit_hmax, fit_sigma, fit_LL;
    dmatrix fit_beta;
    fitTuple fit = lmm2fit(lmm2, X); // # follow GN model in run_other;
    lmm2 = fit.lmmobj;
    log("heritability= ", lmm2.optH, " sigma= ", lmm2.optSigma, " LL= ", fit.fit_LL);
  }

  double[] ps = new double[snps];
  double[] ts = new double[snps];
  info(G.shape);
  info("snps is ", snps);

  for(int i=0; i<snps; i++){
    dmatrix x = getRow(G, i);
    x.shape = [inds, 1];
    auto tsps = lmm2association(lmm2, x, true,true);
    ps[i] = tsps[1];
    ts[i] = tsps[0];
    if(i%1000 == 0){
      log(i, " snps processed");
    }
  }

  return Tuple!(double[], double[])(ts,ps);
}

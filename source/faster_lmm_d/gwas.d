/*
   This code is part of faster_lmm_d and published under the GPLv3
   License (see LICENSE.txt)

   Copyright © 2017 Prasun Anand & Pjotr Prins
*/

module faster_lmm_d.gwas;

import std.experimental.logger;
import std.typecons;

import faster_lmm_d.dmatrix;
import faster_lmm_d.lmm2;
import faster_lmm_d.optmatrix;


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

  LMM lmm = LMM(Y, K, Kva, Kve, X0, true);
  lmm = lmm_transform(lmm);

  if(!refit){
    trace("Computing fit for null model");
    DMatrix X;
    FitTuple fit = lmm_fit(lmm, X); // # follow GN model in run_other;
    lmm = fit.lmmobj;
    log("heritability= ", lmm.opt_H, " sigma= ", lmm.opt_sigma, " LL= ", fit.fit_LL);
  }

  double[] ps = new double[snps];
  double[] ts = new double[snps];
  double[] lod = new double[snps];

  info(G.shape);
  info("snps is ", snps);

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

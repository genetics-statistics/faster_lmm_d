module faster_lmm_d.gwas;
import std.stdio;
import std.typecons;
import std.experimental.logger;

import faster_lmm_d.lmm2;
import faster_lmm_d.dmatrix;
import dstats.distrib;
import faster_lmm_d.optmatrix;

void compute_snp(int j,int n,double[] snps,LMM2 lmmobject, bool REML,double q){
  trace("In compute_snp");
  double[] result;
  int rows = cast(int)(snps.length)/n;
  for(int i = 0; i< rows; i++){
    double[] snp = snps[i*j..(i+1)*j];
    dmatrix x = dmatrix([n,1], snp); //all the SNPs
    //ts,ps,beta,betaVar =
    double a = 0;
    //lmm2association(lmmobject, x, a, REML,true);
    //result.append( (ts,ps) );
  }
}

//void f_init(q){
//  compute_snp.q = q;
//}

auto gwas(double[] Y, dmatrix G, dmatrix K, bool restricted_max_likelihood = true, bool refit=false, bool verbose = true){
  //"""
  //GWAS. The G matrix should be n inds (cols) x m snps (rows)
  //"""
  trace("In gwas.gwas");
  //# matrix_initialize()
  //cpu_num = mp.cpu_count();
  //if(threads.numThreads){
  int cpu_num = 1;
  //}

  //if(gwas_useCUDA(G)){
  //  cpu_num = 1;
  //}

  infof("Using %u threads", cpu_num);

  bool kfile2 = false;
  bool reml = restricted_max_likelihood;

  //info("G",G);
  //G = matrixTranspose(G);
  int n = G.shape[1]; // inds
  int inds = n;
  int m = G.shape[0]; // snps
  int snps = m;
  infof("%d SNPs",snps);
  if(snps<inds){
    log("snps should be larger than inds (snps=%d,inds=%d)", snps,inds);
  }

  //# CREATE LMM object for association
  //# if not kfile2:  L = LMM(Y,K,Kva,Kve,X0,verbose=verbose)
  //# else:  L = LMM_withK2(Y,K,Kva,Kve,X0,verbose=verbose,K2=K2)
  dmatrix Kva;
  dmatrix Kve;
  dmatrix X0;

  LMM2 lmm2 = LMM2(Y,K,Kva,Kve,X0, true);
  lmm2 = lmm2transform(lmm2);
  dmatrix X;
  if(!refit){
    trace("Computing fit for null model");
    double fit_hmax,fit_sigma;
    dmatrix fit_beta;
    double fit_LL;
    fitTuple fit = lmm2fit(lmm2, X); // # follow GN model in run_other;
    lmm2 = fit.lmmobj;
    log("heritability= ", lmm2.optH, " sigma= ", lmm2.optSigma, " LL= ", fit.fit_LL);
  }

  double[] res;
  int q;
  double[] collect; //# container for SNPs to be processed in one batch
  int count = 0;
  int job = 0;
  int jobs_running = 0;
  int jobs_completed = 0;

  double[] ps = new double[m];
  double[] ts = new double[m];
  info(G.shape);
  info("m is ", m);
  for(int i=0; i<m; i++){
    dmatrix x = getRow(G, i);
    x.shape = [n,1];
    auto tsps = lmm2association(lmm2, x, true,true);
    ps[i] = tsps[1];
    ts[i] = tsps[0];
    if(i%1000 == 0){
      log(i, " snps processed");
    }
  }

  return Tuple!(double[], double[])(ts,ps);
}

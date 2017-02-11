module simplelmm.gwas;
import std.parallelism : taskPool;
import std.stdio;
import simplelmm.lmm2;
import simplelmm.dmatrix;
import dstats.distrib;

void compute_snp(int j,int n,double[] snps,LMM2 lmmobject, bool REML,double q){
  writeln("In compute_snp");
  double[] result;
  int rows = cast(int)(snps.length)/n;
  for(int i = 0; i< rows; i++){
    double[] snp = snps[i*j..(i+1)*j];
    dmatrix x = dmatrix([n,1], snp); //all the SNPs
    //ts,ps,beta,betaVar = 
    double a = 0;
    lmm2association(lmmobject, x, a, REML,true);
    //result.append( (ts,ps) );
  }
}

//void f_init(q){
//  compute_snp.q = q;
//}

void gwas(double[] Y, ref dmatrix G, ref dmatrix K, bool restricted_max_likelihood = true, bool refit=false, bool verbose = true){
  //"""
  //GWAS. The G matrix should be n inds (cols) x m snps (rows)
  //"""
  writeln("In gwas.gwas");
  //# matrix_initialize()
  //cpu_num = mp.cpu_count();
  //if(threads.numThreads){
  int cpu_num = std.parallelism.totalCPUs;
  cpu_num = 1;
  //}
        
  //if(gwas_useCUDA(G)){
  //  cpu_num = 1;
  //}
      
  writefln("Using %u threads", cpu_num);

  bool kfile2 = false;
  bool reml = restricted_max_likelihood;

  //writeln("G",G);
  int n = G.shape[1]; // inds
  int inds = n;
  int m = G.shape[0]; // snps
  int snps = m;
  writefln("%d SNPs",snps);
  if(snps<inds){
    writefln("snps should be larger than inds (snps=%d,inds=%d)", (snps,inds));
  }

  //# CREATE LMM object for association
  //# if not kfile2:  L = LMM(Y,K,Kva,Kve,X0,verbose=verbose)
  //# else:  L = LMM_withK2(Y,K,Kva,Kve,X0,verbose=verbose,K2=K2)
  dmatrix Kva;
  dmatrix Kve;
  dmatrix X0;

  LMM2 lmm2 = LMM2(Y,K,Kva,Kve,X0, true);
  //writeln(lmm2);
  dmatrix X;
  if(!refit){
    writeln("Computing fit for null model");
    double fit_hmax,fit_sigma;
    dmatrix fit_beta, fit_LL;
    lmm2fit(fit_hmax,fit_beta,fit_sigma,fit_LL, lmm2, X,100,true); // # follow GN model in run_other;
    writefln("heritability= ", lmm2.optH, " sigma= ", lmm2.optSigma, " LL= ", fit_LL);
  }

  double[] res;
  double q = 0;
       
  double[] collect; //# container for SNPs to be processed in one batch
  writeln(collect);
  int count = 0;
  int job = 0;
  int jobs_running = 0;
  int jobs_completed = 0;
   writeln(collect);
  for(int i = 0; i< cast(int)G.shape[0]; i++){
    double[] snp = G.elements[cast(int)i*G.shape[1]..cast(int)(i+1)*G.shape[1]];
    string snp_id = "SNPID";
    count += 1;
    if(count % 1000 == 0){

      job += 1;
      writefln("Job %d At SNP %d" ,job,count);
      if(cpu_num == 1){
        writeln("Running on 1 THREAD");

        compute_snp(job,n,collect,lmm2,reml,q);
        //double[] collect;
        //j,lst = q.get();
        double j;
        double[] lst;
        //info("Job "+str(j)+" finished");
        jobs_completed += 1;
        writeln("GWAS2 ",jobs_completed, " ", snps/1000);
        res~=lst;
      }
          
    }
    collect~=snp; // add SNP to process in batch
  }
  //writeln("Here goes res");
  //writeln(res);

  //////debug("count=%i running=%i collect=%i" % (count,jobs_running,len(collect)))
  if (collect.length>0){
    job += 1;
    //debug("Collect final batch size %i job %i @%i: " % (len(collect), job, count));
    if(cpu_num == 1){
      compute_snp(job,n,collect,lmm2,reml,q);
    }
    else{
      //p.apply_async(compute_snp,(job,n,collect,lmm2,reml));
    }
        
    jobs_running += 1;
    for(int j=0; j < jobs_running; j++){
      double[] lst;
      //j,lst = q.get(True,15);// time out
      writeln("Job "," finished");
      jobs_running -= 1;
      //debug("jobs_running cleanup (-) %d" % jobs_running);
      jobs_completed += 1;
      writeln("GWAS2 ",jobs_completed," ", snps/1000);
      res~=lst;
    }
  }
  ////mprint("Before sort",[res1[0] for res1 in res]);
  ////res = sorted(res,key=lambda x: x[0]);
  ////mprint("After sort",[res1[0] for res1 in res]);
  ////info([len(res1[1]) for res1 in res]);
  double[] ts;
  double[] ps;
  //foreach
  ////ts = [item[0] for j,res1 in res for item in res1];
  ////ps = [item[1] for j,res1 in res for item in res1];
  //writeln(ts,ps);
}

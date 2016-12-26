module simplelmm.gwas;
import std.parallelism : taskPool;
import std.stdio;
import simplelmm.lmm2;
import simplelmm.dmatrix;

//void compute_snp(j,n,snp_ids,lmm2,REML,q = None){
//  result = [];
//  foreach(snp_id;snp_ids){
//    snp,id = snp_id;
//    x = snp.reshape((n,1)); //all the SNPs
//    //# if refit:
//    //#    L.fit(X=snp,REML=REML)
//    ts,ps,beta,betaVar = lmm2.association(x,REML=REML,returnBeta=True);
//    //# result.append(formatResult(id,beta,np.sqrt(betaVar).sum(),ts,ps))
//    result.append( (ts,ps) );
//  }
//  if(!q){
//    q = compute_snp.q;
//  }
//  q.put([j,result]);
//  return j;

//}

//void f_init(q){
//  compute_snp.q = q;
//}

void gwas(ref dmatrix Y, ref dmatrix G, ref dmatrix K, bool restricted_max_likelihood = true, bool refit=false, bool verbose = true){
  //"""
  //GWAS. The G matrix should be n inds (cols) x m snps (rows)
  //"""
  writeln("In gwas.gwas");
  //# matrix_initialize()
  //cpu_num = mp.cpu_count();
  //if(threads.numThreads){
  int cpu_num = std.parallelism.totalCPUs;
  //}
        
  //if(gwas_useCUDA(G)){
  //  cpu_num = 1;
  //}
      
  writefln("Using %u threads", cpu_num);

  bool kfile2 = false;
  bool reml = restricted_max_likelihood;

  writeln("G",G);
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

  LMM2 lmm2 = LMM2(Y,K); //,Kva,Kve,X0,verbose=verbose);
  if(!refit){
    writeln("Computing fit for null model");
    fit_hmax,fit_beta,fit_sigma,fit_LL = lmm2fit(lmm2); // # follow GN model in run_other;
    //writefln("heritability=%0.3f, sigma=%0.3f, LL=%0.5f" % (lmm2.optH,lmm2.optSigma, fit_LL));
  }

  double res = [];

  ////# Set up the pool
  ////# mp.set_start_method('spawn')
  //q = mp.Queue();
  //if(cpu_num > 1){
  //  p = mp.Pool(cpu_num, f_init, [q]);
  //}
       
  //collect = []; //# container for SNPs to be processed in one batch

  int count = 0;
  int job = 0;
  int jobs_running = 0;
  int jobs_completed = 0;
  //foreach(snp; G){
  //  snp_id = (snp,"SNPID");
  //  count += 1;
  //  if(count % 1000 == 0){
  //    job += 1;
  //    writelnf("Job %d At SNP %d" % (job,count));
  //    if(cpu_num == 1){
  //      //debug("Running on 1 THREAD");
  //      compute_snp(job,n,collect,lmm2,reml,q);
  //      collect = [];
  //      j,lst = q.get();
  //      info("Job "+str(j)+" finished");
  //      jobs_completed += 1;
  //      progress("GWAS2",jobs_completed,snps/1000);
  //      res.append((j,lst));
  //    }
  //    else{
  //      p.apply_async(compute_snp,(job,n,collect,lmm2,reml));
  //      jobs_running += 1;
  //      //debug("jobs_running (+) %d", jobs_running);
  //      collect = [];
  //      while(jobs_running >= cpu_num){// throttle maximum jobs
  //         //try:
  //          j,lst = q.get(false,3);
  //          info("Job "+str(j)+" finished");
  //          jobs_completed += 1;
  //          progress("GWAS2",jobs_completed,snps/1000);
  //          res.append((j,lst));
  //          jobs_running -= 1;
  //          //debug("jobs_running (-) %d" % jobs_running);
  //        //except Queue.Empty:
  //          //debug("Queue is empty count=%i running=%i completed=%i collect=%i" % (count,jobs_running,jobs_completed,len(collect)));
  //          //time.sleep(1.0);
  //        if(jobs_running > cpu_num*2){// # sleep longer if many jobs
  //          time.sleep(1.0);
  //        }
  //      }
  //    }
          
  //  }
  //  collect.append(snp_id); // add SNP to process in batch
  //}
      

  ////debug("count=%i running=%i collect=%i" % (count,jobs_running,len(collect)))
  //if (collect.sizeof>0){
  //  job += 1;
  //  //debug("Collect final batch size %i job %i @%i: " % (len(collect), job, count));
  //  if(cpu_num == 1){
  //    compute_snp(job,n,collect,lmm2,reml,q);
  //  }
  //  else{
  //    p.apply_async(compute_snp,(job,n,collect,lmm2,reml));
  //  }
        
  jobs_running += 1;
  double[] collect = [];
  //  foreach(job; range(jobs_running)){
  //    j,lst = q.get(True,15);// time out
  //    info("Job "+str(j)+" finished");
  //    jobs_running -= 1;
  //    //debug("jobs_running cleanup (-) %d" % jobs_running);
  //    jobs_completed += 1;
  //    progress("GWAS2",jobs_completed,snps/1000);
  //    res.append((j,lst));
  //  }
  //}
  ////mprint("Before sort",[res1[0] for res1 in res]);
  ////res = sorted(res,key=lambda x: x[0]);
  ////mprint("After sort",[res1[0] for res1 in res]);
  ////info([len(res1[1]) for res1 in res]);
  ////ts = [item[0] for j,res1 in res for item in res1];
  ////ps = [item[1] for j,res1 in res for item in res1];
  //return ts,ps;
}

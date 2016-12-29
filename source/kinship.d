module simplelmm.kinship;
import simplelmm.dmatrix;
import simplelmm.optmatrix;
import simplelmm.helpers;
import std.stdio;

//dmatrix compute_W(int job, dmatrix G, int n, int snps, int compute_size){
//  //"""
//  //Read 1000 SNPs at a time into matrix and return the result
//  //"""
//	m = compute_size;
//	dmatrix W;
//	//W = np.ones((n,m)) * np.nan; // W matrix has dimensions individuals x SNPs (initially all NaNs)
//	//for j in range(0,compute_size):
//	  pos = job*m + j; //# real position
//	  if pos >= snps{
//	  	W = W[:,range(0,j)]
//	    break;
//	  }
	     
//	  snp = G[job*compute_size+j];
//	  if(snp.var() == 0){
//	    continue;
//	  }
//	  W[:,j] = snp; // set row to list of SNPs
//  }
//	return W;
//}

dmatrix kinshipComp(dmatrix G, int computeSize=1000){

  //# matrix_initialize(useBLAS)
  writeln("G => ");
  prettyPrint(G);
  int n = G.shape[1]; // inds
  int inds = n;
  int m = G.shape[0]; // snps
  int snps = m;
  writeln("%d SNPs",m);
  if(snps>=inds){
    writeln("WARNING: less snps than inds (%d snps, %d inds)",(snps,inds));
  }
  dmatrix K;

  //q = mp.Queue()
  //if threads.multi():
  //    p = mp.Pool(threads.numThreads, f_init, [q])
  //    cpu_num = mp.cpu_count()
  //    info("CPU cores: %i" % cpu_num)
  int iterations = snps/computeSize + 1;

  //results = []
  //K = np.zeros((n,n))  # The Kinship matrix has dimension individuals x individuals

  //completed = 0
  //for job in range(iterations):
  //   info("Processing job %d first %d SNPs" % (job, ((job+1)*computeSize)))
  //   W = compute_W(job,G,n,snps,computeSize)
  //   if W.shape[1] == 0:
  //     continue

  //   if threads.numThreads == 1:
  //      info("Single-core")
  //      print(q)
  //      compute_matrixMult(job,W,q)
  //      j,x = q.get()
  //      debug("Job "+str(j)+" finished")
  //      progress("kinship",j,iterations)
  //      K_j = x
  //      K = K + K_j
  //   else:
  //      # Multi-core
  //      results.append(p.apply_async(compute_matrixMult, (job,W)))
  //      # Do we have a result?
  //      while (len(results)-completed>cpu_num*2):
  //         time.sleep(0.1)
  //         try:
  //            j,x = q.get_nowait()
  //            debug("Job "+str(j)+" finished")
  //            K_j = x
  //            K = K + K_j
  //            completed += 1
  //            progress("kinship",completed,iterations)
  //         except Queue.Empty:
  //            pass

  //if threads.multi():
  //   for job in range(len(results)-completed):
  //      j,x = q.get(True,15)
  //      debug("Job "+str(j)+" finished")
  //      K_j = x
  //      K = K + K_j
  //      completed += 1
  //      progress("kinship",completed,iterations)

  //K = K / float(snps)
  return K;
}
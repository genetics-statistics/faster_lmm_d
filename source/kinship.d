module simplelmm.kinship;
import simplelmm.dmatrix;
import simplelmm.optmatrix;
import simplelmm.helpers;
import std.stdio;

dmatrix compute_W(int job, dmatrix G, int n, int snps, int compute_size){
  //"""
  //Read 1000 SNPs at a time into matrix and return the result
  //"""
	int m = compute_size;
	dmatrix W;
	W = zerosMatrix(n,m); //* np.nan; // W matrix has dimensions individuals x SNPs (initially all NaNs)
	for(int j = 0; j < compute_size; j++){ // j in range(0,compute_size):
	  int pos = job*m + j; //# real position
	  if(pos >= snps){
	  	//W = W[:,range(0,j)];
	    break;
	  }
	     
	  dmatrix snp = G; //[job*compute_size+j];
	  //if(variation(snp) == 0){
	  //  continue;
	  //}
	  //W[:,j] = snp; // set row to list of SNPs
  }
	return W;
}


dmatrix kinship_full(dmatrix G){
  //"""
  //Calculate the Kinship matrix using a full dot multiplication
  //"""
  writeln("Full kinship matrix used");
  //# mprint("kinship_full G",G)
  int m = G.shape[0]; // snps
  int n = G.shape[1]; // inds
  writeln("%d SNPs",m);
  //assert m>n, "n should be larger than m (%d snps > %d inds)" % (m,n)
  //# m = np.dot(G.T,G)
  dmatrix temp = matrixTranspose(G);
  dmatrix l = matrixMult(temp, G);
  l = divideDmatrixNum(l, G.shape[0]);
  writeln("kinship_full K");
  writeln(l.shape);
  prettyPrint(l);
  return l;
}

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
  K = zerosMatrix(n,n);  // The Kinship matrix has dimension individuals x individuals
  //prettyPrint(K);

  //completed = 0
  //for job in rangeArray(iterations):
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
  for(int job = 0; job < iterations; job++){
    writefln("Processing job %d first %d SNPs",job, ((job+1)*computeSize));
    dmatrix W = compute_W(job,G,n,snps,computeSize);
    compute_matrixMult(job,W);
    //j,x = q.get()
    dmatrix x;
    dmatrix K_j = x;
    K = addDmatrix(K,K_j);
  }


  //if threads.multi():
  //   for job in range(len(results)-completed):
  //      j,x = q.get(True,15)
  //      debug("Job "+str(j)+" finished")
  //      K_j = x
  //      K = K + K_j
  //      completed += 1
  //      progress("kinship",completed,iterations)

  //K = K / float(snps)
  K = divideDmatrixNum(K, cast(double)snps);
  return K;
}

int compute_matrixMult(int job, dmatrix W){
  //"""
  //Compute Kinship(W)*j

  //For every set of SNPs matrixMult is used to multiply matrices T(W)*W
  //"""
  dmatrix res = matrixMultT(W, W);
  //if(not q){
  //  q= compute_matrixMult.q
  //}
  //q.put([job,res])
  writeln(res);
  return job;
}

void kvakve(dmatrix K, ref dmatrix Kva, ref dmatrix Kve){
  //"""

  //Obtain eigendecomposition for K and return Kva,Kve where Kva is cleaned
  //of small values < 1e-6 (notably smaller than zero)
  //"""
  writefln("Obtaining eigendecomposition for %dx%d matrix",K.shape[0],K.shape[1]);
  writeln(K);
  //assert (K.T == K).all(); //# raise "K is not symmetric"
  //(dmatrix input,ref double eigenvalue, ref dmatrix dvl, ref dmatrix dvr)
  dmatrix ev;
  eigh(K, ev, Kva, Kve);
  writeln("Kva", Kva);
  writeln("Kve", Kve);

  //if(sum(Kva) < 0){
    //writefln("Cleaning %d eigen values (Kva<0)",(sum(Kva < 0)));
    //Kva[Kva < 1e-6] = 1e-6;
  //}
      
   //return Kva,Kve
}
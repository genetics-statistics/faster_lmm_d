module simplelmm.lmm;
import simplelmm.dmatrix;
import std.stdio;

//void formatResult(id,beta,betaSD,ts,ps){
//  //return "\t".join([str(x) for x in [id,beta,betaSD,ts,ps]]) + "\n";
//}

void run_gwas(string species,int n,int m,ref dmatrix k, ref double[] y, ref dmatrix geno, int cov, bool reml = true,bool refit = false, string inputfn = "", bool new_code = true){
//  //"""
//  //Invoke pylmm using genotype as a matrix or as a (SNP) iterator.
//  //"""
  writeln("run_gwas");
  writeln("pheno", y.sizeof, y[0..5]);
  writeln(geno.shape);
  //assert(geno.shape[0] == y.size, [np.size(geno[0]), y.size]);
  assert(y.sizeof == n);
  //if(k != null){
    writeln(k.shape[0]);
  //}
  //else{
    writeln("No kinship matrix passed in!");
  //}

  writeln(m,geno.shape);
  assert(geno.shape[1] == m);
//  //sys.exit(1)

//  if(species == "human"){
//    writeln("kinship", k );
//    ps, ts = run_human(pheno_vector = y, covariate_matrix = cov, plink_input_file = inputfn, kinship_matrix = k,refit = refit);
//  }
//  else{
//    writeln("geno", geno.shape, geno);

//    if(new_code){
//      ps, ts = run_other_new(n,m,pheno_vector = y, geno = geno, restricted_max_likelihood = reml, refit = refit);
//    }else{
//      ps, ts = run_other_old(pheno_vector = y, genotype_matrix = geno, restricted_max_likelihood = reml, refit = refit);
//    }

}

//  return ps,ts;
//}

//def run_human(pheno_vector,covariate_matrix,plink_input_file,kinship_matrix,refit=False){

//    v = np.isnan(pheno_vector);
//    keep = true - v;
//    keep = keep.reshape(len(keep));

//    identifier = str(uuid.uuid4());

//    //#writeln("pheno_vector: ", pf(pheno_vector))
//    //#writeln("kinship_matrix: ", pf(kinship_matrix))
//    //#writeln("kinship_matrix.shape: ", pf(kinship_matrix.shape))

//    //#lmm_vars = pickle.dumps(dict(
//    //#    pheno_vector = pheno_vector,
//    //#    covariate_matrix = covariate_matrix,
//    //#    kinship_matrix = kinship_matrix
//    //#))
//    //#Redis.hset(identifier, "lmm_vars", lmm_vars)
//    //#Redis.expire(identifier, 60*60)

//    if(v.sum()){
//      pheno_vector = pheno_vector[keep];
//      writeln("pheno_vector shape is now: ", pf(pheno_vector.shape));
//      covariate_matrix = covariate_matrix[keep,$];
//      writeln("kinship_matrix shape is: ", pf(kinship_matrix.shape));
//      writeln("keep is: ", pf(keep.shape));
//      kinship_matrix = kinship_matrix[keep,$][0,keep];
//    }


//    writeln("kinship_matrix:", pf(kinship_matrix));

//    n = kinship_matrix.shape[0];
//    writeln("n is:", n);
//    lmm_ob = LMM(pheno_vector, kinship_matrix, covariate_matrix);
//    lmm_ob.fit();


//    //# Buffers for pvalues and t-stats
//    p_values = [];
//    t_stats = [];

//    //#writeln("input_file: ", plink_input_file)

//    with(Bench("Opening and loading pickle file")){
//      //with(gzip.open(plink_input_file, "rb") as input_file)){
//        data = pickle.load(input_file);
//      //}
//    }
        

//    plink_input = data["plink_input"];

//    //#plink_input.getSNPIterator()
//    //with Bench("Calculating numSNPs"):
//    //    total_snps = data['numSNPs']

//    //with Bench("snp iterator loop"):
//    //    count = 0

//    //    with Bench("Create list of inputs"):
//    //        inputs = list(plink_input)

//    //    with Bench("Divide into chunks"):
//    //        results = chunks.divide_into_chunks(inputs, 64)

//    //    result_store = []

//    //    key = "plink_inputs"

//    //    //# Todo: Delete below line when done testing
//    //    Redis.delete(key)

//    //    timestamp = datetime.datetime.utcnow().isoformat()

//    //    //# Pickle chunks of input SNPs (from Plink interator) and compress them
//    //    //#writeln("Starting adding loop")
//    //    for part, result in enumerate(results):
//    //        #data = pickle.dumps(result, pickle.HIGHEST_PROTOCOL)
//    //        holder = pickle.dumps(dict( identifier = identifier, part = part, timestamp = timestamp, result = result), pickle.HIGHEST_PROTOCOL)

//    //        //#writeln("Adding:", part)
//    //        Redis.rpush(key, zlib.compress(holder))
//    //    //#writeln("End adding loop")
//    //    //#writeln("***** Added to {} queue *****".format(key))
//    //    for snp, this_id in plink_input:
//    //        //#with Bench("part before association"):
//    //        //#if count > 1000:
//    //        //#    break
//    //        count += 1
//    //        progress("human",count,total_snps)

//    //        //#with Bench("actual association"):
//    //        ps, ts = human_association(snp, n, keep, lmm_ob, pheno_vector, covariate_matrix, kinship_matrix, refit);

//    //        //#with Bench("after association"):
//    //        p_values.append(ps)
//    //        t_stats.append(ts)

//    return p_values, t_stats;
//}

//def run_other_old(pheno_vector, genotype_matrix, restricted_max_likelihood=True, refit=False){

//    //"""Takes the phenotype vector and genotype matrix and returns a set of p-values and t-statistics

//    //restricted_max_likelihood -- whether to use restricted max likelihood; True or False
//    //refit -- whether to refit the variance component for each marker

//    //"""

//    writeln("Running the original LMM engine in run_other (old)");
//    writeln("REML=",restricted_max_likelihood," REFIT=",refit);
//    //with Bench("Calculate Kinship"):
//        //kinship_matrix,genotype_matrix = calculate_kinship_new(genotype_matrix)

//    writeln("kinship_matrix: ", pf(kinship_matrix));
//    writeln("kinship_matrix.shape: ", pf(kinship_matrix.shape));

//    //# with Bench("Create LMM object"):
//    //#     lmm_ob = LMM(pheno_vector, kinship_matrix)

//    //# with Bench("LMM_ob fitting"):
//    //#     lmm_ob.fit()

//    writeln("run_other_old genotype_matrix: ", genotype_matrix.shape);
//    writeln(genotype_matrix);

//    with(Bench("Doing GWAS")){
//      t_stats, p_values = GWAS(pheno_vector, genotype_matrix.T, kinship_matrix, restricted_max_likelihood=True, refit=False);
//    }
        
//    Bench().report();
//    return p_values, t_stats;
//}

//def run_other_new(n,m,pheno_vector, geno, restricted_max_likelihood=True, refit=False){

//    //"""Takes the phenotype vector and genotype matrix and returns a set of p-values and t-statistics

//    //restricted_max_likelihood -- whether to use restricted max likelihood; True or False
//    //refit -- whether to refit the variance component for each marker

//    //"""

//    writeln("Running the new LMM2 engine in run_other_new");
//    writeln("REML=",restricted_max_likelihood," REFIT=",refit);

//    //# Adjust phenotypes
//    n,Y,keep = phenotype.remove_missing_new(n,pheno_vector);

//    //# if options.maf_normalization:
//    //#     G = np.apply_along_axis( genotype.replace_missing_with_MAF, axis=0, arr=g )
//    //#     writeln "MAF replacements: \n",G
//    //# if not options.skip_genotype_normalization:
//    //# G = np.apply_along_axis( genotype.normalize, axis=1, arr=G)

//    geno = geno[0,keep];
//    with(Bench("Calculate Kinship")){
//      K,G = calculate_kinship_new(geno);
//    }
       

//    writeln("kinship_matrix: ", pf(K));
//    writeln("kinship_matrix.shape: ", pf(K.shape));

//    //# with Bench("Create LMM object"):
//    //#     lmm_ob = lmm2.LMM2(Y,K)
//    //# with Bench("LMM_ob fitting"):
//    //#     lmm_ob.fit()

//    writeln("run_other_new genotype_matrix: ", G.shape);
//    writeln(G);

//    with(Bench("Doing GWAS")){
//      t_stats, p_values = gwas.gwas(Y, G, K, restricted_max_likelihood=True, refit=False,verbose=True);
//    }
        
//    Bench().report();
//    return p_values, t_stats;
//}

//def matrixMult(A,B){

//    //# If there is no fblas then we will revert to np.dot()

//    //try:
//    //    linalg.fblas
//    //except AttributeError:
//    //    return np.dot(A,B)

//    //writeln("A is:", pf(A.shape))
//    //writeln("B is:", pf(B.shape))

//    // If the matrices are in Fortran order then the computations will be faster
//    // when using dgemm.  Otherwise, the function will copy the matrix and that takes time.
//    //if not A.flags['F_CONTIGUOUS']:
//    //   AA = A.T
//    //   transA = True
//    //else:
//    //   AA = A
//    //   transA = False

//    //if not B.flags['F_CONTIGUOUS']:
//    //   BB = B.T
//    //   transB = True
//    //else:
//    //   BB = B
//    //   transB = False

//    return linalg.fblas.dgemm(alpha=1.,a=AA,b=BB,trans_a=transA,trans_b=transB);
//}
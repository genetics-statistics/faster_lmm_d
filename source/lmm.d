module simplelmm.lmm;
import simplelmm.dmatrix;
import simplelmm.gwas;
import std.stdio;
import simplelmm.helpers;

//void formatResult(id,beta,betaSD,ts,ps){
//  //return "\t".join([str(x) for x in [id,beta,betaSD,ts,ps]]) + "\n";
//}

void run_gwas(string species,int n,int m,ref dmatrix k, ref double[] y, ref dmatrix geno){
    //int cov, bool reml = true,bool refit = false, string inputfn = "", bool new_code = true){
//  //"""
//  //Invoke pylmm using genotype as a matrix or as a (SNP) iterator.
//  //"""
  int cov;
  bool reml = true;
  bool refit = false;
  string inputfn = "";
  bool new_code = true;
  
  writeln("run_gwas");
  writeln("pheno ", y.length," ", y[0..5]);
  writeln(geno.shape);
  //assert(geno.shape[0] == y.size, [np.size(geno[0]), y.size]);
  assert(y.length == n);
  //if(k != null){
  //writeln(k.shape[0]);
  //}
  //else{
  writeln("No kinship matrix passed in!");
  //}

  writeln(m,geno.shape);
  assert(geno.shape[1] == m);
//  //sys.exit(1)

  if(species == "human"){
    writeln("kinship", k );
    //ps, ts = 
    //run_human(y, cov, inputfn, k, refit);
  }
  else{
    writeln("geno");
    double ps, ts;
    if(new_code){ 
      run_other_new(n, m, y, geno, reml, refit);
    }else{
      run_other_new(n, m, y, geno, reml, refit);
    }
}

}

void run_human(ref double[] pheno_vector, ref dmatrix covariate_matrix, string plink_input_file, ref dmatrix kinship_matrix, bool refit=false){

    bool[] v = isnan(pheno_vector);
    auto keep = 0;
     //= true - v;

    //  reshape for a vector
    //double[] keep = keep.reshape(len(keep));

    //identifier = str(uuid.uuid4());

    //writeln("pheno_vector: ", pf(pheno_vector))
    //writeln("kinship_matrix: ", pf(kinship_matrix))
    //writeln("kinship_matrix.shape: ", pf(kinship_matrix.shape))

    //lmm_vars = pickle.dumps(dict(
    //    pheno_vector = pheno_vector,
    //    covariate_matrix = covariate_matrix,
    //    kinship_matrix = kinship_matrix
    //))
    //Redis.hset(identifier, "lmm_vars", lmm_vars)
    //Redis.expire(identifier, 60*60)

    // there may be a need to create a vector struct

    //if(sum(v)){
      //pheno_vector = pheno_vector[keep];
      //writeln("pheno_vector shape is now: ", pf(pheno_vector.shape));
      //covariate_matrix = covariate_matrix[keep,$];
      //writeln("kinship_matrix shape is: ", pf(kinship_matrix.shape));
      //writeln("keep is: ", pf(keep.shape));
      //kinship_matrix = kinship_matrix[keep,$][0,keep];
    //}


    writeln("kinship_matrix:", kinship_matrix);

    int n = kinship_matrix.shape[0];
    writeln("n is:", n);
    LMM lmm_ob = LMM(pheno_vector, kinship_matrix, covariate_matrix);
    dmatrix a;
    fit(lmm_ob, a , 100, true);


    //# Buffers for pvalues and t-stats
    double[] p_values = [];
    double[] t_stats = [];

    //#writeln("input_file: ", plink_input_file)

    //with(writeln("Opening and loading pickle file")){
    //  //with(gzip.open(plink_input_file, "rb") as input_file)){
    //    data = pickle.load(input_file);
    //  //}
    //}
        

    //plink_input = data["plink_input"];

    //#plink_input.getSNPIterator()
    //with Bench("Calculating numSNPs"):
    //    total_snps = data['numSNPs']

    //with Bench("snp iterator loop"):
    //    count = 0

    //    with Bench("Create list of inputs"):
    //        inputs = list(plink_input)

    //    with Bench("Divide into chunks"):
    //        results = chunks.divide_into_chunks(inputs, 64)

    //    result_store = []

    //    key = "plink_inputs"

    //    //# Todo: Delete below line when done testing
    //    Redis.delete(key)

    //    timestamp = datetime.datetime.utcnow().isoformat()

    //    //# Pickle chunks of input SNPs (from Plink interator) and compress them
    //    //#writeln("Starting adding loop")
    //    for part, result in enumerate(results):
    //        #data = pickle.dumps(result, pickle.HIGHEST_PROTOCOL)
    //        holder = pickle.dumps(dict( identifier = identifier, part = part, timestamp = timestamp, result = result), pickle.HIGHEST_PROTOCOL)

    //        //#writeln("Adding:", part)
    //        Redis.rpush(key, zlib.compress(holder))
    //    //#writeln("End adding loop")
    //    //#writeln("***** Added to {} queue *****".format(key))
    //    for snp, this_id in plink_input:
    //        //#with Bench("part before association"):
    //        //#if count > 1000:
    //        //#    break
    //        count += 1
    //        progress("human",count,total_snps)

    //        //#with Bench("actual association"):
    //        ps, ts = human_association(snp, n, keep, lmm_ob, pheno_vector, covariate_matrix, kinship_matrix, refit);

    //        //#with Bench("after association"):
    //        p_values.append(ps)
    //        t_stats.append(ts)

    //return p_values, t_stats;
}

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

void run_other_new(ref int n, ref int m, ref double[] pheno_vector, ref dmatrix geno, bool restricted_max_likelihood= true, bool refit = false){

    //"""Takes the phenotype vector and genotype matrix and returns a set of p-values and t-statistics

    //restricted_max_likelihood -- whether to use restricted max likelihood; True or False
    //refit -- whether to refit the variance component for each marker

    //"""

    writeln("Running the new LMM2 engine in run_other_new");
    writeln("REML=",restricted_max_likelihood," REFIT=",refit);

    //# Adjust phenotypes
    double[] Y;
    bool[] keep;
    simplelmm.phenotype.remove_missing_new(Y,keep,n,pheno_vector);

    ////# if options.maf_normalization:
    ////#     G = np.apply_along_axis( genotype.replace_missing_with_MAF, axis=0, arr=g )
    ////#     writeln "MAF replacements: \n",G
    ////# if not options.skip_genotype_normalization:
    ////# G = np.apply_along_axis( genotype.normalize, axis=1, arr=G)

    //geno = newDmatrix(geno,0,cast(int)keep);
    dmatrix K, G;
    writeln("Calculate Kinship");
      //K,G = 
    calculate_kinship_new(K,G,geno);
    //}
       

    //writeln("kinship_matrix: ", K);
    //writeln("kinship_matrix.shape: ", K.shape);

    ////# with Bench("Create LMM object"):
    ////#     lmm_ob = lmm2.LMM2(Y,K)
    ////# with Bench("LMM_ob fitting"):
    ////#     lmm_ob.fit()

    //writeln("run_other_new genotype_matrix: ", G.shape);
    //writeln(G);

    ////with(Bench("Doing GWAS")){
    //  //t_stats, p_values = 
    //  //gwas(Y, G, K, restricted_max_likelihood=True, refit=False,verbose=True);
    //  gwas(Y, G, K, true, false, true);
    //}
        
    //Bench().report();
    //return p_values, t_stats;
}


struct LMM{
  double[] Y;
  dmatrix K;
  dmatrix Kva;
  dmatrix Kve;
  dmatrix X0;
  bool verbose;
  dmatrix Yt;
  dmatrix X0t;
  dmatrix X0t_stack;
  dmatrix q;
  double[] H;
  double[] L;
  double[] LLs;
  double optLL;
  double optBeta;
  double optSigma;

  this(double[] Y, dmatrix K, dmatrix Kva,dmatrix Kve,double X0,bool verbose){
    this.Y = Y;
    this.K = K;
    this.Kva = Kva;
    this.Kve = Kve;
    this.verbose = false;
  }

  this(double[] Y, dmatrix K, dmatrix Kva,){
    this.Y = Y;
    this.K = K;
    this.Kva = Kva;
    this.verbose = false;
  }
}

void fit(ref LMM lmmobject,ref dmatrix X, double ngrids=100, bool REML=true){

  //"""
  //   Finds the maximum-likelihood solution for the heritability (h) given the current parameters.
  //   X can be passed and will transformed and concatenated to X0t.  Otherwise, X0t is used as
  //   the covariate matrix.

  //   This function calculates the LLs over a grid and then uses .getMax(...) to find the optimum.
  //   Given this optimum, the function computes the LL and associated ML solutions.
  //"""

    //if(X is None){ 
    //  X = lmmobject.X0t;
    //}
    //else{
    //   //#X = np.hstack([lmmobject.X0t,matrixMult(lmmobject.Kve.T, X)])
    //  lmmobject.X0t_stack[sval,(lmmobject.q)] = matrixMult(lmmobject.Kve.T,X)[sval,0];
    //  X = lmmobject.X0t_stack;
    //}

    //auto H = array(range(ngrids)) / float(ngrids);
    //auto L = array(range(ngrids)) / float(ngrids);
    //np.array([lmmobject.LL(h,X,stack=False,REML=REML)[0] for h in H]);
    //lmmobject.LLs = L;

    //hmax = getMax(H,X,REML);
    //getLL( L, beta,  sigma, betaSTDERR, hmax, X, false, REML);

    //lmmobject.H = H;
    ////false.optH = hmax.sum();
    //lmmobject.optLL = L;
    //lmmobject.optBeta = beta;
    //lmmobject.optSigma = sigma.sum();

    //# debug(["hmax",hmax,"beta",beta,"sigma",sigma,"LL",L])
    //return hmax,beta,sigma,L;
}

void calculate_kinship_new(ref dmatrix K, ref dmatrix G, ref dmatrix genotype_matrix){
    //"""
    //Call the new kinship calculation where genotype_matrix contains
    //inds (columns) by snps (rows).
    //"""
    //assert type(genotype_matrix) is np.ndarray;
    writeln("call genotype.normalize");
    //G = np.apply_along_axis( genotype.normalize, axis=1, arr=genotype_matrix);
    //writeln("G",genotype_matrix);
    writeln("call calculate_kinship_new");
    //if kinship_useCUDA(G) or kinship_doCalcFull(G):
    //    try:
    //        return kinship_full(G),G
    //    except:
    //        pass # when out of memory try the iterator version
    //return kinship(G),G
}
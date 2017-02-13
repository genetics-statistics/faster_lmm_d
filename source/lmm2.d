module simplelmm.lmm2;
import dstats.distrib;
import simplelmm.dmatrix;
import simplelmm.optmatrix;
import simplelmm.helpers;
import simplelmm.kinship;
import std.stdio;

////void calculateKinship(W,center=False){
////  //"""
////  //   W is an n x m matrix encoding SNP minor alleles.
////  //   This function takes a matrix oF SNPs, imputes missing values with the maf,
////  //   normalizes the resulting vectors and returns the RRM matrix.
////  //"""
////  n = W.shape[0];
////  m = W.shape[1];
////  keep = [];
////  foreach(i;range(m)){
////    mn = W[true - np.isnan(W[0..i]),i].mean();
////    W[np.isnan(W[0..i]),i] = mn;
////    vr = W[0..i].var();
////    if(vr == 0){continue;}
////    keep.append(i);
////    W[0..i] = (W[0..i] - mn) / np.sqrt(vr);
////  }
////  W = W[0..keep];
////  K = matrixMult(W,W.T) * 1.0/float(m);
////  if(center){
////    P = np.diag(np.repeat(1,n)) - 1/float(n) * np.ones((n,n));
////    S = np.trace(matrixMult(matrixMult(P,K),P));
////    K_n = (n - 1)*K / S;
////    return K_n;
////  }
////  return K;
////}
////void GWAS(Y, X, K, Kva=[], Kve=[], X0=None, REML=true, refit=False){
////  //"""
////  //  Performs a basic GWAS scan using the LMM.  This function
////  //  uses the LMM module to assess association at each SNP and
////  //  does some simple cleanup, such as removing missing individuals
////  //  per SNP and re-computing the eigen-decomp
////  //  Y - n x 1 phenotype vector
////  //  X - n x m SNP matrix (genotype matrix)
////  //  K - n x n kinship matrix
////  //  Kva,Kve = linalg.eigh(K) - or the eigen vectors and values for K
////  //  X0 - n x q covariate matrix
////  //  REML - use restricted maximum likelihood
////  //  refit - refit the variance component fors each SNP
////  //"""
////  n = X.shape[0];
////  m = X.shape[1];
////  writeln("Initialize GWAS");
////  writeln("genotype matrix n is:", n);
////  writeln("genotype matrix m is:", m);
////  if(X0 is None){
////    X0 = np.ones((n,1));
////  }
////  //# Remove missing values in Y and adjust associated parameters
////  v = np.isnan(Y);
////  if(v.sum()){
////    keep = true - v;
////    keep = keep.reshape((-1,x));
////    Y = Y[keep];
////    X = X[keep,x];
////    X0 = X0[keep,x];
////    K = K[keep,x][x,keep];
////    Kva = [];
////    Kve = [];
////  }
////  if(len(Y) == 0){
////    return np.ones(m)*np.nan,np.ones(m)*np.nan;
////  }
////  L = LMM(Y,K,Kva,Kve,X0);
////  if(!refit){
////    L.fit();
////  } 
////  PS = [];
////  TS = [];
////  n = X.shape[0];
////  m = X.shape[1];
////  foreach(i; range(m)){
////    x = X[la,i].reshape((n,1));
////    v = np.isnan(x).reshape((-1,ra));
////    if(v.sum()){
////      keep = true - v;
////      xs = x[keep,m];
////      if(xs.var() == 0){
////        PS.append(np.nan);
////        TS.append(np.nan);
////        continue;
////      }
////      Ys = Y[keep];
////      X0s = X0[keep,m];
////      Ks = K[keep,m][n,keep];
////      Ls = LMM(Ys,Ks,X0=X0s);
////      if(refit){
////        Ls.fit(X=xs);
////      }
////      else{
////        Ls.fit();
////      }
////      ts,ps = Ls.association(xs,REML=REML);
////    }
////    else{
////      if(x.var() == 0){
////        PS.append(np.nan);
////        TS.append(np.nan);
////        continue;
////      }
////      if(refit){
////        L.fit(X=x);
////      }
////      ts,ps = L.association(x,REML=REML);
////    }
////    PS.append(ps);
////    TS.append(ts);
////  }
////  return TS,PS;
////}

struct LMM2{
  bool init = false;
  dmatrix Y;
  dmatrix Y1;
  dmatrix K;
  dmatrix Kva;
  dmatrix Kve;
  dmatrix X0;
  bool verbose = false;
  dmatrix Yt;
  dmatrix X0t;
  dmatrix X0t_stack;
  int q;
  dmatrix H;
  double N;

  dmatrix optLL;
  dmatrix optBeta;
  double optSigma;
  dmatrix LLs;
  double optH;

  //The constructor takes a phenotype vector or array Y of size n. It
  //takes a kinship matrix K of size n x n.  Kva and Kve can be
  //computed as Kva,Kve = linalg.eigh(K) and cached.  If they are
  //not provided, the constructor will calculate them.  X0 is an
  //optional covariate matrix of size n x q, where there are q
  //covariates.  When this parameter is not provided, the
  //constructor will set X0 to an n x 1 matrix of all ones to
  //represent a mean effect.

  this(double[] Y, dmatrix K, dmatrix Kva, dmatrix Kve, dmatrix X0,bool verbose){
    if(X0.init == false){
      writeln("Initializing LMM2...");
      X0 = onesMatrix(cast(int)Y.length,1);
    }
    this.verbose = verbose;

  //  Y = y;
  //bool[] v;
  //v = isnan(y);
  //writeln(v);
  //keep = negateBool(v);
    bool[] v = isnan(Y);
    bool[] x = negateBool(v);
    

    //if not x.sum() == len(Y):
    //     if self.verbose: sys.stderr.write("Removing %d missing values from Y\n" % ((True - x).sum()))
    //     Y = Y[x]
    //     K = K[x,:][:,x]
    //     X0 = X0[x,:]
    //     Kva = []
    //     Kve = []
    //  self.nonmissing = x

    writeln("this K is:", K.shape, K);

    //if len(Kva) == 0 or len(Kve) == 0:
    //      # if self.verbose: sys.stderr.write("Obtaining eigendecomposition for %dx%d matrix\n" % (K.shape[0],K.shape[1]) )
    //      begin = time.time()
    //      # Kva,Kve = linalg.eigh(K)
    //      Kva,Kve = kinship.kvakve(K)
    //      end = time.time()
    //      if self.verbose: sys.stderr.write("Total time: %0.3f\n" % (end - begin))
    //      print("sum(Kva),sum(Kve)=",sum(Kva),sum(Kve))

    kvakve(K, Kve, Kva);

    this.init = true;
    this.K = K;
    this.Kva = Kva;
    this.Kve = Kve;
    this.N = K.shape[0];
    this.Y =  dmatrix([K.shape[0],1] ,Y); // .reshape((self.N,1))
    this.X0 = X0;
    bool[] com = compareGt(Kva, 1e-6);
    //if(simplelmm.helpers.sum(com)){
    //  writeln("Cleaning eigen values");
    //  foreach(ref double element; Kva.elements){
    //    if(element < 1e-6)
    //    {
    //      element = 1e-6;
    //    }
    //  }
      
    //}
    //pPrint(Kva);
    lmm2transform(this);

  }
}



  void lmm2transform(ref LMM2 lmmobject){
    //"""
    //   Computes a transformation on the phenotype vector and the covariate matrix.
    //   The transformation is obtained by left multiplying each parameter by the transpose of the
    //   eigenvector matrix of K (the kinship).
    //"""
    writeln("In lmm2transform");
    dmatrix KveT = matrixTranspose(lmmobject.Kve);
    lmmobject.Yt = matrixMult(KveT, lmmobject.Y);
    lmmobject.X0t = matrixMult(KveT, lmmobject.X0);
    lmmobject.X0t_stack = horizontallystack(lmmobject.X0t, onesMatrix(cast(int)lmmobject.N,1));
    lmmobject.q = lmmobject.X0t.shape[1];
  }

  void getMLSoln(ref dmatrix beta, ref double sigma,ref dmatrix Q, ref dmatrix XX_i, ref dmatrix XX, ref LMM2 lmmobject,ref double h, ref dmatrix X){

    //"""
    //   Obtains the maximum-likelihood estimates for the covariate coefficients (beta),
    //   the total variance of the trait (sigma) and also passes intermediates that can
    //   be utilized in other functions. The input parameter h is a value between 0 and 1 and represents
    //   the heritability or the proportion of the total variance attributed to genetics.  The X is the
    //   covariate matrix.
    //"""

    dmatrix S;
    dmatrix temppp = addDmatrixNum(multiplyDmatrixNum(lmmobject.Kva,h),(1.0 - h));    S = divideNumDmatrix(1,temppp);
    dmatrix Xt = multiplyDmatrix(X, S);
    Xt = matrixTranspose(Xt);
    XX = matrixMult(Xt,X);
    XX_i = inverse(XX);
    dmatrix temp = matrixMult(XX_i,Xt);
    beta =  matrixMult(temp,lmmobject.Yt);
    dmatrix temp2 = matrixMult(X,beta);
    dmatrix Yt = subDmatrix(lmmobject.Yt, temp2);
    dmatrix YtT = matrixTranspose(Yt);
    S = matrixTranspose(S);
    dmatrix YtTS = multiplyDmatrix(YtT, S);
    Q = matrixMult(YtTS,Yt);
    sigma = Q.elements[0] * 1.0 / (cast(double)(lmmobject.N) - cast(double)(X.shape[1]));
  }

  double LL_brent(ref LMM2 lmmobject, ref double h, ref dmatrix X, ref bool REML){
      //#brent will not be bounded by the specified bracket.
      //# I return a large number if we encounter h < 0 to avoid errors in LL computation during the search.
    if(h < 0){return 1e6;}
    dmatrix beta, betaVAR;
    double sigma;
    dmatrix l;
    return -getLL(l, beta,sigma,betaVAR, lmmobject, h,X,false,REML);
  }

  double getLL(ref dmatrix L, ref dmatrix beta, ref double sigma, ref dmatrix betaVAR, ref LMM2 lmmobject, ref double h, ref dmatrix X, bool stack=true, bool REML=false){
      //"""
      //   Computes the log-likelihood for a given heritability (h).  If X==None, then the
      //   default X0t will be used.  If X is set and stack=True, then X0t will be matrix concatenated with
      //   the input X.  If stack is false, then X is used in place of X0t in the LL calculation.
      //   REML is computed by adding additional terms to the standard LL and can be computed by setting REML=True.
      //"""
      if(X.init != true){

        X = lmmobject.X0t;
      }
      //else if(stack){
      //  //lmmobject.X0t_stack[sval,(lmmobject.q)] = matrixMult(lmmobject.Kve.T,X)[sval,0];
      //  X = lmmobject.X0t_stack;
      //}
      double n = cast(double)lmmobject.N;
      double q = cast(double)X.shape[1];
      //beta,sigma,Q, = 
      dmatrix Q,XX_i,XX;
      getMLSoln(beta,sigma,Q,XX_i,XX, lmmobject, h,X);
      //writeln(beta);
      betaVAR=XX_i;
      double LL  = n * std.math.log(2*std.math.PI) + sum(logDmatrix((addDMatrixNum( multiplyDmatrixNum(lmmobject.Kva,h),(1-h) ) )).elements)+
      + n + n * std.math.log((1.0/n) * Q.elements[0]); //Q

      LL = -0.5 * LL;

      if(REML){
        double LL_REML_part = 0;
        dmatrix XT = matrixTranspose(X);
        double aloo = det(matrixMult(XT, X));

        LL_REML_part = q*std.math.log(2.0*std.math.PI*sigma) + std.math.log(aloo) - std.math.log(det(XX));
        LL = LL + 0.5*LL_REML_part;
      }
      //double LLsum = sum(LL);
      //writeln("beta=",beta.acc(0,0)," sigma=",sigma," LL=",LL);
      L = dmatrix([1,1],[LL]);
      
      return LL;
  }

  double optimizeBrent(){
    return 0;
  }

  double getMax(ref LMM2 lmmobject, ref dmatrix H, ref dmatrix X, bool REML=false){

    //"""
    //   Helper functions for .fit(...).
    //   This function takes a set of LLs computed over a grid and finds possible regions
    //   containing a maximum.  Within these regions, a Brent search is performed to find the
    //   optimum.

    //"""
    int n = cast(int)lmmobject.LLs.shape[0];
    double[] HOpt;
    for(int i=1; i< n-2; i++){
      if(lmmobject.LLs.elements[i-1] < lmmobject.LLs.elements[i] && lmmobject.LLs.elements[i] > lmmobject.LLs.elements[i+1]){
        //HOpt ~= optimizeBrent(lmmobject.LL_brent,X,REML);
          //,brack=(H[i-1],H[i+1])));
        if(std.math.isNaN(HOpt[$])){
          HOpt[$] = H.elements[i-1];
        }
      }
    }

    if(HOpt.length > 1){
      //if(self.verbose){sys.stderr.write("NOTE: Found multiple optima.  Returning first...\n");}
      return HOpt[0];
    }
    else if(HOpt.length == 1){
      return HOpt[0];
    }
    else if(lmmobject.LLs.elements[0] > lmmobject.LLs.elements[n-1]){
      return H.elements[0];
    }
    else{
      return H.elements[n-1];
    }
  }

  void lmm2fit(ref double fit_hmax, ref dmatrix fit_beta, ref double fit_sigma, ref double fit_LL, ref LMM2 lmmobject,ref dmatrix X, double ngrids=100, bool REML=true){

    //"""
    //   Finds the maximum-likelihood solution for the heritability (h) given the current parameters.
    //   X can be passed and will transformed and concatenated to X0t.  Otherwise, X0t is used as
    //   the covariate matrix.

    //   This function calculates the LLs over a grid and then uses .getMax(...) to find the optimum.
    //   Given this optimum, the function computes the LL and associated ML solutions.
    //"""

    if(X.init == false){ 
      X = lmmobject.X0t;
    }
    else{
      dmatrix kveT = matrixTranspose(lmmobject.Kve);
      dmatrix KveTX = matrixMult(kveT , X);
      //lmmobject.X0t_stack[sval,(lmmobject.q)] = matrixMult(lmmobject.Kve.T,X)[sval,0];
      X = lmmobject.X0t_stack;
      //self.X0t_stack[:,(self.q)] = matrixMult(self.Kve.T,X)[:,0]
      //   X = self.X0t_stack
    }
    double[] Harr;
    for(int m = 0; m < ngrids; m++){
      Harr ~= m / cast(double)ngrids;
    }

    dmatrix  beta, betaVAR;
    double sigma;
    dmatrix L;
    double[] elm;
    foreach(h; Harr){
      elm ~= getLL(L, beta,sigma,betaVAR,lmmobject,h,X,false,REML);
    }
    L = dmatrix([cast(int)elm.length,1],elm);
    lmmobject.LLs = L;
    lmmobject.H = dmatrix([cast(int)Harr.length,1],Harr);
    double hmax = getMax(lmmobject, lmmobject.H, X, REML);

    double sum = getLL(L, beta,sigma,betaVAR,lmmobject,hmax,X,false,REML);

    
    lmmobject.optH = hmax;
    lmmobject.optLL = L;
    lmmobject.optBeta = beta;
    lmmobject.optSigma = sigma;

    //# debug(["hmax",hmax,"beta",beta,"sigma",sigma,"LL",L])
    //return hmax,beta,sigma,L;
    fit_beta = beta;
    fit_sigma = sigma;
    fit_LL = sum;
  }

  void lmm2association(ref double psNum, ref double tsNum, ref LMM2 lmmobject, ref dmatrix X, ref double h, bool stack=true, bool REML=true, bool returnBeta=false){
    //"""
    //  Calculates association statitics for the SNPs encoded in the vector X of size n.
    //  If h is None, the optimal h stored in optH is used.

    //"""
    if(false){
      writeln("X=",X);
      writeln("h=",h);
      writeln("q=",lmmobject.q);
      writeln("lmmobject.Kve=",lmmobject.Kve);
      writeln("X0t_stack=",lmmobject.X0t_stack);
    }

    if(stack){
      dmatrix kvet = matrixTranspose(lmmobject.Kve);
      dmatrix m = matrixMult(kvet,X);
      setRow(lmmobject.X0t_stack,lmmobject.q,m);
      X = lmmobject.X0t_stack;
    }
    if(h.init == false){h = lmmobject.optH;}

    dmatrix beta, betaVAR;
    double sigma; 
    dmatrix L;
    //writeln("In lmm2association");
    
    getLL(L,beta,sigma,betaVAR, lmmobject,h, X ,false,REML);
    //writeln("L is ", L.elements[0],  " sigma is  " ,  sigma, "  betaVAR is" , betaVAR, "h is ", h);
    int q  = cast(int)beta.elements.length;

    double ts,ps;
    //writeln(betaVAR);
    tstat(ts, ps, lmmobject, beta.elements[q-1], betaVAR.acc(q-1,q-1),sigma,q);
    tsNum = ts;
    psNum = ps;

    //debug("ts=%0.3f, ps=%0.3f, heritability=%0.3f, sigma=%0.3f, LL=%0.5f" % (ts,ps,h,sigma,L))
    //if(returnBeta){return ts,ps,beta[q-1].sum(),betaVAR[q-1,q-1].sum()*sigma;}
    //return ts,ps;
  }

  void tstat(ref double ts, ref double ps, ref LMM2 lmmobject, double beta, double var, double sigma, double q, bool log=false){

    //"""
    //   Calculates a t-statistic and associated p-value given the estimate of beta and its standard error.
    //   This is actually an F-test, but when only one hypothesis is being performed, it reduces to a t-test.
    //"""S
    ts =  beta / std.math.sqrt(var*sigma);
    //#ps = 2.0*(1.0 - stats.t.cdf(np.abs(ts), lmmobject.N-q))
    //# sf == survival function - this is more accurate -- could also use logsf if the precision is not good enough
    if(log){
      //double psNum = 2.0 + (stats.t.logsf(np.abs(ts), lmmobject.N-q));
    }
    else{
      //check the sign of ts.elements[0]
      ps = 2.0*(normalCDF(-std.math.abs(ts)));
    }
    //if(!(ts.elements.length == 1) || !(ps.elements.length == 1)){
    //  writeln("Something bad happened :(");
    //}
    //writeln("ts = ", ts, " beta = ", beta,  " var = ", var,  " sigma = ", sigma, " ps = ", ps );
  }

  void plotFit( LMM2 lmmobject, string color="b-", string title=""){

    //"""
    //   Simple function to visualize the likelihood space.  It takes the LLs
    //   calcualted over a grid and normalizes them by subtracting off the mean and exponentiating.
    //   The resulting "probabilities" are normalized to one and plotted against heritability.
    //   This can be seen as an approximation to the posterior distribuiton of heritability.

    //   For diagnostic purposes this lets you see if there is one distinct maximum or multiple
    //   and what the variance of the parameter looks like.
    //"""

    //mx = lmmobject.LLs.max();
    //p = np.exp(lmmobject.LLs - mx);
    //p = p/p.sum();

    //pl.plot(lmmobject.H,p,color);
    //pl.xlabel("Heritability");
    //pl.ylabel("Probability of data");
    //pl.title(title);
  }

  void meanAndVar(ref LMM2 lmmobject){
    //mx = lmmobject.LLs.max();
    //p = np.exp(lmmobject.LLs - mx);
    //p = p/p.sum();

    //mn = (lmmobject.H * p).sum();
    //vx = ((lmmobject.H - mn)**2 * p).sum();

    //return mn,vx;
  }

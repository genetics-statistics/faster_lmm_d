module simplelmm.lmm2;
import dstats.distrib;
import simplelmm.dmatrix;
import simplelmm.optmatrix;
import simplelmm.helpers;
import simplelmm.kinship;
import std.stdio;
import std.typecons;

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
    writeln("This is Y");
    writeln(Y);

    if(X0.init == false){
      writeln("Initializing LMM2...");
      X0 = onesMatrix(cast(int)Y.length,1);
    }
    this.verbose = verbose;

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

    //writeln("this K is:", K.shape, K);

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
    nanCounter(this.Y);
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
    //prettyPrint(lmmobject.Yt);
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
    dmatrix temppp = addDmatrixNum(multiplyDmatrixNum(lmmobject.Kva,h),(1.0 - h));
    S = divideNumDmatrix(1,temppp);
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

  double LL_brent(LMM2 lmmobject, double h, dmatrix X, bool stack = true, bool REML = false){
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
        double XTX = det(matrixMult(XT, X));

        LL_REML_part = q*std.math.log(2.0*std.math.PI*sigma) + std.math.log(XTX) - std.math.log(det(XX));
        LL = LL + 0.5*LL_REML_part;
      }
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
    //for(int i=1; i< n-2; i++){
    //  if(lmmobject.LLs.elements[i-1] < lmmobject.LLs.elements[i] && lmmobject.LLs.elements[i] > lmmobject.LLs.elements[i+1]){
    //    HOpt ~= optimizeBrent(LL_brent,X,REML,[H.elements[i-1],H.elements[i+1]]);
    //    if(std.math.isNaN(HOpt[$-1])){
    //      HOpt[$-1] = H.elements[i-1];
    //    }
    //  }
    //}

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

  auto lmm2association(ref LMM2 lmmobject, ref dmatrix X, ref double h, bool stack=true, bool REML=true, bool returnBeta=false){
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
      setCol(lmmobject.X0t_stack,lmmobject.q,m);
      X = lmmobject.X0t_stack;
    }
    if(h.init == false){h = lmmobject.optH;}

    dmatrix beta, betaVAR;
    double sigma; 
    dmatrix L;
    //writeln("In lmm2association");
    
    getLL(L,beta,sigma,betaVAR, lmmobject,h, X ,false,REML);
    int q  = cast(int)beta.elements.length;
    //writeln("heritability= ", lmmobject.optH, " sigma= ", lmmobject.optSigma, " LL= ", L);

    double ts,ps;
    //writeln(betaVAR);
    return tstat(lmmobject, beta.elements[q-1], betaVAR.acc(q-1,q-1),sigma,q);

    //debug("ts=%0.3f, ps=%0.3f, heritability=%0.3f, sigma=%0.3f, LL=%0.5f" % (ts,ps,h,sigma,L))
    //if(returnBeta){return ts,ps,beta[q-1].sum(),betaVAR[q-1,q-1].sum()*sigma;}
    //return ts,ps;
  }

  auto tstat(ref LMM2 lmmobject, double beta, double var, double sigma, double q, bool log=false){

    //"""
    //   Calculates a t-statistic and associated p-value given the estimate of beta and its standard error.
    //   This is actually an F-test, but when only one hypothesis is being performed, it reduces to a t-test.
    //"""S
    double ts =  beta / std.math.sqrt(var*sigma);
    //#ps = 2.0*(1.0 - stats.t.cdf(np.abs(ts), lmmobject.N-q))
    //# sf == survival function - this is more accurate -- could also use logsf if the precision is not good enough
    double ps;
    if(log){
      //double psNum = 2.0 + (stats.t.logsf(np.abs(ts), lmmobject.N-q));
    }
    else{
      //check the sign of ts.elements[0]
      ps = 2.0*(normalCDF(-std.math.abs(ts)));
    }

    return Tuple!(double, double)(ts, ps);
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

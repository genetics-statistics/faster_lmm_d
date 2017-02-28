module faster_lmm_d.lmm2;
import dstats.distrib;
import faster_lmm_d.dmatrix;
import faster_lmm_d.optmatrix;
import faster_lmm_d.helpers;
import faster_lmm_d.kinship;
import gsl.errno;
import gsl.math;
import gsl.min;

import std.stdio;
import std.typecons;

struct LLtuple{
  double sigma, LL;
  dmatrix beta,betaVAR;

  this(double LL, dmatrix beta, double sigma, dmatrix betaVAR){
    this.LL = LL;
    this.beta = beta;
    this.sigma = sigma;
    this.betaVAR = betaVAR; 
  }
}

struct fitTuple{
  double fit_hmax, fit_sigma, fit_LL;
  dmatrix fit_beta;
  LMM2 lmmobj;

  this(LMM2 lmmobj, double fit_hmax, dmatrix fit_beta, double fit_sigma, double fit_LL){
    this.lmmobj = lmmobj;
    this.fit_hmax = fit_hmax;
    this.fit_beta = fit_beta;
    this.fit_sigma = fit_sigma;
    this.fit_LL = fit_LL;
    this.lmmobj = lmmobj;
  }
}

struct mlSol{
  double sigma;
  dmatrix beta, Q, XX_i, XX;

  this(dmatrix beta, double sigma, dmatrix Q, dmatrix XX_i, dmatrix XX){
    this.beta = beta;
    this.sigma = sigma;
    this.Q = Q;
    this.XX_i = XX_i;
    this.XX = XX;
  }
}

struct LMM2{
  int q;
  double N, optH, optSigma, optLL;
  bool init = false;
  bool verbose = false;
  dmatrix X0, Y, K, Kva, Kve;
  dmatrix Yt, X0t, X0t_stack;
  dmatrix H, optBeta, LLs;

  //The constructor takes a phenotype vector or array Y of size n. It
  //takes a kinship matrix K of size n x n.  Kva and Kve can be
  //computed as Kva,Kve = linalg.eigh(K) and cached.  If they are
  //not provided, the constructor will calculate them.  X0 is an
  //optional covariate matrix of size n x q, where there are q
  //covariates.  When this parameter is not provided, the
  //constructor will set X0 to an n x 1 matrix of all ones to
  //represent a mean effect.

  this(double[] Y, dmatrix K, dmatrix Kva, dmatrix Kve, dmatrix X0, bool verbose){
    writeln("This is Y");
    writeln(Y);

    if(X0.init == false){
      writeln("Initializing LMM2...");
      X0 = onesMatrix(cast(int)Y.length,1);
    }

    this.verbose = verbose;
    bool[] v = isnan(Y);
    bool[] x = negateBool(v);
    kvakve(K, Kve, Kva);
    this.init = true;
    this.K = K;
    this.Kva = Kva;
    this.Kve = Kve;
    this.N = K.shape[0];
    this.Y =  dmatrix([K.shape[0],1] ,Y);
    nanCounter(this.Y);
    this.X0 = X0;

   
  }

  //this(int q, double N, double optH, double optSigma, dmatrix X0, dmatrix Y, dmatrix K, dmatrix Kva, dmatrix Kve, dmatrix Yt,
  // dmatrix X0t, dmatrix X0t_stack, dmatrix H, dmatrix optLL, dmatrix optBeta, dmatrix LLs){

  this(LMM2 lmmobject, dmatrix Yt, dmatrix X0t, dmatrix X0t_stack, int q){
    this.verbose = lmmobject.verbose;
    this.init = true;
    this.K = lmmobject.K;
    this.Kve = lmmobject.Kve;
    this.Kva = lmmobject.Kva;
    this.N = lmmobject.N;
    this.Y = lmmobject.Y;
    this.Yt = Yt;
    this.X0 = X0;
    this.X0t = X0t;
    this.X0t_stack = X0t_stack;
    this.q = q;
  }

  this(LMM2 lmmobject, dmatrix LLs, dmatrix H, double hmax, double optLL, dmatrix optBeta, double optSigma){
    this.verbose = lmmobject.verbose;
    this.init = true;
    this.K = lmmobject.K;
    this.Kve = lmmobject.Kve;
    this.Kva = lmmobject.Kva;
    this.N = lmmobject.N;
    this.Y = lmmobject.Y;
    this.Yt = lmmobject.Yt;
    this.X0 = lmmobject.X0;
    this.X0t = lmmobject.X0t;
    this.X0t_stack = lmmobject.X0t_stack;
    this.q = lmmobject.q;

    this.LLs = LLs;
    this.H = H;
    this.optH = hmax;
    this.optLL = optLL;
    this.optBeta = optBeta;
    this.optSigma = optSigma;
  }
}

LMM2 lmm2transform(LMM2 lmmobject){
  //"""
  //   Computes a transformation on the phenotype vector and the covariate matrix.
  //   The transformation is obtained by left multiplying each parameter by the transpose of the
  //   eigenvector matrix of K (the kinship).
  //"""
  writeln("In lmm2transform");
  dmatrix KveT = matrixTranspose(lmmobject.Kve);
  dmatrix Yt = matrixMult(KveT, lmmobject.Y);
  dmatrix X0t = matrixMult(KveT, lmmobject.X0);
  dmatrix X0t_stack = horizontallystack(X0t, onesMatrix(cast(int)lmmobject.N,1));
  int q = X0t.shape[1];
  return LMM2(lmmobject, Yt, X0t, X0t_stack, q);
}

mlSol getMLSoln(LMM2 lmmobject, double h, dmatrix X){

  //"""
  //   Obtains the maximum-likelihood estimates for the covariate coefficients (beta),
  //   the total variance of the trait (sigma) and also passes intermediates that can
  //   be utilized in other functions. The input parameter h is a value between 0 and 1 and represents
  //   the heritability or the proportion of the total variance attributed to genetics.  The X is the
  //   covariate matrix.
  //"""

  dmatrix S = divideNumDmatrix(1,addDmatrixNum(multiplyDmatrixNum(lmmobject.Kva,h),(1.0 - h)));
  dmatrix Xt = matrixTranspose(multiplyDmatrix(X, S));
  dmatrix XX = matrixMult(Xt,X);
  dmatrix XX_i = inverse(XX);
  dmatrix beta =  matrixMult(matrixMult(XX_i,Xt),lmmobject.Yt);
  dmatrix Yt = subDmatrix(lmmobject.Yt, matrixMult(X,beta));
  dmatrix YtT = matrixTranspose(Yt);
  dmatrix YtTS = multiplyDmatrix(YtT, matrixTranspose(S));
  dmatrix Q = matrixMult(YtTS,Yt);
  double sigma = Q.elements[0] * 1.0 / (cast(double)(lmmobject.N) - cast(double)(X.shape[1]));
  return mlSol(beta, sigma, Q, XX_i, XX);
}

LMM2 LMMglob;
dmatrix Xglob;

extern(C) double LL_brent(double h, void *params){
    //#brent will not be bounded by the specified bracket.
    //# I return a large number if we encounter h < 0 to avoid errors in LL computation during the search.
  if(h < 0){return 1e6;}
  dmatrix beta, betaVAR;
  bool REML = false;  
  double sigma;
  dmatrix l;
  return -getLL(LMMglob, h, Xglob, false, REML).LL;
}

LLtuple getLL(LMM2 lmmobject, double h, dmatrix X, bool stack=true, bool REML=false){
  //"""
  //   Computes the log-likelihood for a given heritability (h).  If X==None, then the
  //   default X0t will be used.  If X is set and stack=True, then X0t will be matrix concatenated with
  //   the input X.  If stack is false, then X is used in place of X0t in the LL calculation.
  //   REML is computed by adding additional terms to the standard LL and can be computed by setting REML=True.
  //"""
  if(X.init != true){
    X = lmmobject.X0t;
  }

  double n = cast(double)lmmobject.N;
  double q = cast(double)X.shape[1];

  mlSol ml = getMLSoln(lmmobject, h, X);

  double LL  = n * std.math.log(2*std.math.PI) + sum(logDmatrix((addDMatrixNum( multiplyDmatrixNum(lmmobject.Kva,h),(1-h) ) )).elements)+
  + n + n * std.math.log((1.0/n) * ml.Q.elements[0]); //Q

  LL = -0.5 * LL;

  if(REML){
    double LL_REML_part = 0;
    dmatrix XT = matrixTranspose(X);
    double XTX = det(matrixMult(XT, X));

    LL_REML_part = q*std.math.log(2.0*std.math.PI* ml.sigma) + std.math.log(XTX) - std.math.log(det(ml.XX));
    LL = LL + 0.5*LL_REML_part;
  }
  dmatrix L = dmatrix([1,1],[LL]);
  return LLtuple(LL, ml.beta, ml.sigma, ml.XX_i);
}

double optimizeBrent(LMM2 lmmobject, dmatrix X, bool REML, double lower, double upper){
  int status;
  int iter = 0, max_iter = 100;
  const(gsl_min_fminimizer_type) *T;
  gsl_min_fminimizer *s;
  double a = lower, b = upper;
  double m = (a+b)/2, m_expected = (a+b)/2;
  gsl_function F;
  F.function_ = &LL_brent;

  Xglob = X;
  LMMglob = lmmobject;
  T = gsl_min_fminimizer_brent;
  s = gsl_min_fminimizer_alloc (T);
  gsl_min_fminimizer_set (s, &F, m, a, b);

  do
  {
    iter++;
    status = gsl_min_fminimizer_iterate (s);

    m = gsl_min_fminimizer_x_minimum (s);
    a = gsl_min_fminimizer_x_lower (s);
    b = gsl_min_fminimizer_x_upper (s);

    status = gsl_min_test_interval (a, b, 0.0001, 0.0);

    if (status == GSL_SUCCESS)
      writeln("Converged:\n");
  }
  while (status == GSL_CONTINUE && iter < max_iter);

  gsl_min_fminimizer_free (s);
  return m;
}

double getMax(LMM2 lmmobject, dmatrix L, dmatrix H, dmatrix X, bool REML=false){

  //"""
  //   Helper functions for .fit(...).
  //   This function takes a set of LLs computed over a grid and finds possible regions
  //   containing a maximum.  Within these regions, a Brent search is performed to find the
  //   optimum.

  //"""
  int n = cast(int)L.shape[0];
  double[] HOpt;
  for(int i=1; i< n-2; i++){
    if(L.elements[i-1] < L.elements[i] && L.elements[i] > L.elements[i+1]){
      HOpt ~= optimizeBrent(lmmobject, X, REML, H.elements[i-1],H.elements[i+1]);
      if(std.math.isNaN(HOpt[$-1])){
        HOpt[$-1] = H.elements[i-1];
      }
    }
  }

  if(HOpt.length > 1){
    writeln("NOTE: Found multiple optima.  Returning first...\n");
    return HOpt[0];
  }
  else if(HOpt.length == 1){
    return HOpt[0];
  }
  else if(L.elements[0] > L.elements[n-1]){
    return H.elements[0];
  }
  else{
    return H.elements[n-1];
  }
}

fitTuple lmm2fit(LMM2 lmmobject, dmatrix X, int ngrids=100, bool REML=true){

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
    X = lmmobject.X0t_stack;
  }
  double[] Harr = new double[ngrids];
  for(int m = 0; m < ngrids; m++){
    Harr[m] = m / cast(double)ngrids;
  }

  double[] elm = new double[ngrids];
  for(int h = 0; h < ngrids; h++){
    elm[h] = getLL(lmmobject, Harr[h], X, false, REML).LL;
  }
  dmatrix L = dmatrix([cast(int)elm.length,1],elm);
  dmatrix H = dmatrix([cast(int)Harr.length,1],Harr);
  double hmax = getMax(lmmobject, L, H, X, REML);
  LLtuple ll = getLL(lmmobject, hmax, X, false, REML);

  LMM2 lmmobj = LMM2(lmmobject, L, H, hmax, ll.LL, ll.beta, ll.sigma);

  return fitTuple(lmmobj, hmax, ll.beta, ll.sigma, ll.LL);
}

auto lmm2association(LMM2 lmmobject, dmatrix X, bool stack=true, bool REML=true, bool returnBeta=false){
  //"""
  //  Calculates association statitics for the SNPs encoded in the vector X of size n.
  //  If h is None, the optimal h stored in optH is used.
  //"""
  if(false){
    writeln("X=",X);
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
  double h = lmmobject.optH;
  dmatrix beta, betaVAR;
  double sigma; 
  dmatrix L;
  LLtuple ll = getLL(lmmobject,h, X ,false,REML);
  int q  = cast(int)ll.beta.elements.length;

  double ts,ps;
  return tstat(lmmobject, ll.beta.elements[q-1], ll.betaVAR.acc(q-1,q-1), ll.sigma, q);
}

auto tstat( LMM2 lmmobject, double beta, double var, double sigma, double q, bool log=false){

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
}

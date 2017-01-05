module simplelmm.lmm2;

import simplelmm.dmatrix;
import simplelmm.optmatrix;

//import std.stdio;
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
  dmatrix H;
  double N;

  dmatrix optLL;
  double optBeta;
  double optSigma;
  double LLs;

  this(double[] Y, dmatrix K, dmatrix Kva, dmatrix Kve, double X0,bool verbose){
    this.Y = Y;
    this.K = K;
    this.Kva = Kva;
    this.Kve = Kve;
    //this.X0 = null;
    this.verbose = false;
    //lmm2transform();
  }
}



  void lmm2transform(ref LMM2 lmmobject){
    //"""
    //   Computes a transformation on the phenotype vector and the covariate matrix.
    //   The transformation is obtained by left multiplying each parameter by the transpose of the
    //   eigenvector matrix of K (the kinship).
    //"""

    lmmobject.Yt = matrixMult(matrixTranspose(lmmobject.Kve), lmmobject.Y);
    lmmobject.X0t = matrixMult(matrixTranspose(lmmobject.Kve), lmmobject.X0);
    //lmmobject.X0t_stack = gethstack([lmmobject.X0t, np.ones((lmmobject.N,1))]);
    lmmobject.q = lmmobject.X0t.shape[1];
  }

  void getMLSoln(ref LMM2 lmmobject,ref double h, ref dmatrix X){

    //"""
    //   Obtains the maximum-likelihood estimates for the covariate coefficients (beta),
    //   the total variance of the trait (sigma) and also passes intermediates that can
    //   be utilized in other functions. The input parameter h is a value between 0 and 1 and represents
    //   the heritability or the proportion of the total variance attributed to genetics.  The X is the
    //   covariate matrix.
    //"""

    double S = 1.0/(h*lmmobject.Kva + (1.0 - h));
    dmatrix Xt = X.T*S;
    dmatrix XX = matrixMult(Xt,X);
    dmatrix XX_i = inv(XX);
    beta =  matrixMult(matrixMult(XX_i,Xt),lmmobject.Yt);
    Yt = lmmobject.Yt - matrixMult(X,beta);
    Q = np.dot(Yt.T*S,Yt);
    sigma = Q * 1.0 / (float(lmmobject.N) - float(X.shape[1]));
    //return beta,sigma,Q,XX_i,XX;
  }

//  void LL_brent(ref LMM2 lmmobject, ref double h, ref dmatrix X, ref bool REML){
//      //#brent will not be bounded by the specified bracket.
//      //# I return a large number if we encounter h < 0 to avoid errors in LL computation during the search.
//    if(h < 0){return 1e6;}
//    //return -lmmobject.LL(h,X,stack=False,REML=REML)[0];
//  }

  void getLL(ref LMM2 lmmobject, ref double h, ref dmatrix X, bool stack=true, bool REML=false){
      //"""
      //   Computes the log-likelihood for a given heritability (h).  If X==None, then the
      //   default X0t will be used.  If X is set and stack=True, then X0t will be matrix concatenated with
      //   the input X.  If stack is false, then X is used in place of X0t in the LL calculation.
      //   REML is computed by adding additional terms to the standard LL and can be computed by setting REML=True.
      //"""

      if(X is None){
        double X = lmmobject.X0t;
      }
      else if(stack){
        lmmobject.X0t_stack[sval,(lmmobject.q)] = matrixMult(lmmobject.Kve.T,X)[sval,0];
        X = lmmobject.X0t_stack;
      }
      double n = cast(double)lmmobject.N;
      double q = cast(double)X.shape[1];
      //beta,sigma,Q,XX_i,XX = 
      lmmobject.getMLSoln(h,X);
      double LL = 0;//n*np.log(2*np.pi) + np.log(h*lmmobject.Kva + (1.0-h)).sum() + n + n*np.log(1.0/n * Q);
      LL = -0.5 * LL;

      if(REML){
        double LL_REML_part = 0;
        //q*np.log(2.0*np.pi*sigma) + np.log(det(matrixMultT(X.T))) - np.log(det(XX));
        LL = LL + 0.5*LL_REML_part;
      }
      double LLsum = sumArray(LL);
      //# info(["beta=",beta[0][0]," sigma=",sigma[0][0]," LL=",LLsum])
      //return LLsum,beta,sigma,XX_i;
  }

  double getMax(ref LMM2 lmmobject, ref dmatrix H, ref dmatrix X, bool REML=false){

    //"""
    //   Helper functions for .fit(...).
    //   This function takes a set of LLs computed over a grid and finds possible regions
    //   containing a maximum.  Within these regions, a Brent search is performed to find the
    //   optimum.

    //"""
    n = len(lmmobject.LLs);
    auto HOpt = [];
    foreach(i; range(1,n-2)){
      if(lmmobject.LLs[i-1] < lmmobject.LLs[i] && lmmobject.LLs[i] > lmmobject.LLs[i+1]){
        HOpt.append(optimize.brent(lmmobject.LL_brent,args=(X,REML),brack=(H[i-1],H[i+1])));
        if(np.isnan(HOpt[-1])){
          HOpt[-1] = H[i-1];
        }
        //#if np.isnan(HOpt[-1]): HOpt[-1] = lmmobject.LLs[i-1]
        //#if np.isnan(HOpt[-1][0]): HOpt[-1][0] = [lmmobject.LLs[i-1]]
      }
    }

    if(len(HOpt) > 1){
      if(self.verbose){sys.stderr.write("NOTE: Found multiple optima.  Returning first...\n");}
      return HOpt[0];
    }
    else if(len(HOpt) == 1){
      return HOpt[0];
    }
    else if(self.LLs[0] > self.LLs[n-1]){
      return H[0];
    }
    else{
      return H[n-1];
    }
    return 1;
  }

  void lmm2fit(ref LMM2 lmmobject,ref dmatrix X, double ngrids=100, bool REML=true){

    //"""
    //   Finds the maximum-likelihood solution for the heritability (h) given the current parameters.
    //   X can be passed and will transformed and concatenated to X0t.  Otherwise, X0t is used as
    //   the covariate matrix.

    //   This function calculates the LLs over a grid and then uses .getMax(...) to find the optimum.
    //   Given this optimum, the function computes the LL and associated ML solutions.
    //"""

    if(X is None){ 
      X = lmmobject.X0t;
    }
    else{
       //#X = np.hstack([lmmobject.X0t,matrixMult(lmmobject.Kve.T, X)])
      lmmobject.X0t_stack[sval,(lmmobject.q)] = matrixMult(lmmobject.Kve.T,X)[sval,0];
      X = lmmobject.X0t_stack;
    }

    dmatrix H;
    //= np.array(range(ngrids)) / float(ngrids);
    dmatrix L;
    //  L= np.array([lmmobject.LL(h,X,stack=False,REML=REML)[0] for h in H]);
    lmmobject.LLs = L;

    double hmax = lmmobject.getMax(H,X,REML);
    //L,beta,sigma,betaSTDERR = 
    double beta;
    double[] sigma;
    getLL(lmmobject,hmax,X,false,REML);

    lmmobject.H = H;
    //false.optH = hmax.sum();
    lmmobject.optLL = L;
    lmmobject.optBeta = beta;
    lmmobject.optSigma = sumArray(sigma);

    //# debug(["hmax",hmax,"beta",beta,"sigma",sigma,"LL",L])
    //return hmax,beta,sigma,L;
  }

//  void association(ref LMM2 lmmobject, ref dmatrix X, ref dmatrix h, bool stack=true, bool REML=true, bool returnBeta=false){
//    //"""
//    //  Calculates association statitics for the SNPs encoded in the vector X of size n.
//    //  If h is None, the optimal h stored in optH is used.

//    //"""
//    if(false){
//      writeln("X=",X);
//      writeln("h=",h);
//      writeln("q=",lmmobject.q);
//      writeln("lmmobject.Kve=",lmmobject.Kve);
//      writeln("X0t_stack=",lmmobject.X0t_stack.shape,lmmobject.X0t_stack);
//    }

//    if(stack){
//      //# X = np.hstack([lmmobject.X0t,matrixMult(lmmobject.Kve.T, X)])
//      m = matrixMult(lmmobject.Kve.T,X);
//      //# writeln( "m=",m);
//      m = m[sval,0];
//      lmmobject.X0t_stack[sval,(lmmobject.q)] = m;
//      X = lmmobject.X0t_stack;
//    }
//    if(h is None){h = lmmobject.optH;}

//    L,beta,sigma,betaVAR = lmmobject.LL(h,X,stack=False,REML=REML);
//    q  = len(beta);
//    ts,ps = lmmobject.tstat(beta[q-1],betaVAR[q-1,q-1],sigma,q);

//    //debug("ts=%0.3f, ps=%0.3f, heritability=%0.3f, sigma=%0.3f, LL=%0.5f" % (ts,ps,h,sigma,L))
//    if(returnBeta){return ts,ps,beta[q-1].sum(),betaVAR[q-1,q-1].sum()*sigma;}
//    //return ts,ps;
//  }

//  void tstat(ref LMM2 lmmobject, double beta, double var, double sigma, double q, bool log=false){

//    //"""
//    //   Calculates a t-statistic and associated p-value given the estimate of beta and its standard error.
//    //   This is actually an F-test, but when only one hypothesis is being performed, it reduces to a t-test.
//    //"""

//    ts = beta / np.sqrt(var * sigma);
//    //#ps = 2.0*(1.0 - stats.t.cdf(np.abs(ts), lmmobject.N-q))
//    //# sf == survival function - this is more accurate -- could also use logsf if the precision is not good enough
//    if(log){
//      ps = 2.0 + (stats.t.logsf(np.abs(ts), lmmobject.N-q));
//    }
//    else{
//      ps = 2.0*(stats.t.sf(np.abs(ts), lmmobject.N-q));
//    }
//    if(!(len(ts) == 1) || !(len(ps) == 1)){
//      //raise Exception("Something bad happened :(");
//    }
//    return ts.sum(),ps.sum();
//  }

////  void plotFit(lmmobject,color="b-",title=""){

////    //"""
////    //   Simple function to visualize the likelihood space.  It takes the LLs
////    //   calcualted over a grid and normalizes them by subtracting off the mean and exponentiating.
////    //   The resulting "probabilities" are normalized to one and plotted against heritability.
////    //   This can be seen as an approximation to the posterior distribuiton of heritability.

////    //   For diagnostic purposes this lets you see if there is one distinct maximum or multiple
////    //   and what the variance of the parameter looks like.
////    //"""

////    mx = lmmobject.LLs.max();
////    p = np.exp(lmmobject.LLs - mx);
////    p = p/p.sum();

////    pl.plot(lmmobject.H,p,color);
////    pl.xlabel("Heritability");
////    pl.ylabel("Probability of data");
////    pl.title(title);
////  }

//  void meanAndVar(ref LMM2 lmmobject){
//    mx = lmmobject.LLs.max();
//    p = np.exp(lmmobject.LLs - mx);
//    p = p/p.sum();

//    mn = (lmmobject.H * p).sum();
//    vx = ((lmmobject.H - mn)**2 * p).sum();

//    return mn,vx;
//  }
////}

/*
   This code is part of faster_lmm_d and published under the GPLv3
   License (see LICENSE.txt)

   Copyright © 2017 Prasun Anand & Pjotr Prins
*/

module faster_lmm_d.lmm2;

import std.conv;
import std.experimental.logger;
import std.math;
alias mlog = std.math.log;
import std.typecons;

import dstats.distrib;
import gsl.errno;
import gsl.math;
import gsl.min;

import faster_lmm_d.cuda;
import faster_lmm_d.dmatrix;
import faster_lmm_d.helpers;
import faster_lmm_d.kinship;
import faster_lmm_d.memory;
import faster_lmm_d.optmatrix;

import core.stdc.stdlib : exit;

alias Tuple!(immutable double, "LL", const DMatrix, "beta", immutable double, "sigma", const DMatrix, "beta_var") LLTuple;
alias Tuple!(const DMatrix, "beta", immutable double, "sigma", const DMatrix, "Q", const DMatrix, "XX_i", const DMatrix, "XX") MLSol;

struct LMM{
  ulong q, N;
  double opt_H, opt_sigma, opt_LL;
  bool init = false;
  bool verbose = false;
  DMatrix X0, Y, Kva, Kve, KveT;
  DMatrix K;
  DMatrix Yt, X0t, X0t_stack;
  DMatrix H, opt_beta, LLs;

  //The constructor takes a phenotype vector or array Y of size n. It
  //takes a kinship matrix K of size n x n.  Kva and Kve can be
  //computed as Kva,Kve = linalg.eigh(K) and cached.  If they are
  //not provided, the constructor will calculate them.  X0 is an
  //optional covariate matrix of size n x q, where there are q
  //covariates.  When this parameter is not provided, the
  //constructor will set X0 to an n x 1 matrix of all ones to
  //represent a mean effect.

  this(const double[] Y, const DMatrix K, const DMatrix Kva, const DMatrix Kve, const DMatrix X0, bool verbose){
    trace("Y => ",Y[0..3],"...",Y[$-3..$]);

    auto X0_new = (!X0.init ? ones_dmatrix(Y.length,1) : DMatrix(X0) );

    this.verbose = verbose;
    bool[] v = is_nan(Y);
    bool[] x = negate_bool(v);
    EighTuple keigh = kvakve(K);
    this.init = true;
    this.K = DMatrix(K);
    this.Kva = keigh.kva;
    this.Kve = keigh.kve;
    this.N = K.shape[0];
    this.Y = DMatrix([K.shape[0],1] ,Y);
    this.X0 = X0_new;
  }

  this(const LMM lmmobject, const DMatrix Yt, const DMatrix X0t, const DMatrix X0t_stack, const DMatrix KveT, ulong q){
    this.verbose = lmmobject.verbose;
    this.init = true;
    this.K = DMatrix(lmmobject.K);
    this.Kve = DMatrix(lmmobject.Kve);
    this.Kva = DMatrix(lmmobject.Kva);
    this.N = lmmobject.N;
    this.Y = DMatrix(lmmobject.Y);
    this.Yt = DMatrix(Yt);
    this.X0 = X0;
    this.X0t = DMatrix(X0t);
    this.X0t_stack = DMatrix(X0t_stack);
    this.KveT = DMatrix(KveT);
    this.q = q;
  }

  this(const LMM lmmobject, const DMatrix LLs, const DMatrix H, immutable double hmax, immutable double opt_LL, const DMatrix opt_beta, immutable double opt_sigma){
    this.verbose = lmmobject.verbose;
    this.init = true;
    this.K = DMatrix(lmmobject.K);
    this.Kve = DMatrix(lmmobject.Kve);
    this.Kva = DMatrix(lmmobject.Kva);
    this.N = lmmobject.N;
    this.Y = DMatrix(lmmobject.Y);
    this.Yt = DMatrix(lmmobject.Yt);
    this.X0 = DMatrix(lmmobject.X0);
    this.X0t = DMatrix(lmmobject.X0t);
    this.X0t_stack = DMatrix(lmmobject.X0t_stack);
    this.KveT = DMatrix(lmmobject.KveT);
    this.q = lmmobject.q;

    this.LLs =  DMatrix(LLs);
    this.H = DMatrix(H);
    this.opt_H = hmax;
    this.opt_LL = opt_LL;
    this.opt_beta = DMatrix(opt_beta);
    this.opt_sigma = opt_sigma;
  }

  this(const LMM lmmobject){
    this.verbose = lmmobject.verbose;
    this.init = true;
    this.K = DMatrix(lmmobject.K);
    this.Kve = DMatrix(lmmobject.Kve);
    this.Kva = DMatrix(lmmobject.Kva);
    this.N = lmmobject.N;
    this.Y = DMatrix(lmmobject.Y);
    this.Yt = DMatrix(lmmobject.Yt);
    this.X0 = DMatrix(lmmobject.X0);
    this.X0t = DMatrix(lmmobject.X0t);
    this.X0t_stack = DMatrix(lmmobject.X0t_stack);
    this.KveT = DMatrix(lmmobject.KveT);
    this.q = lmmobject.q;

    this.LLs = DMatrix(lmmobject.LLs);
    this.H = DMatrix(lmmobject.H);
    this.opt_LL = lmmobject.opt_LL;
    this.opt_beta = DMatrix(lmmobject.opt_beta);
    this.opt_sigma = lmmobject.opt_sigma;
  }
}

LMM lmm_transform(const LMM lmmobject){

  //   Computes a transformation on the phenotype vector and the covariate matrix.
  //   The transformation is obtained by left multiplying each parameter by the transpose of the
  //   eigenvector matrix of K (the kinship).

  trace("In lmm_transform");
  DMatrix KveT = matrix_transpose(lmmobject.Kve);
  DMatrix Yt = matrix_mult(KveT, lmmobject.Y);
  DMatrix X0t = matrix_mult(KveT, lmmobject.X0);
  DMatrix X0t_stack = horizontally_stack(X0t, ones_dmatrix(lmmobject.N,1));
  auto q = X0t.shape[1];
  return LMM(lmmobject, Yt, X0t, X0t_stack, KveT, q);
}

MLSol getMLSoln(const LMM lmmobject, const double h, const DMatrix X){

  //   Obtains the maximum-likelihood estimates for the covariate coefficients (beta),
  //   the total variance of the trait (sigma) and also passes intermediates that can
  //   be utilized in other functions. The input parameter h is a value between 0 and 1 and represents
  //   the heritability or the proportion of the total variance attributed to genetics.  The X is the
  //   covariate matrix.

  DMatrix S = divide_num_dmatrix(1,add_dmatrix_num(multiply_dmatrix_num(lmmobject.Kva,h),(1.0 - h)));
  auto temp = S.shape.dup;
  S.shape = [temp[1], temp[0]];
  DMatrix Xt = multiply_dmatrix(matrix_transpose(X), S);
  DMatrix XX = matrix_mult(Xt,X);
  DMatrix XX_i = inverse(XX);
  DMatrix beta =  matrix_mult(matrix_mult(XX_i,Xt),lmmobject.Yt);
  DMatrix Yt = sub_dmatrix(lmmobject.Yt, matrix_mult(X,beta));
  DMatrix YtT = matrix_transpose(Yt);
  DMatrix YtTS = multiply_dmatrix(YtT, S);
  DMatrix Q = matrix_mult(YtTS,Yt);
  double sigma = Q.elements[0] * 1.0 / (to!double(lmmobject.N) - to!double(X.shape[1]));
  return MLSol(beta, sigma, Q, XX_i, XX);
}

LMM LMMglob;
DMatrix Xglob;

extern(C) double LL_brent(double h, void *params){

  // brent will not be bounded by the specified bracket.
  // I return a large number if we encounter h < 0 to avoid errors in LL computation during the search.

  if( h < 0){ return 1e6; }
  return -get_LL(LMMglob, h, Xglob, false, true).LL;
}

LLTuple get_LL(const LMM lmmobject, const double h, const DMatrix param_X, const bool stack=true, const bool REML=false){

  //   Computes the log-likelihood for a given heritability (h).  If X==None, then the
  //   default X0t will be used.  If X is set and stack=True, then X0t will be matrix concatenated with
  //   the input X.  If stack is false, then X is used in place of X0t in the LL calculation.
  //   REML is computed by adding additional terms to the standard LL and can be computed by setting REML=True.
  const DMatrix X = ( param_X.init != true ? lmmobject.X0t : param_X );

  double n = to!double(lmmobject.N);
  double q = to!double(X.shape[1]);

  MLSol ml = getMLSoln(lmmobject, h, X);

  double LL  = n * mlog(2*PI) + sum(log_dmatrix((add_dmatrix_num( multiply_dmatrix_num(lmmobject.Kva,h),(1-h) ) )).elements)+
  + n + n * mlog((1.0/n) * ml.Q.elements[0]); //Q

  LL = -0.5 * LL;

  if(REML){
    double LL_REML_part = 0;
    DMatrix XT = matrix_transpose(X);
    LL_REML_part = q*mlog(2.0*PI* ml.sigma) + mlog(det(matrix_mult(XT, X))) - mlog(det(ml.XX));
    LL = LL + 0.5*LL_REML_part;
  }

  return LLTuple(LL, ml.beta, ml.sigma, ml.XX_i);
}

double optimize_brent(const LMM lmmobject, const DMatrix X, const bool REML, const double lower, const double upper){
  int status;
  ulong iter = 0, max_iter = 100;
  const(gsl_min_fminimizer_type) *T;
  gsl_min_fminimizer *s;
  double a = lower, b = upper;
  double m = (a+b)/2;
  gsl_function F;
  F.function_ = &LL_brent;

  Xglob = DMatrix(X);
  LMMglob = LMM(lmmobject);
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
      trace("Converged:");
  }
  while (status == GSL_CONTINUE && iter < max_iter);

  gsl_min_fminimizer_free (s);
  return m;
}

double get_max(const LMM lmmobject, const DMatrix L, const DMatrix H, const DMatrix X, const bool REML=false){

  //   Helper functions for .fit(...).
  //   This function takes a set of LLs computed over a grid and finds possible regions
  //   containing a maximum.  Within these regions, a Brent search is performed to find the
  //   optimum.

  auto n = L.shape[0];
  double[] HOpt;
  for(auto i=1; i< n-2; i++){
    if(L.elements[i-1] < L.elements[i] && L.elements[i] > L.elements[i+1]){
      HOpt ~= optimize_brent(lmmobject, X, REML, H.elements[i-1],H.elements[i+1]);
      if(isNaN(HOpt[$-1])){
        HOpt[$-1] = H.elements[i-1];
      }
    }
  }

  if(HOpt.length > 1){
    trace("NOTE: Found multiple optima.  Returning first...\n");
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

LMM lmm_fit(const LMM lmmobject, const DMatrix X_param, const ulong ngrids=100, const bool REML=true){

  //   Finds the maximum-likelihood solution for the heritability (h) given the current parameters.
  //   X can be passed and will transformed and concatenated to X0t.  Otherwise, X0t is used as
  //   the covariate matrix.

  //   This function calculates the LLs over a grid and then uses .get_max(...) to find the optimum.
  //   Given this optimum, the function computes the LL and associated ML solutions.
  DMatrix X;

  if(X_param.init == false){
    X = DMatrix(lmmobject.X0t);
  }
  else{
    DMatrix KveTX = matrix_mult(lmmobject.KveT , X_param);
    X = DMatrix(lmmobject.X0t_stack);
  }
  double[] Harr = new double[ngrids];
  for(auto m = 0; m < ngrids; m++){
    Harr[m] = m / to!double(ngrids);
  }

  double[] elm = new double[ngrids];
  for(auto h = 0; h < ngrids; h++){
    elm[h] = get_LL(lmmobject, Harr[h], X, false, REML).LL;
    check_memory();
  }
  DMatrix L = DMatrix([elm.length,1],elm);
  DMatrix H = DMatrix([Harr.length,1],Harr);
  double fit_hmax = get_max(lmmobject, L, H, X, REML);
  LLTuple ll = get_LL(lmmobject, fit_hmax, X, false, REML);

  return LMM(lmmobject, L, H, fit_hmax, ll.LL, ll.beta, ll.sigma);
}

auto lmm_association(const LMM lmmobject, const DMatrix param_X, const bool stack=true, const bool REML=true, const bool return_beta=false){

  //  Calculates association statitics for the SNPs encoded in the vector X of size n.
  //  If h is None, the optimal h stored in opt_H is used.

  if(false){
    trace("X=",param_X);
    trace("q=",lmmobject.q);
    trace("lmmobject.Kve=",lmmobject.Kve);
    trace("X0t_stack=",lmmobject.X0t_stack);
  }

  DMatrix X;
  if(stack){
    DMatrix m = matrix_mult(lmmobject.KveT, param_X);
    X = set_col(lmmobject.X0t_stack,lmmobject.q,m);
  }

  LLTuple ll = get_LL(lmmobject, lmmobject.opt_H, X ,false, REML);
  auto q = ll.beta.elements.length;
  const ulong df = lmmobject.N - q;
  return tstat(ll.beta.elements[q-1], accessor(ll.beta_var,q-1,q-1), ll.sigma, q, df);
}

auto tstat(const double beta, const double var, const double sigma, const double q, const ulong df){

  //   Calculates a t-statistic and associated p-value given the estimate of beta and its standard error.
  //   This is actually an F-test, but when only one hypothesis is being performed, it reduces to a t-test.
  double ts = beta / sqrt(var*sigma);
  double ps = 2.0*( 1 -  studentsTCDF(abs(ts), df));
  double lod = chiSquareCDF(ps, 1);

  return Tuple!(double, double, double)(ts, ps, lod);
}

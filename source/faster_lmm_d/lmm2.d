/*
   This code is part of faster_lmm_d and published under the GPLv3
   License (see LICENSE.txt)

   Copyright © 2017 Prasun Anand & Pjotr Prins
*/

module faster_lmm_d.lmm2;

import std.conv;
import std.exception;
import std.math;
alias mlog = std.math.log;
import std.typecons;
import std.experimental.logger;

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

alias N_Individuals = immutable uint;
alias N_Covariates = immutable uint;
alias Tuple!(double,"ts",double,"p_value",double,"lod") TStat;

struct LMM {
  immutable double opt_H, opt_sigma, opt_LL;
  DMatrix X0, Kva;
  DMatrix Yt;
  DMatrix X0t, X0t_stack;
  DMatrix opt_beta;

  //The constructor takes a phenotype vector or array Y of size n. It
  //takes a kinship matrix K of size n x n.  Kva and Kve can be
  //computed as Kva,Kve = linalg.eigh(K) and cached.  X0 is an
  //optional covariate matrix of size n x q, where there are q
  //covariates.  When this parameter is not provided, the
  //constructor will set X0 to an n x 1 matrix of all ones to
  //represent a mean effect.

  this(const double[] Y, const DMatrix Kva, const DMatrix X0) {
    this.Kva = DMatrix(Kva);
    auto X0_new = (!X0.shape ? ones_dmatrix(Y.length,1) : DMatrix(X0) );
    bool[] v = is_nan(Y);
    bool[] x = negate_bool(v);
    this.X0 = X0_new;
  }

  this(const LMM lmmobject, const DMatrix Yt, const DMatrix X0t,
       const DMatrix X0t_stack) {
    this(lmmobject);
    this.Yt = DMatrix(Yt);
    this.X0 = X0;
    this.X0t = DMatrix(X0t);
    this.X0t_stack = DMatrix(X0t_stack);
  }

  this(const LMM lmmobject,
       immutable double hmax, immutable double opt_LL,
       const DMatrix opt_beta, immutable double opt_sigma) {
    this(lmmobject);
    this.opt_H = hmax;
    this.opt_LL = opt_LL;
    this.opt_beta = DMatrix(opt_beta);
    this.opt_sigma = opt_sigma;
  }

  this(const LMM lmmobject) {
    // this.Kve = DMatrix(lmmobject.Kve);
    this.Kva = DMatrix(lmmobject.Kva);
    this.Yt = DMatrix(lmmobject.Yt);
    this.X0 = DMatrix(lmmobject.X0);
    this.X0t = DMatrix(lmmobject.X0t);
    this.X0t_stack = DMatrix(lmmobject.X0t_stack);

    this.opt_LL = lmmobject.opt_LL;
    this.opt_beta = DMatrix(lmmobject.opt_beta);
    this.opt_sigma = lmmobject.opt_sigma;
  }
}

LMM lmm_transform(const LMM lmmobject, N_Individuals N, const double[] Y, const DMatrix Kve) {
  //   Computes a transformation on the phenotype vector and the
  //   covariate matrix.  The transformation is obtained by left
  //   multiplying each parameter by the transpose of the eigenvector
  //   matrix of K (the kinship).

  DMatrix KveT = slow_matrix_transpose(Kve);
  DMatrix Yt = matrix_mult(KveT, DMatrix(Y));
  DMatrix X0t = matrix_mult(KveT, lmmobject.X0);
  DMatrix X0t_stack = horizontally_stack(X0t, ones_dmatrix(N,1));
  auto q = X0t.shape[1];
  return LMM(lmmobject, Yt, X0t, X0t_stack);
}

MLSol getMLSoln(const double h, const DMatrix X, const DMatrix _Yt, const DMatrix Kva, N_Individuals N) {

  //   Obtains the maximum-likelihood estimates for the covariate
  //   coefficients (beta), the total variance of the trait (sigma)
  //   and also passes intermediates that can be utilized in other
  //   functions. The input parameter h is a value between 0 and 1 and
  //   represents the heritability or the proportion of the total
  //   variance attributed to genetics.  The X is the covariate
  //   matrix.

  DMatrix S = divide_num_dmatrix(1,add_dmatrix_num(multiply_dmatrix_num(Kva,h),(1.0 - h)));
  auto temp = S.shape.dup_fast;
  S.shape = [temp[1], temp[0]];
  DMatrix Xt = slow_multiply_dmatrix(slow_matrix_transpose(X), S);
  DMatrix XX = matrix_mult(Xt,X);
  DMatrix XX_i = inverse(XX);
  DMatrix beta =  matrix_mult(matrix_mult(XX_i,Xt),_Yt);
  DMatrix Yt = subtract_dmatrix(_Yt, matrix_mult(X,beta));
  DMatrix YtT = slow_matrix_transpose(Yt);
  DMatrix YtTS = slow_multiply_dmatrix(YtT, S);
  DMatrix Q = matrix_mult(YtTS,Yt);
  double sigma = Q.elements[0] * 1.0 / (to!double(N) - to!double(X.shape[1]));
  return MLSol(beta, sigma, Q, XX_i, XX);
}

LLTuple get_LL(const double h, const DMatrix param_X,
               N_Individuals N, const DMatrix Kva, const DMatrix Yt, const DMatrix X0t,
               const bool stack=true, const bool REML=false) {

  //   Computes the log-likelihood for a given heritability (h).  If
  //   X==None, then the default X0t will be used.  If X is set and
  //   stack=True, then X0t will be matrix concatenated with the input
  //   X.  If stack is false, then X is used in place of X0t in the LL
  //   calculation.  REML is computed by adding additional terms to
  //   the standard LL and can be computed by setting REML=True.
  const DMatrix X = ( !param_X.shape ? X0t : param_X );

  double n = to!double(N);
  double q = to!double(X.shape[1]);

  MLSol ml = getMLSoln(h, X, Yt, Kva, N);

  double LL  = n * mlog(2*PI) + sum(log_dmatrix((add_dmatrix_num( multiply_dmatrix_num(Kva,h),(1-h) ) )).elements)+
  + n + n * mlog((1.0/n) * ml.Q.elements[0]); //Q

  LL = -0.5 * LL;

  if(REML) {
    double LL_REML_part = 0;
    DMatrix XT = slow_matrix_transpose(X);
    LL_REML_part = q*mlog(2.0*PI* ml.sigma) + mlog(det(matrix_mult(XT, X))) - mlog(det(ml.XX));
    LL = LL + 0.5*LL_REML_part;
  }

  return LLTuple(LL, ml.beta, ml.sigma, ml.XX_i);
}

alias LL_brent_params = Tuple!(LMM,DMatrix);

/*
 * This function is passed into the GSL resolver
 */

extern(C) double LL_brent(double h, void *params) {

  // brent will not be bounded by the specified bracket.  I return a
  // large number if we encounter h < 0 to avoid errors in LL
  // computation during the search.

  if( h < 0) { return 1e6; }
  auto ptr = cast(LL_brent_params *)params;
  auto tup = *ptr;
  auto LMMglob = tup[0];
  auto Xglob = tup[1];
  auto N = cast(N_Individuals)Xglob.shape[0];
  return -get_LL(h, Xglob, N, LMMglob.Kva, LMMglob.Yt, LMMglob.X0t, false, true).LL;
}

double optimize_brent(const LMM lmmobject, const DMatrix X, const bool REML,
                      const double lower, const double upper) {
  int status;
  ulong iter = 0, max_iter = 100;
  const(gsl_min_fminimizer_type) *T;
  gsl_min_fminimizer *s;
  double a = lower, b = upper;
  double m = (a+b)/2;
  gsl_function F;
  F.function_ = &LL_brent;
  auto LMMglob = LMM(lmmobject);
  auto Xglob = DMatrix(X);
  auto params = LL_brent_params(LMMglob,Xglob);
  F.params = cast(void *)&params;

  T = gsl_min_fminimizer_brent;
  s = gsl_min_fminimizer_alloc (T);
  enforce(s);
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

double get_max(const LMM lmmobject, const DMatrix L, const DMatrix H,
               const DMatrix X, const bool REML=false) {

  //   Helper functions for .fit(...).  This function takes a set of
  //   LLs computed over a grid and finds possible regions containing
  //   a maximum.  Within these regions, a Brent search is performed
  //   to find the optimum.

  auto n = L.shape[0];
  double[] HOpt;
  for(auto i=1; i< n-2; i++) {
    if(L.elements[i-1] < L.elements[i] && L.elements[i] > L.elements[i+1]) {
      HOpt ~= optimize_brent(lmmobject, X, REML, H.elements[i-1],H.elements[i+1]);
      if(isNaN(HOpt[$-1])) {
        HOpt[$-1] = H.elements[i-1];
      }
    }
  }

  if(HOpt.length > 1) {
    trace("NOTE: Found multiple optima.  Returning first...\n");
    return HOpt[0];
  }
  else if(HOpt.length == 1) {
    return HOpt[0];
  }
  else if(L.elements[0] > L.elements[n-1]) {
    return H.elements[0];
  }
  else{
    return H.elements[n-1];
  }
}

LMM lmm_fit(const LMM lmmobject, N_Individuals N, const DMatrix X_param, const ulong ngrids=100,
            const bool REML=true) {

  //   Finds the maximum-likelihood solution for the heritability (h)
  //   given the current parameters.  X can be passed and will
  //   transformed and concatenated to X0t.  Otherwise, X0t is used as
  //   the covariate matrix.

  //   This function calculates the LLs over a grid and then uses
  //   .get_max(...) to find the optimum.  Given this optimum, the
  //   function computes the LL and associated ML solutions.

  DMatrix X = (!X_param.shape ? DMatrix(lmmobject.X0t) : DMatrix(lmmobject.X0t_stack));
  double[] Harr = new double[ngrids];
  for(auto m = 0; m < ngrids; m++) {
    Harr[m] = m / to!double(ngrids);
  }

  double[] elm = new double[ngrids];
  for(auto h = 0; h < ngrids; h++) {
    elm[h] = get_LL(Harr[h], X, N, lmmobject.Kva, lmmobject.Yt, lmmobject.X0t, false, REML).LL;
  }
  DMatrix L = DMatrix([elm.length,1],elm);
  DMatrix H = DMatrix([Harr.length,1],Harr);
  double fit_hmax = get_max(lmmobject, L, H, X, REML);
  LLTuple ll = get_LL(fit_hmax, X, N, lmmobject.Kva, lmmobject.Yt, lmmobject.X0t, false, REML);

  return LMM(lmmobject, fit_hmax, ll.LL, ll.beta, ll.sigma);
}

auto lmm_association(m_items i, const LMM lmmobject, N_Individuals N, const DMatrix G, const DMatrix KveT) {
  auto stack=true;
  auto REML=true;
  DMatrix _X = get_row(G, i);
  _X.shape = [N, 1];

  //  Calculates association for the SNPs encoded in the vector X of size n.
  //  If h is None, the optimal h stored in opt_H is used.
  DMatrix m = matrix_mult(KveT, _X);
  N_Covariates n_covariates = 1;
  DMatrix X = set_col(lmmobject.X0t_stack,n_covariates,m);
  LLTuple ll = get_LL(lmmobject.opt_H, X, N, lmmobject.Kva, lmmobject.Yt, lmmobject.X0t, false, REML);
  auto q = ll.beta.elements.length;
  const ulong df = N - q;
  return tstat(ll.beta.elements[q-1], accessor(ll.beta_var,q-1,q-1), ll.sigma, q, df);
}

TStat tstat(const double beta, const double var, const double sigma,
           const double q, const ulong df) {
  //   Calculates a t-statistic and associated p-value given the
  //   estimate of beta and its standard error.  This is actually an
  //   F-test, but when only one hypothesis is being performed, it
  //   reduces to a t-test.
  double ts = beta / sqrt(var*sigma);
  double ps = 2.0*( 1 -  studentsTCDF(abs(ts), df));
  double lod = chiSquareCDF(ps, 1);

  return TStat(ts, ps, lod);
}

module faster_lmm_d.logistic;

import std.bitmanip;
import std.conv;
import std.exception;
import std.file;
import std.math;
import std.parallelism;
alias mlog = std.math.log;
import std.process;
import std.range;
import std.stdio;
import std.typecons;
import std.experimental.logger;
import std.string;

extern(C){

  struct gsl_sf_result
  {
    double val;
    double err;
  };

  double gsl_sf_log_1plusx (double x);
  double gsl_sf_exp (double x);
  int gsl_sf_exp_e (double x, gsl_sf_result* result);
}

import faster_lmm_d.dmatrix;

// I need to bundle all the data that goes to the function to optimze
// together.

struct fix_parm_mixed_T {
  DMatrix_int X;
  DMatrix_int nlev;
  DMatrix y;
  DMatrix Xc; // Continuous covariates matrix Nobs x Kc (NULL if not used).
  double lambdaL1;
  double lambdaL2;
};

double fLogit_mixed(DMatrix beta, DMatrix_int X, DMatrix_int nlev,
                    DMatrix Xc, DMatrix y, double lambdaL1, double lambdaL2) {
  size_t n = y.size;
  size_t npar = beta.size;
  double total = 0;
  double aux = 0;

  // Changed loop start at 1 instead of 0 to avoid regularization of
  // beta_0*\/
  // #pragma omp parallel for reduction (+:total)
  for (int i = 1; i < npar; ++i)
    total += beta.elements[i] * beta.elements[i];
  total = (-total * lambdaL2 / 2);
  // #pragma omp parallel for reduction (+:aux)
  for (int i = 1; i < npar; ++i)
    aux += (beta.elements[i] > 0) ? beta.elements[i] : -beta.elements[i];
  total = total - aux * lambdaL1;
  // #pragma omp parallel for schedule(static) shared(n,beta,X,nlev,y)
  // #reduction (+:total)
  for (size_t i = 0; i < n; ++i) {
    double Xbetai = beta.elements[0];
    int iParm = 1;
    for (size_t k = 0; k < X.shape[1]; ++k) {
      if (X.accessor(i, k) > 0)
        Xbetai += beta.elements[X.accessor(i, k) - 1 + iParm];
      iParm += nlev.elements[k] - 1;
    }
    for (size_t k = 0; k < (Xc.shape[1]); ++k)
      Xbetai += Xc.accessor(i, k) * beta.elements[iParm++];
    total += y.elements[i] * Xbetai - gsl_sf_log_1plusx(gsl_sf_exp(Xbetai));
  }
  return -total;
}

void logistic_mixed_pred(DMatrix beta,     // Vector of parameters
                                               // length = 1 + Sum_k(C_k -1)
                         DMatrix_int X,    // Matrix Nobs x K
                         DMatrix_int nlev, // Vector with number categories
                         DMatrix Xc,       // Continuous covariates matrix:
                                               // obs x Kc (NULL if not used).
                         DMatrix yhat) {   // Vector of prob. predicted by
                                               // the logistic
  for (size_t i = 0; i < X.shape[0]; ++i) {
    double Xbetai = beta.elements[0];
    int iParm = 1;
    for (size_t k = 0; k < X.shape[1]; ++k) {
      if (X.accessor(i, k) > 0)
        Xbetai += beta.elements[X.accessor(i, k) - 1 + iParm];
      iParm += nlev.elements[k] - 1;
    }
    // Adding the continuous.
    for (size_t k = 0; k < (Xc.shape[1]); ++k)
      Xbetai += Xc.accessor(i, k) * beta.elements[iParm++];
    yhat.elements[i] = 1 / (1 + gsl_sf_exp(-Xbetai));
  }
}

// The gradient of f, df = (df/dx, df/dy).
void wgsl_mixed_optim_df(const DMatrix beta, void *params,
                         DMatrix output) {
  fix_parm_mixed_T *p = cast(fix_parm_mixed_T *)params;
  size_t n = p.y.size;
  size_t K = p.X.shape[1];
  size_t Kc = p.Xc.shape[1];
  size_t npar = beta.size;

  // Intitialize gradient output necessary?
  for (size_t i = 0; i < npar; ++i)
    output.elements[i] = 0;

  // Changed loop start at 1 instead of 0 to avoid regularization of beta 0.
  for (size_t i = 1; i < npar; ++i)
    output.elements[i] = p.lambdaL2 * beta.elements[i];
  for (size_t i = 1; i < npar; ++i)
    output.elements[i] += p.lambdaL1 * ((beta.elements[i] > 0) - (beta.elements[i] < 0));

  for (size_t i = 0; i < n; ++i) {
    double pn = 0;
    double Xbetai = beta.elements[0];
    size_t iParm = 1;
    for (size_t k = 0; k < K; ++k) {
      if (p.X.accessor(i, k) > 0)
        Xbetai += beta.elements[p.X.accessor(i, k) - 1 + iParm];
      iParm += p.nlev.elements[k] - 1;
    }

    // Adding the continuous.
    for (size_t k = 0; k < Kc; ++k)
      Xbetai += p.Xc.accessor(i, k) * beta.elements[iParm++];

    pn = -(p.y.elements[i] - 1 / (1 + gsl_sf_exp(-Xbetai)));

    output.elements[0] += pn;
    iParm = 1;
    for (size_t k = 0; k < K; ++k) {
      if (p.X.accessor(i, k) > 0)
        output.elements[p.X.accessor(i, k) - 1 + iParm] += pn;
      iParm += p.nlev.elements[k] - 1;
    }

    // Adding the continuous.
    for (size_t k = 0; k < Kc; ++k) {
      output.elements[iParm++] += p.Xc.accessor(i, k) * pn;
    }
  }
}

// The Hessian of f.
void wgsl_mixed_optim_hessian(const DMatrix beta, void *params, DMatrix output) {
  fix_parm_mixed_T *p = cast(fix_parm_mixed_T *)params;
  size_t n = p.y.size;
  size_t K = p.X.shape[1];
  size_t Kc = p.Xc.shape[1];
  size_t npar = beta.size;
  DMatrix gn = zeros_dmatrix(1, npar); // gn

  // Intitialize Hessian output necessary ???
  output = zeros_dmatrix(output.shape[0], output.shape[1]);

  /* Changed loop start at 1 instead of 0 to avoid regularization of beta 0*/
  for (int i = 1; i < npar; ++i)
    output.set(i, i, (p.lambdaL2)); // Double-check this.

  // L1 penalty not working yet, as not differentiable, I may need to
  // do coordinate descent (as in glm_net)
  for (size_t i = 0; i < n; ++i) {
    double pn = 0;
    double aux = 0;
    double Xbetai = beta.elements[0];
    size_t iParm1 = 1;
    for (size_t k = 0; k < K; ++k) {
      if (p.X.accessor(i, k) > 0)
        Xbetai += beta.elements[p.X.accessor(i, k) - 1 + iParm1];
      iParm1 += p.nlev.elements[k] - 1; //-1?
    }

    // Adding the continuous.
    for (size_t k = 0; k < Kc; ++k)
      Xbetai += p.Xc.accessor(i, k) * beta.elements[iParm1++];

    pn = 1 / (1 + gsl_sf_exp(-Xbetai));

    // Add a protection for pn very close to 0 or 1?
    aux = pn * (1 - pn);

    // Calculate sub-gradient vector gn.
    gn = zeros_dmatrix(gn.shape[0], gn.shape[1]);
    gn.elements[0] = 1;
    iParm1 = 1;
    for (size_t k = 0; k < K; ++k) {
      if (p.X.accessor(i, k) > 0)
        gn.elements[p.X.accessor(i, k) - 1 + iParm1] = 1;
      iParm1 += p.nlev.elements[k] - 1;
    }

    // Adding the continuous.
    for (size_t k = 0; k < Kc; ++k) {
      gn.elements[iParm1++] = p.Xc.accessor(i, k);
    }

    for (size_t k1 = 0; k1 < npar; ++k1)
      if (gn.elements[k1] != 0)
        for (size_t k2 = 0; k2 < npar; ++k2)
          if (gn.elements[k2] != 0)
            output.set(k1, k2, output.accessor(k1, k2) + aux * gn.elements[k1] * gn.elements[k2]);
  }
}

double wgsl_mixed_optim_f(DMatrix v, void *params) {
  fix_parm_mixed_T *p = cast(fix_parm_mixed_T *)params;
  return fLogit_mixed(v, p.X, p.nlev, p.Xc, p.y, p.lambdaL1, p.lambdaL2);
}

// Compute both f and df together.
void wgsl_mixed_optim_fdf(DMatrix x, void *params, double *f, DMatrix df) {
  *f = wgsl_mixed_optim_f(x, params);
  wgsl_mixed_optim_df(x, params, df);
}

// Xc is the matrix of continuous covariates, Nobs x Kc (NULL if not used).
int logistic_mixed_fit(DMatrix beta, DMatrix_int X,
                       DMatrix_int nlev, DMatrix Xc, DMatrix y,
                       double lambdaL1, double lambdaL2) {
  // double mLogLik = 0;
  fix_parm_mixed_T p;
  size_t npar = beta.size;
  size_t iter = 0;
  double maxchange = 0;

  // Intializing fix parameters.
  p.X = X;
  p.Xc = Xc;
  p.nlev = nlev;
  p.y = y;
  p.lambdaL1 = lambdaL1;
  p.lambdaL2 = lambdaL2;

  // Initial fit.
  // auto mLogLik = wgsl_mixed_optim_f(beta, &p);

  DMatrix myH = zeros_dmatrix(npar, npar); // Hessian matrix.
  DMatrix stBeta = zeros_dmatrix(1, npar);    // Direction to move.

  DMatrix myG = zeros_dmatrix(1, npar); // Gradient.
  DMatrix tau = zeros_dmatrix(1, npar); // tau for QR.

  for (iter = 0; iter < 100; iter++) {
    wgsl_mixed_optim_hessian(beta, &p, myH); // Calculate Hessian.
    wgsl_mixed_optim_df(beta, &p, myG);      // Calculate Gradient.
    //gsl_linalg_QR_decomp(myH, tau); // TODO          // Calculate next beta.
    //gsl_linalg_QR_solve(myH, tau, myG, stBeta);
    beta = beta - stBeta;

    // Monitor convergence.
    maxchange = 0;
    for (size_t i = 0; i < npar; i++)
      if (maxchange < fabs(stBeta.elements[i]))
        maxchange = fabs(stBeta.elements[i]);

    if (maxchange < 1E-4)
      break;
  }

  // Final fit.
  // mLogLik = wgsl_mixed_optim_f(beta, &p);

  return 0;
}

/***************/
/* Categorical */
/***************/

// I need to bundle all the data that goes to the function to optimze
// together.
struct fix_parm_cat_T{
  DMatrix_int X;
  DMatrix_int nlev;
  DMatrix y;
  double lambdaL1;
  double lambdaL2;
};

double fLogit_cat(DMatrix beta, DMatrix_int X, DMatrix_int nlev,
                  DMatrix y, double lambdaL1, double lambdaL2) {
  size_t n = y.size;
  size_t npar = beta.size;
  double total = 0;
  double aux = 0;

  // omp_set_num_threads(ompthr); /\* Changed loop start at 1 instead
  // of 0 to avoid regularization of beta 0*\/ /\*#pragma omp parallel
  // for reduction (+:total)*\/
  for (size_t i = 1; i < npar; ++i)
    total += beta.elements[i] * beta.elements[i];
  total = (-total * lambdaL2 / 2);

  // /\*#pragma omp parallel for reduction (+:aux)*\/
  for (size_t i = 1; i < npar; ++i)
    aux += (beta.elements[i] > 0) ? beta.elements[i] : -beta.elements[i];
  total = total - aux * lambdaL1;

  // #pragma omp parallel for schedule(static) shared(n,beta,X,nlev,y)
  // #reduction (+:total)
  for (size_t i = 0; i < n; ++i) {
    double Xbetai = beta.elements[0];
    size_t iParm = 1;
    for (size_t k = 0; k < X.shape[1]; ++k) {
      if (X.accessor(i, k) > 0)
        Xbetai += beta.elements[X.accessor(i, k) - 1 + iParm];
      iParm += nlev.elements[k] - 1;
    }
    total += y.elements[i] * Xbetai - gsl_sf_log_1plusx(gsl_sf_exp(Xbetai));
  }
  return -total;
}

void logistic_cat_pred(DMatrix beta,     // Vector of parameters
                                             // length = 1 + Sum_k(C_k-1).
                       DMatrix_int X,    // Matrix Nobs x K
                       DMatrix_int nlev, // Vector with #categories
                       DMatrix yhat) {   // Vector of prob. predicted by
                                             // the logistic.
  for (size_t i = 0; i < X.shape[0]; ++i) {
    double Xbetai = beta.elements[0];
    size_t iParm = 1;
    for (size_t k = 0; k < X.shape[1]; ++k) {
      if (X.accessor(i, k) > 0)
        Xbetai += beta.elements[X.accessor(i, k) - 1 + iParm];
      iParm += nlev.elements[k] - 1;
    }
    yhat.elements[i] = 1 / (1 + gsl_sf_exp(-Xbetai));
  }
}

// The gradient of f, df = (df/dx, df/dy).
void wgsl_cat_optim_df(const DMatrix beta, void *params, DMatrix output) {
  fix_parm_cat_T *p = cast(fix_parm_cat_T *)params;
  size_t n = p.y.size;
  size_t K = p.X.shape[1];
  size_t npar = beta.size;

  // Intitialize gradient output necessary?
  for (size_t i = 0; i < npar; ++i)
    output.elements[i] = 0;

  // Changed loop start at 1 instead of 0 to avoid regularization of beta 0.
  for (size_t i = 1; i < npar; ++i)
    output.elements[i] = p.lambdaL2 * beta.elements[i];
  for (size_t i = 1; i < npar; ++i)
    output.elements[i] += p.lambdaL1 * ((beta.elements[i] > 0) - (beta.elements[i] < 0));

  for (size_t i = 0; i < n; ++i) {
    double pn = 0;
    double Xbetai = beta.elements[0];
    size_t iParm = 1;
    for (size_t k = 0; k < K; ++k) {
      if (p.X.accessor(i, k) > 0)
        Xbetai += beta.elements[p.X.accessor(i, k) - 1 + iParm];
      iParm += p.nlev.elements[k] - 1;
    }

    pn = -(p.y.elements[i] - 1 / (1 + gsl_sf_exp(-Xbetai)));

    output.elements[0] += pn;
    iParm = 1;
    for (size_t k = 0; k < K; ++k) {
      if (p.X.accessor(i, k) > 0)
        output.elements[p.X.accessor(i, k) - 1 + iParm] += pn;
      iParm += p.nlev.elements[k] - 1;
    }
  }
}

// The Hessian of f.
void wgsl_cat_optim_hessian(const DMatrix beta, void *params, DMatrix output) {
  fix_parm_cat_T *p = cast(fix_parm_cat_T *)params;
  size_t n = p.y.size;
  size_t K = p.X.shape[1];
  size_t npar = beta.size;

  // Intitialize Hessian output necessary.
  output = zeros_dmatrix(output.shape[0], output.shape[1]);

  // Changed loop start at 1 instead of 0 to avoid regularization of beta.
  for (size_t i = 1; i < npar; ++i)
    output.set(i, i, p.lambdaL2); // Double-check this.

  // L1 penalty not working yet, as not differentiable, I may need to
  // do coordinate descent (as in glm_net).
  for (size_t i = 0; i < n; ++i) {
    double pn = 0;
    double aux = 0;
    double Xbetai = beta.elements[0];
    size_t iParm2 = 1;
    size_t iParm1 = 1;
    for (size_t k = 0; k < K; ++k) {
      if (p.X.accessor(i, k) > 0)
        Xbetai += beta.elements[p.X.accessor(i, k) - 1 + iParm1];
      iParm1 += p.nlev.elements[k] - 1; //-1?
    }

    pn = 1 / (1 + gsl_sf_exp(-Xbetai));

    // Add a protection for pn very close to 0 or 1?
    aux = pn * (1 - pn);
    output.elements[0] += aux;
    iParm2 = 1;
    for (size_t k2 = 0; k2 < K; ++k2) {
      if (p.X.accessor(i, k2) > 0)
        output.elements[p.X.accessor(i, k2) - 1 + iParm2] += aux;
      iParm2 += p.nlev.elements[k2] - 1; //-1?
    }
    iParm1 = 1;
    for (size_t k1 = 0; k1 < K; ++k1) {
      if (p.X.accessor(i, k1) > 0)
        output.elements[(p.X.accessor(i, k1) - 1 + iParm1) * output.rows] += aux;
      iParm2 = 1;
      for (size_t k2 = 0; k2 < K; ++k2) {
        if ((p.X.accessor(i, k1) > 0) && (p.X.accessor(i, k2) > 0)){
          size_t some_val = ((p.X.accessor(i, k1) - 1 + iParm1 ) * output.rows) + (p.X.accessor(i, k2) - 1 + iParm2);
          output.elements[some_val] += aux;
        }
        iParm2 += p.nlev.elements[k2] - 1; //-1?
      }
      iParm1 += p.nlev.elements[k1] - 1; //-1?
    }
  }
}

double wgsl_cat_optim_f(DMatrix v, void *params) {
  double mLogLik = 0;
  fix_parm_cat_T *p = cast(fix_parm_cat_T *)params;
  mLogLik = fLogit_cat(v, p.X, p.nlev, p.y, p.lambdaL1, p.lambdaL2);
  return mLogLik;
}

// Compute both f and df together.
void wgsl_cat_optim_fdf(DMatrix x, void *params, double *f, DMatrix df) {
  *f = wgsl_cat_optim_f(x, params);
  wgsl_cat_optim_df(x, params, df);
}

int logistic_cat_fit(DMatrix beta, DMatrix_int X, DMatrix_int nlev,
                     DMatrix y, double lambdaL1, double lambdaL2) {
  // double mLogLik = 0;
  fix_parm_cat_T p;
  size_t npar = beta.size;
  size_t iter = 0;
  double maxchange = 0;

  // Intializing fix parameters.
  p.X = X;
  p.nlev = nlev;
  p.y = y;
  p.lambdaL1 = lambdaL1;
  p.lambdaL2 = lambdaL2;

//#ifdef _RPR_DEBUG_
  // Initial fit.
  auto mLogLik = wgsl_cat_optim_f(beta, &p);
//#endif

  DMatrix myH = zeros_dmatrix(npar, npar);    // Hessian matrix.
  DMatrix stBeta = zeros_dmatrix(1, npar);    // Direction to move.

  DMatrix myG = zeros_dmatrix(1, npar);       // Gradient.
  DMatrix tau = zeros_dmatrix(1, npar);       // tau for QR.

  for (iter = 0; iter < 100; iter++) {
    wgsl_cat_optim_hessian(beta, &p, myH); // Calculate Hessian.
    wgsl_cat_optim_df(beta, &p, myG);      // Calculate Gradient.
    //gsl_linalg_QR_decomp(myH, tau);     // TODO    // Calculate next beta.
    //gsl_linalg_QR_solve(myH, tau, myG, stBeta);
    beta = beta - stBeta;

    // Monitor convergence.
    maxchange = 0;
    for (size_t i = 0; i < npar; i++)
      if (maxchange < fabs(stBeta.elements[i]))
        maxchange = fabs(stBeta.elements[i]);

//#ifdef _RPR_DEBUG_
    mLogLik = wgsl_cat_optim_f(beta, &p);
//#endif

    if (maxchange < 1E-4)
      break;
  }

  // Final fit.
  // mLogLik = wgsl_cat_optim_f(beta, &p);

  return 0;
}

/***************/
/* Continuous  */
/***************/

// I need to bundle all the data that goes to the function to optimze
// together.
struct fix_parm_cont_T {
  DMatrix Xc; // continuous covariates; Matrix Nobs x Kc
  DMatrix y;
  double lambdaL1;
  double lambdaL2;
};

double fLogit_cont(const DMatrix beta, const DMatrix Xc, const DMatrix y,
                   double lambdaL1, double lambdaL2) {
  size_t n = y.size;
  size_t npar = beta.size;
  double total = 0;
  double aux = 0;

  // omp_set_num_threads(ompthr); /\* Changed loop start at 1 instead
  // of 0 to avoid regularization of beta_0*\/ /\*#pragma omp parallel
  // for reduction (+:total)*\/
  for (size_t i = 1; i < npar; ++i)
    total += beta.elements[i] * beta.elements[i];
  total = (-total * lambdaL2 / 2);

  // /\*#pragma omp parallel for reduction (+:aux)*\/
  for (size_t i = 1; i < npar; ++i)
    aux += (beta.elements[i]) > 0 ? beta.elements[i] : -beta.elements[i];
  total = total - aux * lambdaL1;

  // #pragma omp parallel for schedule(static) shared(n,beta,X,nlev,y)
  // #reduction (+:total)
  for (size_t i = 0; i < n; ++i) {
    double Xbetai = beta.elements[0];
    size_t iParm = 1;
    for (size_t k = 0; k < (Xc.shape[1]); ++k)
      Xbetai += Xc.accessor(i, k) * beta.elements[iParm++];
    total += y.elements[i] * Xbetai - gsl_sf_log_1plusx(gsl_sf_exp(Xbetai));
  }
  return -total;
}

void logistic_cont_pred(DMatrix beta,   // Vector of parameters
                                            // length = 1 + Sum_k(C_k-1).
                        DMatrix Xc,     // Continuous covariates matrix,
                                            // Nobs x Kc (NULL if not used).
                        DMatrix yhat) { // Vector of prob. predicted by
                                            // the logistic.
  for (size_t i = 0; i < Xc.shape[0]; ++i) {
    double Xbetai = beta.elements[0];
    size_t iParm = 1;
    for (size_t k = 0; k < (Xc.shape[1]); ++k)
      Xbetai += Xc.accessor(i, k) * beta.elements[iParm++];
    yhat.elements[i] = 1 / (1 + gsl_sf_exp(-Xbetai));
  }
}

// The gradient of f, df = (df/dx, df/dy).
void wgsl_cont_optim_df(const DMatrix beta, const void *params, DMatrix output) {
  fix_parm_cont_T *p = cast(fix_parm_cont_T *)params;
  size_t n = p.y.size;
  size_t Kc = p.Xc.shape[1];
  size_t npar = beta.size;

  // size_titialize gradient output necessary?
  for (size_t i = 0; i < npar; ++i)
    output.elements[i] = 0;

  // Changed loop start at 1 instead of 0 to avoid regularization of beta 0.
  for (size_t i = 1; i < npar; ++i)
    output.elements[i] = p.lambdaL2 * beta.elements[i];
  for (size_t i = 1; i < npar; ++i)
    output.elements[i] += p.lambdaL1 * ((beta.elements[i] > 0) - (beta.elements[i] < 0));

  for (size_t i = 0; i < n; ++i) {
    double pn = 0;
    double Xbetai = beta.elements[0];
    size_t iParm = 1;
    for (size_t k = 0; k < Kc; ++k)
      Xbetai += p.Xc.accessor(i, k) * beta.elements[iParm++];

    pn = -(p.y.elements[i] - 1 / (1 + gsl_sf_exp(-Xbetai)));

    output.elements[0] += pn;
    iParm = 1;

    // Adding the continuous.
    for (size_t k = 0; k < Kc; ++k) {
      output.elements[iParm++] += p.Xc.accessor(i, k) * pn;
    }
  }
}

// The Hessian of f.
void wgsl_cont_optim_hessian(const DMatrix beta, void *params,
                             DMatrix output) {
  fix_parm_cont_T *p = cast(fix_parm_cont_T *)params;
  size_t n = p.y.size;
  size_t Kc = p.Xc.shape[1];
  size_t npar = beta.size;
  DMatrix gn = zeros_dmatrix(1, npar); // gn.

  // Intitialize Hessian output necessary ???

  output = zeros_dmatrix(output.shape[0], output.shape[1]);

  // Changed loop start at 1 instead of 0 to avoid regularization of
  // beta 0.
  for (size_t i = 1; i < npar; ++i)
    output.set(i, i, (p.lambdaL2)); // Double-check this.

  // L1 penalty not working yet, as not differentiable, I may need to
  // do coordinate descent (as in glm_net).
  for (size_t i = 0; i < n; ++i) {
    double pn = 0;
    double aux = 0;
    double Xbetai = beta.elements[0];
    size_t iParm1 = 1;
    for (size_t k = 0; k < Kc; ++k)
      Xbetai += p.Xc.accessor(i, k) * beta.elements[iParm1++];

    pn = 1 / (1 + gsl_sf_exp(-Xbetai));

    // Add a protection for pn very close to 0 or 1?
    aux = pn * (1 - pn);

    // Calculate sub-gradient vector gn.
    gn = zeros_dmatrix(1, npar);
    gn.elements[0] = 1;
    iParm1 = 1;
    for (size_t k = 0; k < Kc; ++k) {
      gn.elements[iParm1++] = p.Xc.accessor(i, k);
    }

    for (size_t k1 = 0; k1 < npar; ++k1)
      if (gn.elements[k1] != 0)
        for (size_t k2 = 0; k2 < npar; ++k2)
          if (gn.elements[k2] != 0)
            output.set(k1, k2, output.accessor(k1, k2) + aux * gn.elements[k1] * gn.elements[k2]);
  }
}

double wgsl_cont_optim_f(const DMatrix v, const void *params) {
  double mLogLik = 0;
  fix_parm_cont_T *p = cast(fix_parm_cont_T *)params;
  mLogLik = fLogit_cont(v, p.Xc, p.y, p.lambdaL1, p.lambdaL2);
  return mLogLik;
}

// Compute both f and df together.
void wgsl_cont_optim_fdf(const DMatrix x, const void *params, double *f,
                         DMatrix df) {
  *f = wgsl_cont_optim_f(x, params);
  wgsl_cont_optim_df(x, params, df);
}

int logistic_cont_fit(DMatrix beta,
                      DMatrix Xc, // Continuous covariates matrix,
                                      // Nobs x Kc (NULL if not used).
                      DMatrix y, double lambdaL1, double lambdaL2) {

  fix_parm_cont_T p;
  size_t npar = beta.size;
  size_t iter = 0;
  double maxchange = 0;

  // Initializing fix parameters.
  p.Xc = Xc;
  p.y = y;
  p.lambdaL1 = lambdaL1;
  p.lambdaL2 = lambdaL2;

//#ifdef _RPR_DEBUG_
  // Initial fit.
  auto mLogLik = wgsl_cont_optim_f(beta, &p);
//#endif

  DMatrix myH = zeros_dmatrix(npar, npar); // Hessian matrix.
  DMatrix stBeta = zeros_dmatrix(1, npar);    // Direction to move.

  DMatrix myG = zeros_dmatrix(1, npar); // Gradient.
  DMatrix tau = zeros_dmatrix(1, npar); // tau for QR.

  for (iter = 0; iter < 100; iter++) {
    wgsl_cont_optim_hessian(beta, &p, myH); // Calculate Hessian.
    wgsl_cont_optim_df(beta, &p, myG);      // Calculate Gradient.
    //gsl_linalg_QR_decomp(myH, tau);         // Calculate next beta. //TODO
    //gsl_linalg_QR_solve(myH, tau, myG, stBeta);
    beta = beta - stBeta;

    // Monitor convergence.
    maxchange = 0;
    for (size_t i = 0; i < npar; i++)
      if (maxchange < fabs(stBeta.elements[i]))
        maxchange = fabs(stBeta.elements[i]);

//#ifdef _RPR_DEBUG_
    mLogLik = wgsl_cont_optim_f(beta, &p);
//#endif

    if (maxchange < 1E-4)
      break;
  }

  // Final fit.
  // mLogLik = wgsl_cont_optim_f(beta, &p);

  return 0;
}
/*
   This code is part of faster_lmm_d and published under the GPLv3
   License (see LICENSE.txt)

   Copyright © 2017-2018 Prasun Anand & Pjotr Prins
*/

module faster_lmm_d.bslmm;

import core.stdc.stdlib : exit;

import std.conv;
import core.stdc.time;
import std.exception;
import std.file;
import std.math;
import std.parallelism;
import std.algorithm: min, max, reduce, countUntil;
alias mlog = std.math.log;
import std.process;
import std.range;
import std.stdio;
import std.typecons;
import std.experimental.logger;
import std.string;

import faster_lmm_d.dmatrix;
import faster_lmm_d.gemma_lmm;
import faster_lmm_d.gemma_param;
import faster_lmm_d.helpers;
import faster_lmm_d.optmatrix;
import gsl.rng;
import gsl.randist;

struct pair{
  double first;
  double second;
}

struct pair2{
  size_t first;
  double second;
}

struct HYPBSLMM{
  int id;
  int n_gamma;
  double h;
  double pve, rho, pge, logp;
}

// If a_mode==13, then run probit model.
void MCMC(const DMatrix X, const DMatrix y) {

  HYPBSLMM cHyp_old, cHyp_new;

  DMatrix Result_hyp; // = gsl_matrix_alloc(w_pace, 6);
  DMatrix Result_gamma; // = gsl_matrix_alloc(w_pace, s_max);

  DMatrix Xb_new; // = gsl_vector_alloc(ni_test);
  DMatrix Xb_old; // = gsl_vector_alloc(ni_test);
  DMatrix z_hat;  // = gsl_vector_alloc(ni_test);
  DMatrix z;      // = gsl_vector_alloc(ni_test);

  DMatrix Xgamma_old; // = gsl_matrix_alloc(ni_test, s_max);
  DMatrix XtX_old;    // = gsl_matrix_alloc(s_max, s_max);
  DMatrix Xtz_old;    // = gsl_vector_alloc(s_max);
  DMatrix beta_old;   // = gsl_vector_alloc(s_max);

  DMatrix Xgamma_new; // = gsl_matrix_alloc(ni_test, s_max);
  DMatrix XtX_new; // = gsl_matrix_alloc(s_max, s_max);
  DMatrix Xtz_new;  // = gsl_vector_alloc(s_max);
  DMatrix beta_new;   // = gsl_vector_alloc(s_max);

  double ztz = 0.0;

  // define later
  size_t w_step, s_step;

  //gsl_vector_memcpy(z, y);

  // For quantitative traits, y is centered already in
  // gemma.cpp, but just in case.
  double mean_z = CenterVector(z);
  ztz = vector_ddot(z, z);

  double logPost_new, logPost_old;
  double logMHratio;

  //define later
  size_t w_pace, s_max, ns_test, randseed, ni_test;
  int a_mode;
  double pheno_mean;
  size_t[] mapRank2pos;

  Result_gamma = zeros_dmatrix(w_pace, s_max);
  if (a_mode == 13) {
    pheno_mean = 0.0;
  }

  pair[] beta_g;
  for (size_t i = 0; i < ns_test; i++) {
    beta_g ~= pair(0.0, 0.0);
  }

  size_t[] rank_new, rank_old;
  pair2[] pos_loglr;

  MatrixCalcLmLR(X, z, pos_loglr);

  //stable_sort(pos_loglr.begin(), pos_loglr.end(), comp_lr);
  for (size_t i = 0; i < ns_test; ++i) {
    mapRank2pos[i] = pos_loglr[i].first;
  }

  // Calculate proposal distribution for gamma (unnormalized),
  // and set up gsl_r and gsl_t.
  gsl_rng_env_setup();
  const gsl_rng_type *gslType = gsl_rng_default;

  if (randseed < 0) {
    time_t rawtime;
    time(&rawtime);
    tm *ptm = gmtime(&rawtime);

    randseed =
        to!size_t(ptm.tm_hour % 24 * 3600 + ptm.tm_min * 60 + ptm.tm_sec);
  }
  gsl_rng* gsl_r = gsl_rng_alloc(gslType);
  gsl_rng_set(gsl_r, randseed);

  double[] p_gamma = new double[ns_test];
  CalcPgamma(p_gamma);

  gsl_ran_discrete_t* gsl_t = gsl_ran_discrete_preproc(ns_test, p_gamma.ptr);

  // Initial parameters.
  InitialMCMC(X, z, rank_old, cHyp_old, pos_loglr);

  cHyp_initial = cHyp_old;

  if (cHyp_old.n_gamma == 0) {
    logPost_old = CalcPosterior(ztz, cHyp_old);
  } else {
    SetXgamma(Xgamma_old, X, rank_old);
    CalcXtX(Xgamma_old, z, rank_old.length, XtX_old, Xtz_old);
    logPost_old = CalcPosterior(Xgamma_old, XtX_old, Xtz_old, ztz,
                                rank_old.length, Xb_old, beta_old, cHyp_old);
  }

  // Calculate centered z_hat, and pve.
  if (a_mode == 13) {
    if (cHyp_old.n_gamma == 0) {
      CalcCC_PVEnZ(z_hat, cHyp_old);
    } else {
      CalcCC_PVEnZ(Xb_old, z_hat, cHyp_old);
    }
  }

  // Start MCMC.
  int accept;
  size_t total_step = w_step + s_step;
  size_t w = 0, w_col, pos;
  size_t repeat = 0;

  for (size_t t = 0; t < total_step; ++t) {

    if (a_mode == 13) {
      SampleZ(y, z_hat, z);
      mean_z = CenterVector(z);
      ztz = vector_ddot(z, z);

      // First proposal.
      if (cHyp_old.n_gamma == 0) {
        logPost_old = CalcPosterior(ztz, cHyp_old);
      } else {
        //gsl_matrix_view
        DMatrix Xold_sub = get_sub_dmatrix(Xgamma_old, 0, 0, ni_test, rank_old.length);
        //gsl_vector_view
        DMatrix Xtz_sub; // = gsl_vector_subvector(Xtz_old, 0, rank_old.length);
        Xtz_sub = matrix_mult(Xold_sub.T, z);
        logPost_old = CalcPosterior(Xgamma_old, XtX_old, Xtz_old, ztz, rank_old.length, Xb_old, beta_old, cHyp_old);
      }
    }

    // M-H steps.
    for (size_t i = 0; i < n_mh; ++i) {
      if (gsl_rng_uniform(gsl_r) < 0.33) {
        repeat = 1 + gsl_rng_uniform_int(gsl_r, 20);
      } else {
        repeat = 1;
      }

      logMHratio = 0.0;
      logMHratio += ProposeHnRho(cHyp_old, cHyp_new, repeat);
      logMHratio +=
          ProposeGamma(rank_old, rank_new, p_gamma, cHyp_old, cHyp_new, repeat);
      logMHratio += ProposePi(cHyp_old, cHyp_new, repeat);

      if (cHyp_new.n_gamma == 0) {
        logPost_new = CalcPosterior(ztz, cHyp_new);
      } else {

        // This makes sure that rank_old.length ==
        // rank_remove.length does not happen.
        if (cHyp_new.n_gamma <= 20 || cHyp_old.n_gamma <= 20) {
          SetXgamma(Xgamma_new, X, rank_new);
          CalcXtX(Xgamma_new, z, rank_new.length, XtX_new, Xtz_new);
        } else {
          SetXgamma(X, Xgamma_old, XtX_old, Xtz_old, z, rank_old, rank_new,
                    Xgamma_new, XtX_new, Xtz_new);
        }
        logPost_new =
            CalcPosterior(Xgamma_new, XtX_new, Xtz_new, ztz, rank_new.length,
                          Xb_new, beta_new, cHyp_new);
      }
      logMHratio += logPost_new - logPost_old;

      if (logMHratio > 0 || mlog(gsl_rng_uniform(gsl_r)) < logMHratio) {
        accept = 1;
        n_accept++;
      } else {
        accept = 0;
      }

      if (accept == 1) {
        logPost_old = logPost_new;
        cHyp_old = cHyp_new;
        gsl_vector_memcpy(Xb_old, Xb_new);

        rank_old.clear();
        if (rank_new.length != 0) {
          for (size_t i = 0; i < rank_new.length; ++i) {
            rank_old ~= rank_new[i];
          }

          //gsl_matrix_view
          DMatrix Xold_sub = get_sub_dmatrix(Xgamma_old, 0, 0, ni_test, rank_new.length);
          //gsl_matrix_view
          DMatrix XtXold_sub = get_sub_dmatrix(XtX_old, 0, 0, rank_new.length, rank_new.length);
          //gsl_vector_view
          DMatrix Xtzold_sub; // = gsl_vector_subvector(Xtz_old, 0, rank_new.length);
          //gsl_vector_view
          DMatrix betaold_sub; // = gsl_vector_subvector(beta_old, 0, rank_new.length);

          //gsl_matrix_view
          DMatrix Xnew_sub = get_sub_dmatrix(Xgamma_new, 0, 0, ni_test, rank_new.length);
          //gsl_matrix_view
          DMatrix XtXnew_sub = get_sub_dmatrix(XtX_new, 0, 0, rank_new.length, rank_new.length);
          //gsl_vector_view
          DMatrix Xtznew_sub; // = gsl_vector_subvector(Xtz_new, 0, rank_new.length);
          //gsl_vector_view
          DMatrix betanew_sub; // = gsl_vector_subvector(beta_new, 0, rank_new.length);

          //gsl_matrix_memcpy(Xold_sub, Xnew_sub);
          //gsl_matrix_memcpy(XtXold_sub, XtXnew_sub);
          //gsl_vector_memcpy(Xtzold_sub, Xtznew_sub);
          //gsl_vector_memcpy(betaold_sub, betanew_sub);
        }
      } else {
        cHyp_new = cHyp_old;
      }
    }

    // Calculate z_hat, and pve.
    if (a_mode == 13) {
      if (cHyp_old.n_gamma == 0) {
        CalcCC_PVEnZ(z_hat, cHyp_old);
      } else {
        CalcCC_PVEnZ(Xb_old, z_hat, cHyp_old);
      }

      // Sample mu and update z_hat.
      z = subtract_dmatrix(z, z_hat);
      mean_z += CenterVector(z);
      mean_z += gsl_ran_gaussian(gsl_r, sqrt(1.0 / to!double(ni_test)));

      z_hat = add_dmatrix_num(z_hat, mean_z);
    }

    // define
    size_t w_step, r_pace;


    // Save data.
    if (t < w_step) {
      continue;
    } else {
      if (t % r_pace == 0) {
        w_col = w % w_pace;
        if (w_col == 0) {
          if (w == 0) {
            WriteResult(0, Result_hyp, Result_gamma, w_col);
          } else {
            WriteResult(1, Result_hyp, Result_gamma, w_col);
            Result_hyp = zeros_dmatrix(Result_hyp.shape[0], Result_hyp.shape[1]);
            Result_gamma = zeros_dmatrix(Result_gamma.shape[0], Result_gamma.shape[1]);
          }
        }

        Result_hyp.set(w_col, 0, cHyp_old.h);
        Result_hyp.set(w_col, 1, cHyp_old.pve);
        Result_hyp.set(w_col, 2, cHyp_old.rho);
        Result_hyp.set(w_col, 3, cHyp_old.pge);
        Result_hyp.set(w_col, 4, cHyp_old.logp);
        Result_hyp.set(w_col, 5, cHyp_old.n_gamma);

        for (size_t i = 0; i < cHyp_old.n_gamma; ++i) {
          pos = mapRank2pos[rank_old[i]] + 1;
          Result_hyp.set(w_col, i, pos);

          beta_g[pos - 1].first += beta_old.elements[i];
          beta_g[pos - 1].second += 1.0;
        }

        if (a_mode == 13) {
          pheno_mean += mean_z;
        }

        w++;
      }
    }
  }
  writeln("\n");

  w_col = w % w_pace;
  WriteResult(1, Result_hyp, Result_gamma, w_col);

  DMatrix alpha; // = gsl_vector_alloc(ns_test);
  alpha = zeros_dmatrix(alpha.shape[0], alpha.shape[1]);
  WriteParam(beta_g, alpha, w);

  return;
}

// If a_mode==13 then Uty==y.
void MCMC(const DMatrix U, const DMatrix UtX,
                 const DMatrix Uty, const DMatrix K_eval,
                 const DMatrix y) {

  HYPBSLMM cHyp_old, cHyp_new;

  // define later
  size_t w_pace, s_max, ns_test, randseed;
  int a_mode;
  double pheno_mean;

  DMatrix Result_hyp; // = gsl_matrix_alloc(w_pace, 6);
  DMatrix Result_gamma = zeros_dmatrix(w_pace, s_max);

  DMatrix alpha_prime; // = gsl_vector_alloc(ni_test);
  DMatrix alpha_new; // = gsl_vector_alloc(ni_test);
  DMatrix alpha_old; // = gsl_vector_alloc(ni_test);
  DMatrix Utu; // = gsl_vector_alloc(ni_test);
  DMatrix Utu_new; // = gsl_vector_alloc(ni_test);
  DMatrix Utu_old; // = gsl_vector_alloc(ni_test);

  DMatrix UtXb_new; // = gsl_vector_alloc(ni_test);
  DMatrix UtXb_old; // = gsl_vector_alloc(ni_test);

  DMatrix z_hat; // = gsl_vector_alloc(ni_test);
  DMatrix z; // = gsl_vector_alloc(ni_test);
  DMatrix Utz; // = gsl_vector_alloc(ni_test);

  //gsl_vector_memcpy(Utz, Uty);


  double logPost_new, logPost_old;
  double logMHratio;
  double mean_z = 0.0;

  Utu = zeros_dmatrix(Utu.shape[0], Utu.shape[1]);
  alpha_prime = zeros_dmatrix(alpha_prime.shape[0], alpha_prime.shape[1]);
  if (a_mode == 13) {
    pheno_mean = 0.0;
  }

  pair[] beta_g;
  for (size_t i = 0; i < ns_test; i++) {
    beta_g ~= (make_pair(0.0, 0.0));
  }

  size_t[] rank_new, rank_old;
  double[] beta_new, beta_old;

  pair2[] pos_loglr;

  //MatrixCalcLR(U, UtX, Utz, K_eval, l_min, l_max, n_region, pos_loglr);

  stable_sort(pos_loglr.begin(), pos_loglr.end(), comp_lr);
  for (size_t i = 0; i < ns_test; ++i) {
    mapRank2pos[i] = pos_loglr[i].first;
  }

  // Calculate proposal distribution for gamma (unnormalized),
  // and set up gsl_r and gsl_t.
  gsl_rng_env_setup();
  const gsl_rng_type *gslType;
  gslType = gsl_rng_default;
  if (randseed < 0) {
    time_t rawtime;
    time(&rawtime);
    tm *ptm = gmtime(&rawtime);

    randseed = to!size_t(ptm.tm_hour % 24 * 3600 + ptm.tm_min * 60 + ptm.tm_sec);
  }
  gsl_rng* gsl_r = gsl_rng_alloc(gslType);
  gsl_rng_set(gsl_r, randseed);

  double *p_gamma = new double[ns_test];
  CalcPgamma(p_gamma);

  gsl_t = gsl_ran_discrete_preproc(ns_test, p_gamma);

  // Initial parameters.
  InitialMCMC(UtX, Utz, rank_old, cHyp_old, pos_loglr);

  cHyp_initial = cHyp_old;

  if (cHyp_old.n_gamma == 0 || cHyp_old.rho == 0) {
    logPost_old = CalcPosterior(Utz, K_eval, Utu_old, alpha_old, cHyp_old);

    beta_old = [];
    for (size_t i = 0; i < cHyp_old.n_gamma; ++i) {
      beta_old ~= 0;
    }
  } else {
    DMatrix UtXgamma; // = gsl_matrix_alloc(ni_test, cHyp_old.n_gamma);
    DMatrix beta; // = gsl_vector_alloc(cHyp_old.n_gamma);
    SetXgamma(UtXgamma, UtX, rank_old);
    logPost_old = CalcPosterior(UtXgamma, Utz, K_eval, UtXb_old, Utu_old, alpha_old, beta, cHyp_old);

    beta_old = [];
    for (size_t i = 0; i < beta.size; ++i) {
      beta_old.push_back(gsl_vector_get(beta, i));
    }
  }

  // Calculate centered z_hat, and pve.
  if (a_mode == 13) {
    if (cHyp_old.n_gamma == 0 || cHyp_old.rho == 0) {
      CalcCC_PVEnZ(U, Utu_old, z_hat, cHyp_old);
    } else {
      CalcCC_PVEnZ(U, UtXb_old, Utu_old, z_hat, cHyp_old);
    }
  }

  // Start MCMC.
  int accept;
  size_t total_step = w_step + s_step;
  size_t w = 0, w_col, pos;
  size_t repeat = 0;

  for (size_t t = 0; t < total_step; ++t) {
    if (t % d_pace == 0 || t == total_step - 1) {
    }

    if (a_mode == 13) {
      SampleZ(y, z_hat, z);
      mean_z = CenterVector(z);

      Utz = matrix_mult(U.T, z);

      // First proposal.
      if (cHyp_old.n_gamma == 0 || cHyp_old.rho == 0) {
        logPost_old = CalcPosterior(Utz, K_eval, Utu_old, alpha_old, cHyp_old);
        beta_old.clear();
        for (size_t i = 0; i < cHyp_old.n_gamma; ++i) {
          beta_old.push_back(0);
        }
      } else {
        DMatrix UtXgamma; // = gsl_matrix_alloc(ni_test, cHyp_old.n_gamma);
        DMatrix beta; // = gsl_vector_alloc(cHyp_old.n_gamma);
        SetXgamma(UtXgamma, UtX, rank_old);
        logPost_old = CalcPosterior(UtXgamma, Utz, K_eval, UtXb_old, Utu_old,
                                    alpha_old, beta, cHyp_old);

        beta_old = [];
        for (size_t i = 0; i < beta.size; ++i) {
          beta_old ~= beta.elements[i];
        }
      }
    }

    // M-H steps.
    for (size_t i = 0; i < n_mh; ++i) {
      if (gsl_rng_uniform(gsl_r) < 0.33) {
        repeat = 1 + gsl_rng_uniform_int(gsl_r, 20);
      } else {
        repeat = 1;
      }

      logMHratio = 0.0;
      logMHratio += ProposeHnRho(cHyp_old, cHyp_new, repeat);
      logMHratio +=
          ProposeGamma(rank_old, rank_new, p_gamma, cHyp_old, cHyp_new, repeat);
      logMHratio += ProposePi(cHyp_old, cHyp_new, repeat);

      if (cHyp_new.n_gamma == 0 || cHyp_new.rho == 0) {
        logPost_new = CalcPosterior(Utz, K_eval, Utu_new, alpha_new, cHyp_new);
        beta_new.clear();
        for (size_t i = 0; i < cHyp_new.n_gamma; ++i) {
          beta_new.push_back(0);
        }
      } else {
        DMatrix UtXgamma; // = gsl_matrix_alloc(ni_test, cHyp_new.n_gamma);
        DMatrix beta; // = gsl_vector_alloc(cHyp_new.n_gamma);
        SetXgamma(UtXgamma, UtX, rank_new);
        logPost_new = CalcPosterior(UtXgamma, Utz, K_eval, UtXb_new, Utu_new,
                                    alpha_new, beta, cHyp_new);
        beta_new = [];
        for (size_t i = 0; i < beta.size; ++i) {
          beta_new ~= beta.elements[i];
        }
      }

      logMHratio += logPost_new - logPost_old;

      if (logMHratio > 0 || log(gsl_rng_uniform(gsl_r)) < logMHratio) {
        accept = 1;
        n_accept++;
      } else {
        accept = 0;
      }

      if (accept == 1) {
        logPost_old = logPost_new;
        rank_old = [];
        beta_old = [];
        if (rank_new.length != 0) {
          for (size_t i = 0; i < rank_new.length; ++i) {
            rank_old ~= rank_new[i];
            beta_old ~= beta_new[i];
          }
        }
        cHyp_old = cHyp_new;
        gsl_vector_memcpy(alpha_old, alpha_new);
        gsl_vector_memcpy(UtXb_old, UtXb_new);
        gsl_vector_memcpy(Utu_old, Utu_new);
      } else {
        cHyp_new = cHyp_old;
      }
    }

    // Calculate z_hat, and pve.
    if (a_mode == 13) {
      if (cHyp_old.n_gamma == 0 || cHyp_old.rho == 0) {
        CalcCC_PVEnZ(U, Utu_old, z_hat, cHyp_old);
      } else {
        CalcCC_PVEnZ(U, UtXb_old, Utu_old, z_hat, cHyp_old);
      }

      // Sample mu and update z_hat.
      gsl_vector_sub(z, z_hat);
      mean_z += CenterVector(z);
      mean_z += gsl_ran_gaussian(gsl_r, sqrt(1.0 / to!double(ni_test)));
      z_hat = add_dmatrix_num(z_hat, mean_z);

    }

    // Save data.
    if (t < w_step) {
      continue;
    } else {
      if (t % r_pace == 0) {
        w_col = w % w_pace;
        if (w_col == 0) {
          if (w == 0) {
            WriteResult(0, Result_hyp, Result_gamma, w_col);
          } else {
            WriteResult(1, Result_hyp, Result_gamma, w_col);
            Result_hyp = zeros_dmatrix(Result_hyp.shape[0], Result_hyp.shape[1]);
            Result_gamma = zeros_dmatrix(Result_gamma.shape[0], Result_gamma.shape[1]);
          }
        }

        Result_hyp.set(w_col, 0, cHyp_old.h);
        Result_hyp.set(w_col, 1, cHyp_old.pve);
        Result_hyp.set(w_col, 2, cHyp_old.rho);
        Result_hyp.set(w_col, 3, cHyp_old.pge);
        Result_hyp.set(w_col, 4, cHyp_old.logp);
        Result_hyp.set(w_col, 5, cHyp_old.n_gamma);

        for (size_t i = 0; i < cHyp_old.n_gamma; ++i) {
          pos = mapRank2pos[rank_old[i]] + 1;

          Result_gamma.set(w_col, i, pos);

          beta_g[pos - 1].first += beta_old[i];
          beta_g[pos - 1].second += 1.0;
        }

        alpha_prime = add_dmatrix(alpha_prime, alpha_old);
        Utu = add_dmatrix(Utu, Utu_old);

        if (a_mode == 13) {
          pheno_mean += mean_z;
        }

        w++;
      }
    }
  }
  write("\n");

  w_col = w % w_pace;
  WriteResult(1, Result_hyp, Result_gamma, w_col);

  alpha_prime = divide_dmatrix_num(alpha_prime, to!double(w));
  Utu = divide_dmatrix_num(Utu, to!double(w));
  if (a_mode == 13) {
    pheno_mean /= to!double(w);
  }

  //DMatrix alpha = matrix_mult(CblasTrans, 1.0 / (double)ns_test, UtX, alpha_prime, 0.0, alpha);
  WriteParam(beta_g, alpha, w);

  alpha_prime = matrix_mult(U, Utu);
  WriteBV(alpha_prime);


  return;
}

// Make sure that both y and X are centered already.
void MatrixCalcLmLR(const DMatrix X, const DMatrix y,
                    pair2[] pos_loglr) {
  double yty, xty, xtx, log_lr;
  yty = vector_ddot(y, y);

  for (size_t i = 0; i < X.shape[1]; ++i) {
    //gsl_vector_const_view
    DMatrix X_col = get_col(X, i);
    xtx = vector_ddot(X_col, X_col);
    xty = vector_ddot(X_col, y);

    log_lr = 0.5 * to!double(y.size )* (log(yty) - log(yty - xty * xty / xtx));
    pos_loglr ~= pair2(i, log_lr);
  }

  return;
}

// Center the vector y.
double CenterVector(DMatrix y) {
  double d = 0.0;

  for (size_t i = 0; i < y.size; ++i) {
    d += y.elements[i];
  }
  d /= to!double(y.size);

  sub_dmatrix_num(y, d);

  return d;
}

// Center the vector y.
void CenterVector(DMatrix y, const DMatrix W) {
  DMatrix WtW; // = gsl_matrix_safe_alloc(W->size2, W->size2);
  DMatrix Wty; // = gsl_vector_safe_alloc(W->size2);
  DMatrix WtWiWty; // = gsl_vector_safe_alloc(W->size2);

  WtW = matrix_mult(W.T, W);
  Wty = matrix_mult(W.T, y);

  int sig;
  gsl_permutation *pmt = gsl_permutation_alloc(W.shape[1]);
  LUDecomp(WtW, pmt, &sig);
  LUSolve(WtW, pmt, Wty, WtWiWty);

  // note -1
  //gsl_blas_dgemv(CblasNoTrans, -1.0, W, WtWiWty, 1.0, y);

  return;
}

void InitialMCMC(const DMatrix UtX, const DMatrix Uty,
                  size_t[] rank, HYPBSLMM cHyp,
                  pair2[] pos_loglr) {
  double q_genome = gsl_cdf_chisq_Qinv(0.05 / to!double(ns_test), 1);

  cHyp.n_gamma = 0;
  for (size_t i = 0; i < pos_loglr.length; ++i) {
    if (2.0 * pos_loglr[i].second > q_genome) {
      cHyp.n_gamma++;
    }
  }
  if (cHyp.n_gamma < 10) {
    cHyp.n_gamma = 10;
  }

  if (cHyp.n_gamma > s_max) {
    cHyp.n_gamma = s_max;
  }
  if (cHyp.n_gamma < s_min) {
    cHyp.n_gamma = s_min;
  }

  rank = [];
  for (size_t i = 0; i < cHyp.n_gamma; ++i) {
    rank ~= i;
  }

  cHyp.logp = mlog(to!double(cHyp.n_gamma) / to!double(ns_test));
  cHyp.h = pve_null;

  if (cHyp.logp == 0) {
    cHyp.logp = -0.000001;
  }
  if (cHyp.h == 0) {
    cHyp.h = 0.1;
  }

  DMatrix UtXgamma; // = gsl_matrix_alloc(ni_test, cHyp.n_gamma);
  SetXgamma(UtXgamma, UtX, rank);
  double sigma_a2;
  if (trace_G != 0) {
    sigma_a2 = cHyp.h * 1.0 /
               (trace_G * (1 - cHyp.h) * exp(cHyp.logp) * to!double(ns_test));
  } else {
    sigma_a2 = cHyp.h * 1.0 / ((1 - cHyp.h) * exp(cHyp.logp) * to!double(ns_test));
  }
  if (sigma_a2 == 0) {
    sigma_a2 = 0.025;
  }
  cHyp.rho = CalcPveLM(UtXgamma, Uty, sigma_a2) / cHyp.h;

  if (cHyp.rho > 1.0) {
    cHyp.rho = 1.0;
  }

  if (cHyp.h < h_min) {
    cHyp.h = h_min;
  }
  if (cHyp.h > h_max) {
    cHyp.h = h_max;
  }
  if (cHyp.rho < rho_min) {
    cHyp.rho = rho_min;
  }
  if (cHyp.rho > rho_max) {
    cHyp.rho = rho_max;
  }
  if (cHyp.logp < logp_min) {
    cHyp.logp = logp_min;
  }
  if (cHyp.logp > logp_max) {
    cHyp.logp = logp_max;
  }

  writeln("initial value of h = ", cHyp.h);
  writeln("initial value of rho = ", cHyp.rho);
  writeln("initial value of pi = ", exp(cHyp.logp));
  writeln("initial value of |gamma| = ", cHyp.n_gamma);

  return;
}

double CalcPosterior(const DMatrix Uty, const DMatrix K_eval,
                      DMatrix Utu, DMatrix alpha_prime, HYPBSLMM cHyp) {

  double sigma_b2 = cHyp.h * (1.0 - cHyp.rho) / (trace_G * (1 - cHyp.h));

  DMatrix Utu_rand; // = gsl_vector_alloc(Uty.size);
  DMatrix weight_Hi; // = gsl_vector_alloc(Uty.size);

  double logpost = 0.0;
  double d, ds, uy, Hi_yy = 0, logdet_H = 0.0;
  for (size_t i = 0; i < ni_test; ++i) {
    d = K_eval.elements[i] * sigma_b2;
    ds = d / (d + 1.0);
    d = 1.0 / (d + 1.0);
    weight_Hi.elements[i] = d;

    logdet_H -= mlog(d);
    uy = Uty.elements[i];
    Hi_yy += d * uy * uy;

    Utu_rand.elements[i] =gsl_ran_gaussian(gsl_r, 1) * sqrt(ds);
  }

  // Sample tau.
  double tau = 1.0;
  if (a_mode == 11) {
    tau = gsl_ran_gamma(gsl_r, to!double(ni_test) / 2.0, 2.0 / Hi_yy);
  }

  // Sample alpha.
  //gsl_vector_memcpy(alpha_prime, Uty);
  alpha_prime = slow_multiply_dmatrix(alpha_prime, weight_Hi);
  alpha_prime = multiply_dmatrix_num(alpha_prime, sigma_b2);

  // Sample u.
  //gsl_vector_memcpy(Utu, alpha_prime);
  Utu  = slow_multiply_dmatrix(Utu, K_eval);
  if (a_mode == 11) {
    Utu_rand = multiply_dmatrix_num(Utu_rand, sqrt(1.0 / tau));
  }
  Utu = add_dmatrix(Utu, Utu_rand);

  // For quantitative traits, calculate pve and ppe.
  if (a_mode == 11) {
    d = vector_ddot(Utu, Utu);
    cHyp.pve = d / to!double(ni_test);
    cHyp.pve /= cHyp.pve + 1.0 / tau;
    cHyp.pge = 0.0;
  }

  // Calculate likelihood.
  logpost = -0.5 * logdet_H;
  if (a_mode == 11) {
    logpost -= 0.5 * to!double(ni_test) * mlog(Hi_yy);
  } else {
    logpost -= 0.5 * Hi_yy;
  }

  logpost += (to!double(cHyp.n_gamma) - 1.0) * cHyp.logp +
             (to!double(ns_test) - to!double(cHyp.n_gamma)) * mlog(1 - exp(cHyp.logp));


  return logpost;
}

double CalcPosterior(const DMatrix UtXgamma, const DMatrix Uty,
                            const DMatrix K_eval, DMatrix UtXb,
                            DMatrix Utu, DMatrix alpha_prime,
                            DMatrix beta, HYPBSLMM cHyp) {
  clock_t time_start;

  double sigma_a2 = cHyp.h * cHyp.rho /
                    (trace_G * (1 - cHyp.h) * exp(cHyp.logp) * to!double(ns_test));
  double sigma_b2 = cHyp.h * (1.0 - cHyp.rho) / (trace_G * (1 - cHyp.h));

  double logpost = 0.0;
  double d, ds, uy, P_yy = 0, logdet_O = 0.0, logdet_H = 0.0;

  DMatrix UtXgamma_eval; // = gsl_matrix_alloc(UtXgamma->size1, UtXgamma->size2);
  DMatrix Omega; // = gsl_matrix_alloc(UtXgamma->size2, UtXgamma->size2);
  DMatrix XtHiy; // = gsl_vector_alloc(UtXgamma->size2);
  DMatrix beta_hat; // = gsl_vector_alloc(UtXgamma->size2);
  DMatrix Utu_rand; // = gsl_vector_alloc(UtXgamma->size1);
  DMatrix weight_Hi; // = gsl_vector_alloc(UtXgamma->size1);

  //gsl_matrix_memcpy(UtXgamma_eval, UtXgamma);

  logdet_H = 0.0;
  P_yy = 0.0;
  for (size_t i = 0; i < ni_test; ++i) {
    //gsl_vector_view
    DMatrix UtXgamma_row = get_row(UtXgamma_eval, i);
    d = K_eval.elements[i] * sigma_b2;
    ds = d / (d + 1.0);
    d = 1.0 / (d + 1.0);
    weight_Hi.elements[i] = d;

    logdet_H -= mlog(d);
    uy = Uty.elements[i];
    P_yy += d * uy * uy;
    UtXgamma_row = multiply_dmatrix_num(UtXgamma_row, d);

    Utu_rand.elements[i] = gsl_ran_gaussian(gsl_r, 1) * sqrt(ds);
  }

  // Calculate Omega.
  Omega = ones_dmatrix(Omega.shape[0], Omega.shape[1]);

  //lapack_dgemm((char *)"T", (char *)"N", sigma_a2, UtXgamma_eval, UtXgamma, 1.0, Omega);

  // Calculate beta_hat.
  XtHiy = matrix_mult(UtXgamma_eval.T, Uty);

  logdet_O = CholeskySolve(Omega, XtHiy, beta_hat);

  gsl_vector_scale(beta_hat, sigma_a2);

  d = vector_ddot(XtHiy, beta_hat);
  P_yy -= d;

  // Sample tau.
  double tau = 1.0;
  if (a_mode == 11) {
    tau = gsl_ran_gamma(gsl_r,  to!double(ni_test) / 2.0, 2.0 / P_yy);
  }

  // Sample beta.
  for (size_t i = 0; i < beta.size; i++) {
    d = gsl_ran_gaussian(gsl_r, 1);
    beta.elements[i] = d;
  }
  gsl_blas_dtrsv(CblasUpper, CblasNoTrans, CblasNonUnit, Omega, beta);

  // This computes inv(L^T(Omega)) %*% beta.
  gsl_vector_scale(beta, sqrt(sigma_a2 / tau));
  gsl_vector_add(beta, beta_hat);
  gsl_blas_dgemv(CblasNoTrans, 1.0, UtXgamma, beta, 0.0, UtXb);

  // Sample alpha.
  //gsl_vector_memcpy(alpha_prime, Uty);
  alpha_prime =  subtract_dmatrix(alpha_prime, UtXb);
  alpha_prime = slow_multiply_dmatrix(alpha_prime, weight_Hi);
  alpha_prime = multiply_dmatrix_num(alpha_prime, sigma_b2);

  // Sample u.
  //gsl_vector_memcpy(Utu, alpha_prime);
  Utu = slow_multiply_dmatrix(Utu, K_eval);

  if (a_mode == 11) {
    Utu_rand = multiply_dmatrix_num(Utu_rand, sqrt(1.0 / tau));
  }
  Utu = add_dmatrix(Utu, Utu_rand);

  // For quantitative traits, calculate pve and pge.
  if (a_mode == 11) {
    d = vector_ddot(UtXb, UtXb);
    cHyp.pge = d / to!double(ni_test);

    d = vector_ddot(Utu, Utu);
    cHyp.pve = cHyp.pge + d / to!double(ni_test);

    if (cHyp.pve == 0) {
      cHyp.pge = 0.0;
    } else {
      cHyp.pge /= cHyp.pve;
    }
    cHyp.pve /= cHyp.pve + 1.0 / tau;
  }

  logpost = -0.5 * logdet_H - 0.5 * logdet_O;
  if (a_mode == 11) {
    logpost -= 0.5 * to!double(ni_test) * mlog(P_yy);
  } else {
    logpost -= 0.5 * P_yy;
  }
  logpost +=
      (to!double(cHyp.n_gamma) - 1.0) * cHyp.logp +
      (to!double(ns_test) - to!double(cHyp.n_gamma)) * log(1.0 - exp(cHyp.logp));

  return logpost;
}

double CalcPosterior(const double yty, HYPBSLMM cHyp) {
  double logpost = 0.0;

  // For quantitative traits, calculate pve and pge.
  // Pve and pge for case/control data are calculted in CalcCC_PVEnZ.
  if (a_mode == 11) {
    cHyp.pve = 0.0;
    cHyp.pge = 1.0;
  }

  // Calculate likelihood.
  if (a_mode == 11) {
    logpost -= 0.5 * to!double(ni_test) * mlog(yty);
  } else {
    logpost -= 0.5 * yty;
  }

  logpost += (to!double(cHyp.n_gamma) - 1.0) * cHyp.logp +
             (to!double(ns_test) - to!double(cHyp.n_gamma)) * mlog(1 - exp(cHyp.logp));

  return logpost;
}

double CalcPosterior(const DMatrix Xgamma, const DMatrix XtX,
                      const DMatrix Xty, const double yty,
                      const size_t s_size, DMatrix Xb,
                      DMatrix beta, HYPBSLMM cHyp) {
  double sigma_a2 = cHyp.h / ((1 - cHyp.h) * exp(cHyp.logp) * to!double(ns_test));
  double logpost = 0.0;
  double d, P_yy = yty, logdet_O = 0.0;

  //gsl_matrix_const_view
  DMatrix Xgamma_sub = get_sub_dmatrix(Xgamma, 0, 0, Xgamma.shape[0], s_size);
  //gsl_matrix_const_view
  DMatrix XtX_sub = get_sub_dmatrix(XtX, 0, 0, s_size, s_size);
  //gsl_vector_const_view Xty_sub = gsl_vector_const_subvector(Xty, 0, s_size);

  DMatrix Omega; // = gsl_matrix_alloc(s_size, s_size);
  DMatrix M_temp; // = gsl_matrix_alloc(s_size, s_size);
  DMatrix beta_hat; // = gsl_vector_alloc(s_size);
  DMatrix Xty_temp; // = gsl_vector_alloc(s_size);

  //gsl_vector_memcpy(Xty_temp, &Xty_sub.vector);

  // Calculate Omega.
  //gsl_matrix_memcpy(Omega, &XtX_sub.matrix);
  Omega = multiply_dmatrix_num(Omega, sigma_a2);
  //gsl_matrix_set_identity(M_temp);
  Omega = add_dmatrix(Omega, M_temp);

  // Calculate beta_hat.
  logdet_O = CholeskySolve(Omega, Xty_temp, beta_hat);
  gsl_vector_scale(beta_hat, sigma_a2);

  d = vector_ddot(Xty_temp, beta_hat);
  P_yy -= d;

  // Sample tau.
  double tau = 1.0;
  if (a_mode == 11) {
    tau = gsl_ran_gamma(gsl_r, to!double(ni_test) / 2.0, 2.0 / P_yy);
  }

  // Sample beta.
  for (size_t i = 0; i < s_size; i++) {
    d = gsl_ran_gaussian(gsl_r, 1);
    beta.elements[i] = d;
  }
  //gsl_vector_view beta_sub = gsl_vector_subvector(beta, 0, s_size);
  gsl_blas_dtrsv(CblasUpper, CblasNoTrans, CblasNonUnit, Omega, &beta_sub.vector);

  // This computes inv(L^T(Omega)) %*% beta.
  beta_sub = multiply_dmatrix_num(beta_sub, sqrt(sigma_a2 / tau));
  beta_sub = add_dmatrix(beta_sub, beta_hat);
  //gsl_blas_dgemv(CblasNoTrans, 1.0, &Xgamma_sub.matrix, &beta_sub.vector, 0.0, Xb);

  // For quantitative traits, calculate pve and pge.
  if (a_mode == 11) {
    d = vector_ddot(Xb, Xb);
    cHyp.pve = d / to!double(ni_test);
    cHyp.pve /= cHyp.pve + 1.0 / tau;
    cHyp.pge = 1.0;
  }

  logpost = -0.5 * logdet_O;
  if (a_mode == 11) {
    logpost -= 0.5 * to!double(ni_test) * mlog(P_yy);
  } else {
    logpost -= 0.5 * P_yy;
  }

  logpost +=
      (to!double(cHyp.n_gamma) - 1.0) * cHyp.logp +
      (to!double(ns_test) - to!double(cHyp.n_gamma)) * mlog(1.0 - exp(cHyp.logp));

  return logpost;
}

// Calculate pve and pge, and calculate z_hat for case-control data.
void CalcCC_PVEnZ(DMatrix z_hat, HYPBSLMM cHyp) {
  z_hat = zeros_dmatrix(z_hat.shape[0], z_hat.shape[1]);
  cHyp.pve = 0.0;
  cHyp.pge = 1.0;
  return;
}

// Calculate pve and pge, and calculate z_hat for case-control data.
void CalcCC_PVEnZ(const DMatrix Xb, DMatrix z_hat, HYPBSLMM cHyp) {

  double d = vector_ddot(Xb, Xb);
  cHyp.pve = d / to!double(ni_test);
  cHyp.pve /= cHyp.pve + 1.0;
  cHyp.pge = 1.0;

  gsl_vector_memcpy(z_hat, Xb);

  return;
}


// Calculate pve and pge, and calculate z_hat for case-control data.
void CalcCC_PVEnZ(const DMatrix U, const DMatrix Utu, DMatrix z_hat, HYPBSLMM cHyp) {
  double d;

  d = vector_ddot(Utu, Utu);
  cHyp.pve = d / to!double(ni_test);

  z_hat = matrix_mult(U, Utu);

  cHyp.pve /= cHyp.pve + 1.0;
  cHyp.pge = 0.0;

  return;
}

// Calculate pve and pge, and calculate z_hat for case-control data.
void CalcCC_PVEnZ(const DMatrix U, const DMatrix UtXb,
                         const DMatrix Utu, DMatrix z_hat,
                         HYPBSLMM cHyp) {
  double d;
  DMatrix UtXbU; // = gsl_vector_alloc(Utu->size);

  d = vector_ddot(UtXb, UtXb);
  cHyp.pge = d / to!double(ni_test);

  gsl_blas_ddot(Utu, Utu, &d);
  cHyp.pve = cHyp.pge + d / to!double(ni_test);

  //gsl_vector_memcpy(UtXbU, Utu);
  UtXbU = add_dmatrix(UtXbU, UtXb);
  z_hat = matrix_mult(U, UtXbU);

  if (cHyp.pve == 0) {
    cHyp.pge = 0.0;
  } else {
    cHyp.pge /= cHyp.pve;
  }

  cHyp.pve /= cHyp.pve + 1.0;

  return;
}

void SampleZ(const DMatrix y, const DMatrix z_hat, DMatrix z) {
  double d1, d2, z_rand = 0.0;
  for (size_t i = 0; i < z.size; ++i) {
    d1 = y.elements[i];
    d2 = z_hat.elements[i];

    // y is centered for case control studies.
    if (d1 <= 0.0) {

      // Control, right truncated.
      do {
        z_rand = d2 + gsl_ran_gaussian(gsl_r, 1.0);
      } while (z_rand > 0.0);
    } else {
      do {
        z_rand = d2 + gsl_ran_gaussian(gsl_r, 1.0);
      } while (z_rand < 0.0);
    }

    z.elements[i] = z_rand;
  }

  return;
}

double ProposeHnRho(const HYPBSLMM cHyp_old, HYPBSLMM cHyp_new, const size_t repeat) {

  double h = cHyp_old.h, rho = cHyp_old.rho;

  double d_h = (h_max - h_min) * h_scale,
         d_rho = (rho_max - rho_min) * rho_scale;

  for (size_t i = 0; i < repeat; ++i) {
    h = h + (gsl_rng_uniform(gsl_r) - 0.5) * d_h;
    if (h < h_min) {
      h = 2 * h_min - h;
    }
    if (h > h_max) {
      h = 2 * h_max - h;
    }

    rho = rho + (gsl_rng_uniform(gsl_r) - 0.5) * d_rho;
    if (rho < rho_min) {
      rho = 2 * rho_min - rho;
    }
    if (rho > rho_max) {
      rho = 2 * rho_max - rho;
    }
  }
  cHyp_new.h = h;
  cHyp_new.rho = rho;
  return 0.0;
}

double ProposePi(const HYPBSLMM cHyp_old, HYPBSLMM cHyp_new, const size_t repeat) {
  double logp_old = cHyp_old.logp, logp_new = cHyp_old.logp;
  double log_ratio = 0.0;

  double d_logp = min(0.1, (logp_max - logp_min) * logp_scale);

  for (size_t i = 0; i < repeat; ++i) {
    logp_new = logp_old + (gsl_rng_uniform(gsl_r) - 0.5) * d_logp;
    if (logp_new < logp_min) {
      logp_new = 2 * logp_min - logp_new;
    }
    if (logp_new > logp_max) {
      logp_new = 2 * logp_max - logp_new;
    }
    log_ratio += logp_new - logp_old;
    logp_old = logp_new;
  }
  cHyp_new.logp = logp_new;

  return log_ratio;
}

bool comp_vec(size_t a, size_t b) { return (a < b); }

double ProposeGamma(const size_t[] rank_old, size_t[] rank_new, const double p_gamma,
                           const HYPBSLMM cHyp_old, HYPBSLMM cHyp_new, const size_t repeat) {
  size_t[int] mapRank2in;
  size_t r;
  double unif, logp = 0.0;
  int flag_gamma;
  size_t r_add, r_remove, col_id;

  rank_new = [];
  if (cHyp_old.n_gamma != rank_old.length) {
    writeln("size wrong");
  }

  if (cHyp_old.n_gamma != 0) {
    for (size_t i = 0; i < rank_old.length; ++i) {
      r = rank_old[i];
      rank_new ~= r;
      mapRank2in[r] = 1;
    }
  }
  cHyp_new.n_gamma = cHyp_old.n_gamma;

  for (size_t i = 0; i < repeat; ++i) {
    unif = gsl_rng_uniform(gsl_r);

    if (unif < 0.40 && cHyp_new.n_gamma < s_max) {
      flag_gamma = 1;
    } else if (unif >= 0.40 && unif < 0.80 && cHyp_new.n_gamma > s_min) {
      flag_gamma = 2;
    } else if (unif >= 0.80 && cHyp_new.n_gamma > 0 &&
               cHyp_new.n_gamma < ns_test) {
      flag_gamma = 3;
    } else {
      flag_gamma = 4;
    }

    if (flag_gamma == 1) {

      // Add a SNP.
      do {
        r_add = gsl_ran_discrete(gsl_r, gsl_t);
      } while (mapRank2in.count(r_add) != 0);

      double prob_total = 1.0;
      for (size_t i = 0; i < cHyp_new.n_gamma; ++i) {
        r = rank_new[i];
        prob_total -= p_gamma[r];
      }

      mapRank2in[r_add] = 1;
      rank_new ~= r_add;
      cHyp_new.n_gamma++;
      logp += -mlog(p_gamma[r_add] / prob_total) - mlog(to!double(cHyp_new.n_gamma));
    } else if (flag_gamma == 2) {

      // Delete a SNP.
      col_id = gsl_rng_uniform_int(gsl_r, cHyp_new.n_gamma);
      r_remove = rank_new[col_id];

      double prob_total = 1.0;
      for (size_t i = 0; i < cHyp_new.n_gamma; ++i) {
        r = rank_new[i];
        prob_total -= p_gamma[r];
      }
      prob_total += p_gamma[r_remove];

      mapRank2in.erase(r_remove);
      rank_new.erase(rank_new.begin() + col_id);
      logp += mlog(p_gamma[r_remove] / prob_total) + mlog(to!double(cHyp_new.n_gamma));
      cHyp_new.n_gamma--;
    } else if (flag_gamma == 3) {

      // Switch a SNP.
      col_id = gsl_rng_uniform_int(gsl_r, cHyp_new.n_gamma);
      r_remove = rank_new[col_id];

      // Be careful with the proposal.
      do {
        r_add = gsl_ran_discrete(gsl_r, gsl_t);
      } while (mapRank2in.count(r_add) != 0);

      double prob_total = 1.0;
      for (size_t i = 0; i < cHyp_new.n_gamma; ++i) {
        r = rank_new[i];
        prob_total -= p_gamma[r];
      }

      logp += mlog(p_gamma[r_remove] /
                  (prob_total + p_gamma[r_remove] - p_gamma[r_add]));
      logp -= mlog(p_gamma[r_add] / prob_total);

      mapRank2in.erase(r_remove);
      mapRank2in[r_add] = 1;
      rank_new.erase(rank_new.begin() + col_id);
      rank_new ~= r_add;
    } else {
      logp += 0;
    } // Do not change.
  }

  stable_sort(rank_new.begin(), rank_new.end(), comp_vec);

  mapRank2in = [];
  return logp;
}

bool comp_lr(pair2 a, pair2 b) {
  return (a.second > b.second);
}

// Below fits MCMC for rho=1.
void CalcXtX(const DMatrix X, const DMatrix y,
              const size_t s_size, DMatrix XtX, DMatrix Xty) {
  DMatrix X_sub = get_sub_dmatrix(X, 0, 0, X.shape[0], s_size);
  //gsl_matrix_view
  DMatrix XtX_sub = get_sub_dmatrix(XtX, 0, 0, s_size, s_size);
  //gsl_vector_view Xty_sub = gsl_vector_subvector(Xty, 0, s_size);

  XtX_sub = matrix_mult(X_sub.T, X_sub.matrix);
  Xty_sub = matrix_mult(X_sub.T, y);

  return;
}

void CalcPgamma(double[] p_gamma) {
  double p, s = 0.0;
  for (size_t i = 0; i < ns_test; ++i) {
    p = 0.7 * gsl_ran_geometric_pdf(i + 1, 1.0 / geo_mean) +
        0.3 / to!double(ns_test);
    p_gamma[i] = p;
    s += p;
  }
  for (size_t i = 0; i < ns_test; ++i) {
    p = p_gamma[i];
    p_gamma[i] = p / s;
  }
  return;
}

void SetXgamma(DMatrix Xgamma, const DMatrix X, size_t[] rank) {
  size_t pos;
  for (size_t i = 0; i < rank.length; ++i) {
    pos = mapRank2pos[rank[i]];
    //gsl_vector_view
    DMatrix Xgamma_col = get_col(Xgamma, i);
    //gsl_vector_const_view
    DMatrix X_col = get_col(X, pos);
    gsl_vector_memcpy(Xgamma_col, X_col);
  }

  return;
}

void SetXgamma(const DMatrix X, const DMatrix X_old,
                const DMatrix XtX_old, const DMatrix Xty_old,
                const DMatrix y, const size_t[] rank_old,
                const size_t[] rank_new, DMatrix X_new,
                DMatrix XtX_new, DMatrix Xty_new) {
  double d;

  // rank_old and rank_new are sorted already inside PorposeGamma
  // calculate vectors rank_remove and rank_add.
  // make sure that v_size is larger than repeat.
  size_t v_size = 20;
  size_t[] rank_remove = new size_t[v_size];
  size_t[] rank_add = new size_t[v_size];
  size_t[] rank_union = new size_t[s_max + v_size];
  //vector<size_t>::iterator it;

  it = set_difference(rank_old.begin(), rank_old.end(), rank_new.begin(),
                      rank_new.end(), rank_remove.begin());
  rank_remove.resize(it - rank_remove.begin());

  it = set_difference(rank_new.begin(), rank_new.end(), rank_old.begin(),
                      rank_old.end(), rank_add.begin());
  rank_add.resize(it - rank_add.begin());

  it = set_union(rank_new.begin(), rank_new.end(), rank_old.begin(),
                 rank_old.end(), rank_union.begin());
  rank_union.resize(it - rank_union.begin());

  // Map rank_remove and rank_add.
  pair2[] mapRank2in_remove, mapRank2in_add;
  for (size_t i = 0; i < rank_remove.length; i++) {
    mapRank2in_remove[rank_remove[i]] = 1;
  }
  for (size_t i = 0; i < rank_add.length; i++) {
    mapRank2in_add[rank_add[i]] = 1;
  }

  // Obtain the subset of matrix/vector.
  //gsl_matrix_const_view
  DMatrix Xold_sub = get_sub_dmatrix(X_old, 0, 0, X_old.shape[0], rank_old.length);
  //gsl_matrix_const_view
  DMatrix XtXold_sub = get_sub_dmatrix(XtX_old, 0, 0, rank_old.length, rank_old.length);
  //gsl_vector_const_view
  DMatrix Xtyold_sub = gsl_vector_const_subvector(Xty_old, 0, rank_old.length);

  //gsl_matrix_view
  DMatrix Xnew_sub = get_sub_dmatrix(X_new, 0, 0, X_new.shape[0], rank_new.length);
  //gsl_matrix_view
  DMatrix XtXnew_sub = get_sub_dmatrix(XtX_new, 0, 0, rank_new.length, rank_new.length);
  //gsl_vector_view
  DMatrix Xtynew_sub = gsl_vector_subvector(Xty_new, 0, rank_new.length);

  // Get X_new and calculate XtX_new.
  if (rank_remove.length == 0 && rank_add.length == 0) {
    gsl_matrix_memcpy(&Xnew_sub.matrix, &Xold_sub.matrix);
    gsl_matrix_memcpy(&XtXnew_sub.matrix, &XtXold_sub.matrix);
    gsl_vector_memcpy(&Xtynew_sub.vector, &Xtyold_sub.vector);
  } else {
    size_t i_old, j_old, i_new, j_new, i_add, j_add, i_flag, j_flag;
    if (rank_add.length == 0) {
      i_old = 0;
      i_new = 0;
      for (size_t i = 0; i < rank_union.length; i++) {
        if (mapRank2in_remove.count(rank_old[i_old]) != 0) {
          i_old++;
          continue;
        }

        //gsl_vector_view
        DMatrix Xnew_col = get_col(X_new, i_new);
        //gsl_vector_const_view
        DMatrix Xcopy_col = get_col(X_old, i_old);
        gsl_vector_memcpy(Xnew_col, Xcopy_col);

        d = Xty_old.elements[i_old];
        Xty_new.elements[i_new] = d;

        j_old = i_old;
        j_new = i_new;
        for (size_t j = i; j < rank_union.length; j++) {
          if (mapRank2in_remove.count(rank_old[j_old]) != 0) {
            j_old++;
            continue;
          }

          d = XtX_old.accessor(i_old, j_old);

          XtX_new.set(i_new, j_new, d);
          if (i_new != j_new) {
            XtX_new.set(j_new, i_new, d);
          }

          j_old++;
          j_new++;
        }
        i_old++;
        i_new++;
      }
    } else {
      DMatrix X_add; // = gsl_matrix_alloc(X_old->size1, rank_add.length);
      DMatrix XtX_aa; // = gsl_matrix_alloc(X_add->size2, X_add->size2);
      DMatrix XtX_ao; // = gsl_matrix_alloc(X_add->size2, X_old->size2);
      DMatrix Xty_add; // = gsl_vector_alloc(X_add->size2);

      // Get X_add.
      SetXgamma(X_add, X, rank_add);

      // Somehow the lapack_dgemm does not work here.
      XtX_aa = matrix_mult(X_add.T, X_add);
      XtX_ao = matrix_mult(X_add.T, X_old);
      Xty_add = matrix_mult(X_add.T, y);

      // Save to X_new, XtX_new and Xty_new.
      i_old = 0;
      i_new = 0;
      i_add = 0;
      for (size_t i = 0; i < rank_union.length; i++) {
        if (mapRank2in_remove.count(rank_old[i_old]) != 0) {
          i_old++;
          continue;
        }
        if (mapRank2in_add.count(rank_new[i_new]) != 0) {
          i_flag = 1;
        } else {
          i_flag = 0;
        }

        //gsl_vector_view
        DMatrix Xnew_col = get_col(X_new, i_new);
        if (i_flag == 1) {
          //gsl_vector_view
          DMatrix Xcopy_col = get_col(X_add, i_add);
          //gsl_vector_memcpy(&Xnew_col.vector, &Xcopy_col.vector);
        } else {
          //gsl_vector_const_view
          DMatrix Xcopy_col = get_col(X_old, i_old);
          //gsl_vector_memcpy(&Xnew_col.vector, &Xcopy_col.vector);
        }

        if (i_flag == 1) {
          d = Xty_add.elements[i_add];
        } else {
          d = Xty_old.elements[i_old];
        }
        Xty_new.elements[i_new] = d;

        j_old = i_old;
        j_new = i_new;
        j_add = i_add;
        for (size_t j = i; j < rank_union.length; j++) {
          if (mapRank2in_remove.count(rank_old[j_old]) != 0) {
            j_old++;
            continue;
          }
          if (mapRank2in_add.count(rank_new[j_new]) != 0) {
            j_flag = 1;
          } else {
            j_flag = 0;
          }

          if (i_flag == 1 && j_flag == 1) {
            d = XtX_aa.accessor(i_add, j_add);
          } else if (i_flag == 1) {
            d = XtX_ao.accessor(i_add, j_old);
          } else if (j_flag == 1) {
            d = XtX_ao.accessor(j_add, i_old);
          } else {
            d = XtX_old.accessor(i_old, j_old);
          }

          XtX_new.set(i_new, j_new, d);
          if (i_new != j_new) {
            XtX_new.set(j_new, i_new, d);
          }

          j_new++;
          if (j_flag == 1) {
            j_add++;
          } else {
            j_old++;
          }
        }
        i_new++;
        if (i_flag == 1) {
          i_add++;
        } else {
          i_old++;
        }
      }

    }
  }

  return;
}

void WriteResult(const int flag, const DMatrix Result_hyp,
                  const DMatrix Result_gamma, const size_t w_col) {
  string file_gamma, file_hyp;
  file_gamma = path_out + "/" + file_out;
  file_gamma += ".gamma.txt";
  file_hyp = path_out + "/" + file_out;
  file_hyp += ".hyp.txt";

  ofstream outfile_gamma, outfile_hyp;

  if (flag == 0) {
    //outfile_gamma.open(file_gamma.c_str(), ofstream);
    //outfile_hyp.open(file_hyp.c_str(), ofstream);
    if (!outfile_gamma) {
      writeln("error writing file: ", file_gamma);
      return;
    }
    if (!outfile_hyp) {
      writeln("error writing file: ", file_hyp);
      return;
    }

    //outfile_hyp += "h \t pve \t rho \t pge \t pi \t n_gamma";

    for (size_t i = 0; i < s_max; ++i) {
      //outfile_gamma += "s" + i +"\t";
    }
    outfile_gamma << endl;
  } else {
    //outfile_gamma.open(file_gamma.c_str(), ofstream::app);
    //outfile_hyp.open(file_hyp.c_str(), ofstream::app);
    if (!outfile_gamma) {
      writeln("error writing file: ", file_gamma);
      return;
    }
    if (!outfile_hyp) {
      writeln("error writing file: ", file_hyp);
      return;
    }

    size_t w;
    if (w_col == 0) {
      w = w_pace;
    } else {
      w = w_col;
    }

    for (size_t i = 0; i < w; ++i) {
      //outfile_hyp += scientific;
      for (size_t j = 0; j < 4; ++j) {
        //outfile_hyp += setprecision(6) , Result_hyp.accessor(i, j) + "\t";
      }
      //outfile_hyp += setprecision(6) << exp(Result_hyp.accessor(i, 4)) + "\t";
      //outfile_hyp += to!int(Result_hyp.accessor(i, 5)) + "\t";
    }

    for (size_t i = 0; i < w; ++i) {
      for (size_t j = 0; j < s_max; ++j) {
        //outfile_gamma += to!int(Result_gamma.accessor(i, j)) + "\t";
      }
    }
  }

  return;
}

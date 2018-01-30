/*
   This code is part of faster_lmm_d and published under the GPLv3
   License (see LICENSE.txt)

   Copyright © 2017-2018 Prasun Anand & Pjotr Prins
*/

module faster_lmm_d.mvlmm;

import core.stdc.stdlib : exit;

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

import faster_lmm_d.dmatrix;
import faster_lmm_d.gemma_lmm;
import faster_lmm_d.gemma_param;
import faster_lmm_d.helpers;
import faster_lmm_d.optmatrix;

import gsl.permutation;

void analyze_bimbam_mvlmm(Param cPar){
  //const gsl_matrix *U, const gsl_vector *eval,
                          //const gsl_matrix *UtW, const gsl_matrix *UtY) {

  DMatrix U, eval, UtW, UtY;
  string filename = cPar.file_geno;
  auto pipe = pipeShell("gunzip -c " ~ filename);
  File input = pipe.stdout;

  string line;
  char *ch_ptr;

  double logl_H0 = 0.0, logl_H1 = 0.0, p_wald = 0, p_lrt = 0, p_score = 0;
  double crt_a, crt_b, crt_c;
  int n_miss, c_phen;
  double geno, x_mean;
  size_t c = 0;
  size_t n_size = UtY.shape[0], d_size = UtY.shape[1], c_size = UtW.shape[1];

  size_t dc_size = d_size * (c_size + 1), v_size = d_size * (d_size + 1) / 2;

  // Create a large matrix.
  size_t LMM_BATCH_SIZE = 2000;
  size_t msize = LMM_BATCH_SIZE;
  DMatrix Xlarge = zeros_dmatrix(U.shape[0], msize);
  DMatrix UtXlarge; // = gsl_matrix_alloc(U.shape[0], msize);
  //gsl_matrix_set_zero(Xlarge);

  // Large matrices for EM.
  DMatrix U_hat;    //= gsl_matrix_alloc(d_size, n_size);
  DMatrix E_hat;    //= gsl_matrix_alloc(d_size, n_size);
  DMatrix OmegaU;   //= gsl_matrix_alloc(d_size, n_size);
  DMatrix OmegaE;   //= gsl_matrix_alloc(d_size, n_size);
  DMatrix UltVehiY;   //= gsl_matrix_alloc(d_size, n_size);
  DMatrix UltVehiBX;    //= gsl_matrix_alloc(d_size, n_size);
  DMatrix UltVehiU;   //= gsl_matrix_alloc(d_size, n_size);
  DMatrix UltVehiE;   //= gsl_matrix_alloc(d_size, n_size);

  // Large matrices for NR.
  // Each dxd block is H_k^{-1}.
  DMatrix Hi_all;   //= gsl_matrix_alloc(d_size, d_size * n_size);

  // Each column is H_k^{-1}y_k.
  DMatrix Hiy_all;    //= gsl_matrix_alloc(d_size, n_size);

  // Each dcxdc block is x_k \otimes H_k^{-1}.
  DMatrix xHi_all;    //= gsl_matrix_alloc(dc_size, d_size * n_size);
  DMatrix Hessian;    //= gsl_matrix_alloc(v_size * 2, v_size * 2);

  DMatrix x;    // = gsl_vector_alloc(n_size);
  DMatrix x_miss;   // = gsl_vector_alloc(n_size);

  DMatrix Y;    //= gsl_matrix_alloc(d_size, n_size);
  DMatrix X;    //= gsl_matrix_alloc(c_size + 1, n_size);
  DMatrix V_g;    //= gsl_matrix_alloc(d_size, d_size);
  DMatrix V_e;    //= gsl_matrix_alloc(d_size, d_size);
  DMatrix B;    //= gsl_matrix_alloc(d_size, c_size + 1);
  DMatrix beta;   // = gsl_vector_alloc(d_size);
  DMatrix Vbeta;    //= gsl_matrix_alloc(d_size, d_size);

  // Null estimates for initial values.
  DMatrix V_g_null;   //= gsl_matrix_alloc(d_size, d_size);
  DMatrix V_e_null;   //= gsl_matrix_alloc(d_size, d_size);
  DMatrix B_null;   //= gsl_matrix_alloc(d_size, c_size + 1);
  DMatrix se_B_null;    //= gsl_matrix_alloc(d_size, c_size);

  //gsl_matrix_view
  DMatrix X_sub = get_sub_dmatrix(X, 0, 0, c_size, n_size);
  //gsl_matrix_view
  DMatrix B_sub = get_sub_dmatrix(B, 0, 0, d_size, c_size);

  //gsl_matrix_view
  DMatrix xHi_all_sub = get_sub_dmatrix(xHi_all, 0, 0, d_size * c_size, d_size * n_size);

  //gsl_matrix_transpose_memcpy(Y, UtY);

  Y = UtY.T;

  //gsl_matrix_transpose_memcpy(&X_sub.matrix, UtW);
  X_sub = UtW.T;

  //gsl_vector_view
  DMatrix X_row = get_row(X, c_size);
  //gsl_vector_set_zero(&X_row.vector);

  //gsl_vector_view
  DMatrix B_col = get_col(B, c_size);
  //gsl_vector_set_zero(&B_col.vector);

  MphInitial(em_iter, em_prec, nr_iter, nr_prec, eval, X_sub, Y, l_min,
             l_max, n_region, V_g, V_e, B_sub);
  logl_H0 = MphEM('R', em_iter, em_prec, eval, X_sub, Y, U_hat, E_hat, OmegaU,
                    OmegaE, UltVehiY, UltVehiBX, UltVehiU, UltVehiE, V_g, V_e, B_sub);
  logl_H0 = MphNR('R', nr_iter, nr_prec, eval, X_sub, Y, Hi_all, xHi_all_sub, Hiy_all, V_g, V_e, Hessian, crt_a, crt_b, crt_c);
  MphCalcBeta(eval, X_sub, Y, V_g, V_e, UltVehiY, B_sub, se_B_null);

  c = 0;
  Vg_remle_null.clear();
  Ve_remle_null.clear();
  for (size_t i = 0; i < d_size; i++) {
    for (size_t j = i; j < d_size; j++) {
      Vg_remle_null ~= V_g.accessor(i, j);
      Ve_remle_null ~= V_e.accessor(i, j);
      //cpar params
      //VVg_remle_null ~= Hessian.accessor(c, c);
      //VVe_remle_null ~= Hessian.accessor(c + v_size, c + v_size);
      c++;
    }
  }
  beta_remle_null.clear();
  se_beta_remle_null.clear();
  for (size_t i = 0; i < se_B_null.shape[0]; i++) {
    for (size_t j = 0; j < se_B_null.shape[1]; j++) {
      beta_remle_null ~= B.accessor(i, j);
      se_beta_remle_null ~= se_B_null.accessor(i, j);
    }
  }
  logl_remle_H0 = logl_H0;

  writeln("REMLE estimate for Vg in the null model: ");
  for (size_t i = 0; i < d_size; i++) {
    for (size_t j = 0; j <= i; j++) {
      write(V_g.accessor(i, j), "\t");
    }
    write("\n");
  }

  writeln("se(Vg): ");
  for (size_t i = 0; i < d_size; i++) {
    for (size_t j = 0; j <= i; j++) {
      c = GetIndex(i, j, d_size);
      write(Hessian.accessor(c, c), "\t");
    }
    write("\n");
  }
  writeln("REMLE estimate for Ve in the null model: ");
  for (size_t i = 0; i < d_size; i++) {
    for (size_t j = 0; j <= i; j++) {
      write(V_e.accessor(i, j), "\t");
    }
    write("\n");
  }
  writeln("se(Ve): ");
  for (size_t i = 0; i < d_size; i++) {
    for (size_t j = 0; j <= i; j++) {
      c = GetIndex(i, j, d_size);
      write(sqrt(Hessian.accessor(c + v_size, c + v_size)), "\t");
    }
    write("\n");
  }
  writeln("REMLE likelihood = ", logl_H0);

  logl_H0 = MphEM('L', em_iter, em_prec, eval, &X_sub.matrix, Y, U_hat, E_hat,
                  OmegaU, OmegaE, UltVehiY, UltVehiBX, UltVehiU, UltVehiE, V_g,
                  V_e, &B_sub.matrix);
  logl_H0 = MphNR('L', nr_iter, nr_prec, eval, &X_sub.matrix, Y, Hi_all,
                  &xHi_all_sub.matrix, Hiy_all, V_g, V_e, Hessian, crt_a, crt_b,
                  crt_c);
  MphCalcBeta(eval, &X_sub.matrix, Y, V_g, V_e, UltVehiY, &B_sub.matrix,
              se_B_null);

  c = 0;
  Vg_mle_null.clear();
  Ve_mle_null.clear();
  for (size_t i = 0; i < d_size; i++) {
    for (size_t j = i; j < d_size; j++) {
      Vg_mle_null ~= V_g.accessor(i, j);
      Ve_mle_null ~= V_e.accessor(i, j);
      VVg_mle_null ~= Hessian.accessor(c, c);
      VVe_mle_null ~= Hessian.accessor(c + v_size, c + v_size);
      c++;
    }
  }
  beta_mle_null.clear();
  se_beta_mle_null.clear();
  for (size_t i = 0; i < se_B_null.shape[0]; i++) {
    for (size_t j = 0; j < se_B_null.shape[1]; j++) {
      beta_mle_null ~= B.accessor(i, j);
      se_beta_mle_null ~= se_B_null.accessor(i, j);
    }
  }
  logl_mle_H0 = logl_H0;

  writeln("MLE estimate for Vg in the null model: ");
  for (size_t i = 0; i < d_size; i++) {
    for (size_t j = 0; j <= i; j++) {
      write(V_g.accessor(i, j), "\t");
    }
    write("\n");
  }
  writeln("se(Vg): ");
  for (size_t i = 0; i < d_size; i++) {
    for (size_t j = 0; j <= i; j++) {
      c = GetIndex(i, j, d_size);
      write(sqrt(Hessian.accessor(c, c)), "\t");
    }
    write("\n");
  }
  writeln("MLE estimate for Ve in the null model: ");
  for (size_t i = 0; i < d_size; i++) {
    for (size_t j = 0; j <= i; j++) {
      write(V_e.accessor(i, j), "\t");
    }
    write("\n");
  }
  writeln("se(Ve): ");
  for (size_t i = 0; i < d_size; i++) {
    for (size_t j = 0; j <= i; j++) {
      c = GetIndex(i, j, d_size);
      write(sqrt(Hessian.accessor(c + v_size, c + v_size)), "\t");
    }
    write("\n");
  }
  writeln("MLE likelihood = ", logl_H0);
  double[] v_beta, v_Vg, v_Ve, v_Vbeta;
  for (size_t i = 0; i < d_size; i++) {
    v_beta ~= 0;
  }
  for (size_t i = 0; i < d_size; i++) {
    for (size_t j = i; j < d_size; j++) {
      v_Vg ~= 0;
      v_Ve ~= 0;
      v_Vbeta ~= 0;
    }
  }

  V_g_null = V_g;
  V_e_null = V_e;
  B_null = B;

  // Start reading genotypes and analyze.
  size_t csnp = 0, t_last = 0;
  for (size_t t = 0; t < indicator_snp.size(); ++t) {
    if (indicator_snp[t] == 0) {
      continue;
    }
    t_last++;
  }
  for (size_t t = 0; t < indicator_snp.size(); ++t) {
    safeGetline(infile, line).eof();
    if (t % d_pace == 0 || t == (ns_total - 1)) {
      ProgressBar("Reading SNPs", t, ns_total - 1);
    }
    if (indicator_snp[t] == 0) {
      continue;
    }

    //ch_ptr = strtok_safe((char *)line.c_str(), " , \t");
    //ch_ptr = strtok_safe(NULL, " , \t");
    //ch_ptr = strtok_safe(NULL, " , \t");

    x_mean = 0.0;
    c_phen = 0;
    n_miss = 0;
    gsl_vector_set_zero(x_miss);
    for (size_t i = 0; i < ni_total; ++i) {
      ch_ptr = strtok_safe(NULL, " , \t");
      if (indicator_idv[i] == 0) {
        continue;
      }

      if (strcmp(ch_ptr, "NA") == 0) {
        gsl_vector_set(x_miss, c_phen, 0.0);
        n_miss++;
      } else {
        geno = atof(ch_ptr);

        x[c_phen] = geno;
        x_miss[c_phen] = 1.0;
        x_mean += geno;
      }
      c_phen++;
    }

    x_mean /= to!double(ni_test - n_miss);

    for (size_t i = 0; i < ni_test; ++i) {
      if (gsl_vector_get(x_miss, i) == 0) {
        x[i] = x_mean;
      }
      geno = x[i];
    }

    DMatrix Xlarge_col = get_col(Xlarge, csnp % msize);
    //gsl_vector_memcpy(&Xlarge_col.vector, x);
    csnp++;

    if (csnp % msize == 0 || csnp == t_last) {
      size_t l = 0;
      if (csnp % msize == 0) {
        l = msize;
      } else {
        l = csnp % msize;
      }

      //gsl_matrix_view
      DMatrix Xlarge_sub =
          get_sub_dmatrix(Xlarge, 0, 0, Xlarge.shape[0], l);
      //gsl_matrix_view
      DMatrix UtXlarge_sub =
          get_sub_dmatrix(UtXlarge, 0, 0, UtXlarge.shape[0], l);

      fast_dgemm("T", "N", 1.0, U, &Xlarge_sub.matrix, 0.0,
                 &UtXlarge_sub.matrix);

      //gsl_matrix_set_zero(Xlarge);
      Xlarge = zeros_dmatrix(Xlarge.shape[0], Xlarge.shape[1]);

      for (size_t i = 0; i < l; i++) {
        //gsl_vector_view
        DMatrix UtXlarge_col = get_col(UtXlarge, i);
        //gsl_vector_memcpy(&X_row.vector, &UtXlarge_col.vector);

        // Initial values.
        V_g = V_g_null;
        V_e = V_e_null;
        B = B_null;

        // 3 is before 1.
        if (a_mode == 3 || a_mode == 4) {
          p_score = MphCalcP(eval, &X_row.vector, &X_sub.matrix, Y, V_g_null,
                             V_e_null, UltVehiY, beta, Vbeta);
          if (p_score < p_nr && crt == 1) {
            logl_H1 = MphNR('R', 1, nr_prec * 10, eval, X, Y, Hi_all, xHi_all,
                            Hiy_all, V_g, V_e, Hessian, crt_a, crt_b, crt_c);
            p_score = PCRT(3, d_size, p_score, crt_a, crt_b, crt_c);
          }
        }

        if (a_mode == 2 || a_mode == 4) {
          logl_H1 = MphEM('L', em_iter / 10, em_prec * 10, eval, X, Y, U_hat,
                          E_hat, OmegaU, OmegaE, UltVehiY, UltVehiBX, UltVehiU,
                          UltVehiE, V_g, V_e, B);

          // Calculate beta and Vbeta.
          p_lrt = MphCalcP(eval, &X_row.vector, &X_sub.matrix, Y, V_g, V_e,
                           UltVehiY, beta, Vbeta);
          p_lrt = gsl_cdf_chisq_Q(2.0 * (logl_H1 - logl_H0), to!double(d_size));

          if (p_lrt < p_nr) {
            logl_H1 =
                MphNR('L', nr_iter / 10, nr_prec * 10, eval, X, Y, Hi_all,
                      xHi_all, Hiy_all, V_g, V_e, Hessian, crt_a, crt_b, crt_c);

            // Calculate beta and Vbeta.
            p_lrt = MphCalcP(eval, &X_row.vector, &X_sub.matrix, Y, V_g, V_e,
                             UltVehiY, beta, Vbeta);
            p_lrt = gsl_cdf_chisq_Q(2.0 * (logl_H1 - logl_H0), to!double(d_size));

            if (crt == 1) {
              p_lrt = PCRT(2, d_size, p_lrt, crt_a, crt_b, crt_c);
            }
          }
        }

        if (a_mode == 1 || a_mode == 4) {
          logl_H1 = MphEM('R', em_iter / 10, em_prec * 10, eval, X, Y, U_hat,
                          E_hat, OmegaU, OmegaE, UltVehiY, UltVehiBX, UltVehiU,
                          UltVehiE, V_g, V_e, B);
          p_wald = MphCalcP(eval, &X_row.vector, &X_sub.matrix, Y, V_g, V_e,
                            UltVehiY, beta, Vbeta);

          if (p_wald < p_nr) {
            logl_H1 =
                MphNR('R', nr_iter / 10, nr_prec * 10, eval, X, Y, Hi_all,
                      xHi_all, Hiy_all, V_g, V_e, Hessian, crt_a, crt_b, crt_c);
            p_wald = MphCalcP(eval, &X_row.vector, &X_sub.matrix, Y, V_g, V_e,
                              UltVehiY, beta, Vbeta);

            if (crt == 1) {
              p_wald = PCRT(1, d_size, p_wald, crt_a, crt_b, crt_c);
            }
          }
        }

        // Store summary data.
        for (size_t i = 0; i < d_size; i++) {
          v_beta[i] = beta[i];
        }

        c = 0;
        for (size_t i = 0; i < d_size; i++) {
          for (size_t j = i; j < d_size; j++) {
            v_Vg[c] = V_g.accessor(i, j);
            v_Ve[c] = V_e.accessor(i, j);
            v_Vbeta[c] = Vbeta.accessor(i, j);
            c++;
          }
        }

        MPHSUMSTAT SNPs = {v_beta, p_wald, p_lrt, p_score, v_Vg, v_Ve, v_Vbeta};
        sumStat ~= SNPs;
      }
    }
  }
  return;
}

// Initialize Vg, Ve and B.
void MphInitial(){
  //const size_t em_iter, const double em_prec,
  //              const size_t nr_iter, const double nr_prec,
  //              const gsl_vector *eval, const gsl_matrix *X,
  //              const gsl_matrix *Y, const double l_min, const double l_max,
  //              const size_t n_region, gsl_matrix *V_g, gsl_matrix *V_e,
  //              gsl_matrix *B) {

  V_g = zeros_dmatrix(V_g.shape[0], V_g.shape[1]);
  V_e = zeros_dmatrix(V_e.shape[0], V_e.shape[1]);
  B   = zeros_dmatrix(B.shape[0], B.shape[1]);

  size_t n_size = eval.length, c_size = X.shape[0], d_size = Y.shape[0];
  double a, b, c;
  double lambda, logl, vg, ve;

  // Initialize the diagonal elements of Vg and Ve using univariate
  // LMM and REML estimates.
  DMatrix Xt; //= gsl_matrix_alloc(n_size, c_size);
  DMatrix beta_temp; // = gsl_vector_alloc(c_size);
  DMatrix se_beta_temp; // = gsl_vector_alloc(c_size);

  Xt = X.T;

  for (size_t i = 0; i < d_size; i++) {
    //gsl_vector_const_view
    DMatrix Y_row = get_row(Y, i);
    CalcLambda('R', eval, Xt, &Y_row.vector, l_min, l_max, n_region, lambda,
               logl);
    CalcLmmVgVeBeta(eval, Xt, &Y_row.vector, lambda, vg, ve, beta_temp,
                    se_beta_temp);

    V_g.set(i, i) = vg;
    V_e.set(i, i) = ve;
  }

  // If number of phenotypes is above four, then obtain the off
  // diagonal elements with two trait models.
  if (d_size > 4) {

    // First obtain good initial values.
    // Large matrices for EM.
    DMatrix U_hat; // = gsl_matrix_alloc(2, n_size);
    DMatrix E_hat; // = gsl_matrix_alloc(2, n_size);
    DMatrix OmegaU; // = gsl_matrix_alloc(2, n_size);
    DMatrix OmegaE; // = gsl_matrix_alloc(2, n_size);
    DMatrix UltVehiY; // = gsl_matrix_alloc(2, n_size);
    DMatrix UltVehiBX; // = gsl_matrix_alloc(2, n_size);
    DMatrix UltVehiU; // = gsl_matrix_alloc(2, n_size);
    DMatrix UltVehiE; // = gsl_matrix_alloc(2, n_size);

    // Large matrices for NR. Each dxd block is H_k^{-1}.
    DMatrix Hi_all; // = gsl_matrix_alloc(2, 2 * n_size);

    // Each column is H_k^{-1}y_k.
    DMatrix Hiy_all; // = gsl_matrix_alloc(2, n_size);

    // Each dcxdc block is x_k\otimes H_k^{-1}.
    DMatrix xHi_all; // = gsl_matrix_alloc(2 * c_size, 2 * n_size);
    DMatrix Hessian; // = gsl_matrix_alloc(6, 6);

    // 2 by n matrix of Y.
    DMatrix Y_sub; // = gsl_matrix_alloc(2, n_size);
    DMatrix Vg_sub; // = gsl_matrix_alloc(2, 2);
    DMatrix Ve_sub; // = gsl_matrix_alloc(2, 2);
    DMatrix B_sub; // = gsl_matrix_alloc(2, c_size);

    for (size_t i = 0; i < d_size; i++) {
      //gsl_vector_view
      DMatrix Y_sub1 = get_row(Y_sub, 0);
      //gsl_vector_const_view
      DMatrix Y_1 = ger_row(Y, i);
      Y_sub1 = Y_1;

      for (size_t j = i + 1; j < d_size; j++) {
        //gsl_vector_view
        DMatrix Y_sub2 = get_row(Y_sub, 1);
        //gsl_vector_const_view
        DMatrix Y_2 = get_row(Y, j);
        Y_sub2 = Y_2;

        Vg_sub = zeros_dmatrix(Vg_sub.shape[0], Vg_sub.shape[1]);
        //gsl_matrix_set_zero(Ve_sub);
        VE_sub = zeros_dmatrix(Ve_sub.shape[0], Ve_sub.shape[1]);
        Vg_sub.set(0, 0) = V_g.accessor(i, i);
        Ve_sub.set(0, 0) = V_e.accessor(i, i);
        Vg_sub.set(1, 1) = V_g.accessor(j, j);
        Ve_sub.set(1, 1) = V_e.accessor(j, j);

        logl = MphEM('R', em_iter, em_prec, eval, X, Y_sub, U_hat, E_hat,
                     OmegaU, OmegaE, UltVehiY, UltVehiBX, UltVehiU, UltVehiE,
                     Vg_sub, Ve_sub, B_sub);
        logl = MphNR('R', nr_iter, nr_prec, eval, X, Y_sub, Hi_all, xHi_all,
                     Hiy_all, Vg_sub, Ve_sub, Hessian, a, b, c);

        V_g.set(i, j) = Vg_sub.accessor(0, 1);
        V_g.set(j, i) = Vg_sub.accessor(0, 1);

        V_e.set(i, j) = Ve_sub.accessor(0, 1);
        V_e.set(j, i) = Ve_sub.accessor(0, 1);
      }
    }
  }

  // Calculate B hat using GSL estimate.
  DMatrix UltVehiY; // = gsl_matrix_alloc(d_size, n_size);

  DMatrix D_l; // = gsl_vector_alloc(d_size);
  DMatrix UltVeh; // = gsl_matrix_alloc(d_size, d_size);
  DMatrix UltVehi; // = gsl_matrix_alloc(d_size, d_size);
  DMatrix Qi; // = gsl_matrix_alloc(d_size * c_size, d_size * c_size);
  DMatrix XHiy; // = gsl_vector_alloc(d_size * c_size);
  DMatrix beta; // = gsl_vector_alloc(d_size * c_size);

  XHiy = zeros_dmatrix(XHiy.shape[0], XHiy.shape[1]);

  double dl, d, delta, dx, dy;

  // Eigen decomposition and calculate log|Ve|.
  // double logdet_Ve = EigenProc(V_g, V_e, D_l, UltVeh, UltVehi);
  EigenProc(V_g, V_e, D_l, UltVeh, UltVehi);

  // Calculate Qi and log|Q|.
  // double logdet_Q = CalcQi(eval, D_l, X, Qi);
  CalcQi(eval, D_l, X, Qi);

  // Calculate UltVehiY.
  UltVehiY =  matrix_mult(UltVehi, Y).elements[0];

  // calculate XHiy
  for (size_t i = 0; i < d_size; i++) {
    dl = D_l.elements[i];

    for (size_t j = 0; j < c_size; j++) {
      d = 0.0;
      for (size_t k = 0; k < n_size; k++) {
        delta = eval.elements[k];
        dx = X.accessor(j, k);
        dy = UltVehiY.accessor(i, k);
        d += dy * dx / (delta * dl + 1.0);
      }
      XHiy.elements[j * d_size + i] = d;
    }
  }

  gsl_blas_dgemv(CblasNoTrans, 1.0, Qi, XHiy, 0.0, beta);

  // Multiply beta by UltVeh and save to B.
  for (size_t i = 0; i < c_size; i++) {
    //gsl_vector_view
    DMatrix B_col = get_col(B, i);
    //gsl_vector_view
    DMatrix beta_sub = gsl_vector_subvector(beta, i * d_size, d_size);
    //gsl_blas_dgemv(CblasTrans, 1.0, UltVeh, &beta_sub.vector, 0.0, &B_col.vector);
    B_col = matrix_mult(beta_sub, UltVeh);
  }

  // Free memory.

  return;
}

size_t GetIndex(const size_t i, const size_t j, const size_t d_size) {
  if (i >= d_size || j >= d_size) {
    writeln("error in GetIndex.");
    return 0;
  }

  size_t s, l;
  if (j < i) {
    s = j;
    l = i;
  } else {
    s = i;
    l = j;
  }
  return (2 * d_size - s + 1) * s / 2 + l - s;
}

double MphEM(){
             //const char func_name, const size_t max_iter, const double max_prec,
             //const gsl_vector *eval, const gsl_matrix *X, const gsl_matrix *Y,
             //gsl_matrix *U_hat, gsl_matrix *E_hat, gsl_matrix *OmegaU,
             //gsl_matrix *OmegaE, gsl_matrix *UltVehiY, gsl_matrix *UltVehiBX,
             //gsl_matrix *UltVehiU, gsl_matrix *UltVehiE, gsl_matrix *V_g,
             //gsl_matrix *V_e, gsl_matrix *B) {
  if (func_name != 'R' && func_name != 'L' && func_name != 'r' && func_name != 'l') {
    writeln("func_name only takes 'R' or 'L': 'R' for log-restricted likelihood, 'L' for log-likelihood.");
    return 0.0;
  }

  size_t n_size = eval.length, c_size = X.shape[0], d_size = Y.shape[0];
  size_t dc_size = d_size * c_size;

  DMatrix XXt; // = gsl_matrix_alloc(c_size, c_size);
  DMatrix XXti; // = gsl_matrix_alloc(c_size, c_size);
  DMatrix D_l; // = gsl_vector_alloc(d_size);
  DMatrix UltVeh; // = gsl_matrix_alloc(d_size, d_size);
  DMatrix UltVehi; // = gsl_matrix_alloc(d_size, d_size);
  DMatrix UltVehiB; // = gsl_matrix_alloc(d_size, c_size);
  DMatrix Qi; // = gsl_matrix_alloc(dc_size, dc_size);
  DMatrix Sigma_uu; // = gsl_matrix_alloc(d_size, d_size);
  DMatrix Sigma_ee; // = gsl_matrix_alloc(d_size, d_size);
  DMatrix xHiy; // = gsl_vector_alloc(dc_size);
  gsl_permutation pmt = gsl_permutation_alloc(c_size);

  double logl_const = 0.0, logl_old = 0.0, logl_new = 0.0;
  double logdet_Q, logdet_Ve;
  int sig;

  // Calculate |XXt| and (XXt)^{-1}.
  gsl_blas_dsyrk(CblasUpper, CblasNoTrans, 1.0, X, 0.0, XXt);
  for (size_t i = 0; i < c_size; ++i) {
    for (size_t j = 0; j < i; ++j) {
      XXt.set(i, j) = XXt.accessor(j, i);
    }
  }

  LUDecomp(XXt, pmt, &sig);
  LUInvert(XXt, pmt, XXti);

  // Calculate the constant for logl.
  if (func_name == 'R' || func_name == 'r') {
    logl_const =
        -0.5 * to!double(n_size - c_size) * to!double(d_size) * log(2.0 * M_PI) +
        0.5 * to!double(d_size) * LULndet(XXt);
  } else {
    logl_const = -0.5 * to!double(n_size) * to!double(d_size) * log(2.0 * M_PI);
  }

  // Start EM.
  for (size_t t = 0; t < max_iter; t++) {
    logdet_Ve = EigenProc(V_g, V_e, D_l, UltVeh, UltVehi);

    logdet_Q = CalcQi(eval, D_l, X, Qi);

    UltVehiY = matrix_mult(UltVehi, Y);
    CalcXHiY(eval, D_l, X, UltVehiY, xHiy);

    // Calculate log likelihood/restricted likelihood value, and
    // terminate if change is small.
    logl_new = logl_const + MphCalcLogL(eval, xHiy, D_l, UltVehiY, Qi) -
               0.5 * to!double(n_size) * logdet_Ve;
    if (func_name == 'R' || func_name == 'r') {
      logl_new += -0.5 * (logdet_Q - to!double(c_size) * logdet_Ve);
    }
    if (t != 0 && abs(logl_new - logl_old) < max_prec) {
      break;
    }
    logl_old = logl_new;

    CalcOmega(eval, D_l, OmegaU, OmegaE);

    // Update UltVehiB, UltVehiU.
    if (func_name == 'R' || func_name == 'r') {
      UpdateRL_B(xHiy, Qi, UltVehiB);
      UltVehiBX = matrix_mult(UltVehiB, X);
    } else if (t == 0) {
      UltVehiB = matrix_mult(UltVehi, B);
      UltVehiBX = matrix_mult(UltVehiB, X);
    }

    UpdateU(OmegaE, UltVehiY, UltVehiBX, UltVehiU);

    if (func_name == 'L' || func_name == 'l') {

      // UltVehiBX is destroyed here.
      UpdateL_B(X, XXti, UltVehiY, UltVehiU, UltVehiBX, UltVehiB);
      UltVehiBX = matrix_mult(UltVehiB, X);
    }

    UpdateE(UltVehiY, UltVehiBX, UltVehiU, UltVehiE);

    // Calculate U_hat, E_hat and B.
    U_hat = matrix_mult(UltVeh, UltVehiU);
    E_hat = matrix_mult(UltVeh, UltVehiE);
    B = matrix_mult(UltVeh, UltVehiB);

    // Calculate Sigma_uu and Sigma_ee.
    CalcSigma(func_name, eval, D_l, X, OmegaU, OmegaE, UltVeh, Qi, Sigma_uu,
              Sigma_ee);

    // Update V_g and V_e.
    UpdateV(eval, U_hat, E_hat, Sigma_uu, Sigma_ee, V_g, V_e);
  }

  gsl_permutation_free(pmt);

  return logl_new;
}

double MphNR(){
             //const char func_name, const size_t max_iter, const double max_prec,
             //const gsl_vector *eval, const gsl_matrix *X, const gsl_matrix *Y,
             //gsl_matrix *Hi_all, gsl_matrix *xHi_all, gsl_matrix *Hiy_all,
             //gsl_matrix *V_g, gsl_matrix *V_e, gsl_matrix *Hessian_inv,
             //double &crt_a, double &crt_b, double &crt_c) {
  if (func_name != 'R' && func_name != 'L' && func_name != 'r' && func_name != 'l') {
    writeln("func_name only takes 'R' or 'L': 'R' for log-restricted likelihood, 'L' for log-likelihood.");
    return 0.0;
  }
  size_t n_size = eval.length, c_size = X.shape[0], d_size = Y.shape[0];
  size_t dc_size = d_size * c_size;
  size_t v_size = d_size * (d_size + 1) / 2;

  double logdet_H, logdet_Q, yPy, logl_const;
  double logl_old = 0.0, logl_new = 0.0, step_scale;
  int sig;
  size_t step_iter, flag_pd;

  DMatrix Vg_save; // = gsl_matrix_alloc(d_size, d_size);
  DMatrix Ve_save; // = gsl_matrix_alloc(d_size, d_size);
  DMatrix V_temp; // = gsl_matrix_alloc(d_size, d_size);
  DMatrix U_temp; // = gsl_matrix_alloc(d_size, d_size);
  DMatrix D_temp; // = gsl_vector_alloc(d_size);
  DMatrix xHiy; // = gsl_vector_alloc(dc_size);
  DMatrix QixHiy; // = gsl_vector_alloc(dc_size);
  DMatrix Qi; // = gsl_matrix_alloc(dc_size, dc_size);
  DMatrix XXt; // = gsl_matrix_alloc(c_size, c_size);

  DMatrix gradient; // = gsl_vector_alloc(v_size * 2);

  // Calculate |XXt| and (XXt)^{-1}.
  gsl_blas_dsyrk(CblasUpper, CblasNoTrans, 1.0, X, 0.0, XXt);
  for (size_t i = 0; i < c_size; ++i) {
    for (size_t j = 0; j < i; ++j) {
      XXt.set(i, j) = XXt.accessor(j, i);
    }
  }

  gsl_permutation pmt = gsl_permutation_alloc(c_size);
  LUDecomp(XXt, pmt, &sig);
  gsl_permutation_free(pmt);

  // Calculate the constant for logl.
  if (func_name == 'R' || func_name == 'r') {
    logl_const =
        -0.5 * to!double(n_size - c_size) * to!double(d_size) * log(2.0 * M_PI) +
        0.5 * to!double(d_size) * LULndet(XXt);
  } else {
    logl_const = -0.5 * to!double(n_size) * to!double(d_size) * log(2.0 * M_PI);
  }

  // Optimization iterations.
  for (size_t t = 0; t < max_iter; t++) {
    Vg_save = V_g; // Check dup
    Ve_save = V_e;

    step_scale = 1.0;
    step_iter = 0;
    do {
      V_g = Vg_save;
      V_e = Ve_save;

      // Update Vg, Ve, and invert Hessian.
      if (t != 0) {
        UpdateVgVe(Hessian_inv, gradient, step_scale, V_g, V_e);
      }

      // Check if both Vg and Ve are positive definite.
      flag_pd = 1;
      V_temp = V_e;
      EigenDecomp(V_temp, U_temp, D_temp, 0);
      for (size_t i = 0; i < d_size; i++) {
        if (gsl_vector_get(D_temp, i) <= 0) {
          flag_pd = 0;
        }
      }
      V_temp = V_g;
      EigenDecomp(V_temp, U_temp, D_temp, 0);
      for (size_t i = 0; i < d_size; i++) {
        if (gsl_vector_get(D_temp, i) <= 0) {
          flag_pd = 0;
        }
      }

      // If flag_pd==1, continue to calculate quantities
      // and logl.
      if (flag_pd == 1) {
        CalcHiQi(eval, X, V_g, V_e, Hi_all, Qi, logdet_H, logdet_Q);
        Calc_Hiy_all(Y, Hi_all, Hiy_all);
        Calc_xHi_all(X, Hi_all, xHi_all);

        // Calculate QixHiy and yPy.
        Calc_xHiy(Y, xHi_all, xHiy);
        gsl_blas_dgemv(CblasNoTrans, 1.0, Qi, xHiy, 0.0, QixHiy);

        gsl_blas_ddot(QixHiy, xHiy, &yPy);
        yPy = Calc_yHiy(Y, Hiy_all) - yPy;

        // Calculate log likelihood/restricted likelihood value.
        if (func_name == 'R' || func_name == 'r') {
          logl_new = logl_const - 0.5 * logdet_H - 0.5 * logdet_Q - 0.5 * yPy;
        } else {
          logl_new = logl_const - 0.5 * logdet_H - 0.5 * yPy;
        }
      }

      step_scale /= 2.0;
      step_iter++;

    } while (
        (flag_pd == 0 || logl_new < logl_old || logl_new - logl_old > 10) &&
        step_iter < 10 && t != 0);

    // Terminate if change is small.
    if (t != 0) {
      if (logl_new < logl_old || flag_pd == 0) {
        V_g = Vg_save;  // Check dup
        V_e = Ve_save;
        break;
      }

      if (logl_new - logl_old < max_prec) {
        break;
      }
    }

    logl_old = logl_new;

    CalcDev(func_name, eval, Qi, Hi_all, xHi_all, Hiy_all, QixHiy, gradient,
            Hessian_inv, crt_a, crt_b, crt_c);
  }

  // Mutiply Hessian_inv with -1.0.
  // Now Hessian_inv is the variance matrix.
  Hessian_inv = multiply_dmatrix_num(Hessian_inv, -1.0);

  return logl_new;
}

void MphCalcBeta(){
                 //const gsl_vector *eval, const gsl_matrix *W,
                 //const gsl_matrix *Y, const gsl_matrix *V_g,
                 //const gsl_matrix *V_e, gsl_matrix *UltVehiY, gsl_matrix *B,
                 //gsl_matrix *se_B) {
  size_t n_size = eval.length, c_size = W.shape[0], d_size = W.shape[1];
  size_t dc_size = d_size * c_size;
  double delta, dl, d, dy, dw; // , logdet_Ve, logdet_Q;

  DMatrix D_l; // = gsl_vector_alloc(d_size);
  DMatrix UltVeh; // = gsl_matrix_alloc(d_size, d_size);
  DMatrix UltVehi; // = gsl_matrix_alloc(d_size, d_size);
  DMatrix Qi; // = gsl_matrix_alloc(dc_size, dc_size);
  DMatrix Qi_temp; // = gsl_matrix_alloc(dc_size, dc_size);
  DMatrix WHiy; // = gsl_vector_alloc(dc_size);
  DMatrix QiWHiy; // = gsl_vector_alloc(dc_size);
  DMatrix beta; // = gsl_vector_alloc(dc_size);
  DMatrix Vbeta; // = gsl_matrix_alloc(dc_size, dc_size);

  WHiy = zeros_dmatrix(WHiy.shape[0], WHiy.shape[1]);

  // Eigen decomposition and calculate log|Ve|.
  // double logdet_Ve = EigenProc(V_g, V_e, D_l, UltVeh, UltVehi);
  EigenProc(V_g, V_e, D_l, UltVeh, UltVehi);

  // Calculate Qi and log|Q|.
  // double logdet_Q = CalcQi(eval, D_l, W, Qi);
  CalcQi(eval, D_l, W, Qi);

  // Calculate UltVehiY.
  UltVehiY = matrix_mult(UltVehi, Y);

  // Calculate WHiy.
  for (size_t i = 0; i < d_size; i++) {
    dl = D_l.accessor(i);

    for (size_t j = 0; j < c_size; j++) {
      d = 0.0;
      for (size_t k = 0; k < n_size; k++) {
        delta = eval.elements[k];
        dw = W.accessor(j, k);
        dy = UltVehiY.accessor(i, k);

        d += dy * dw / (delta * dl + 1.0);
      }
      WHiy.elements[j * d_size + i] = d;
    }
  }

  QiWHiy = matrix_mult(Qi, WHiy);

  // Need to multiply I_c\otimes UltVehi on both sides or one side.
  for (size_t i = 0; i < c_size; i++) {
    //gsl_vector_view
    DMatrix QiWHiy_sub ;//= gsl_vector_subvector(QiWHiy, i * d_size, d_size);
    //gsl_vector_view
    DMatrix beta_sub;// = gsl_vector_subvector(beta, i * d_size, d_size);
    beta_sub = matrix_mult(UltVeh, QiWHiy_sub);

    for (size_t j = 0; j < c_size; j++) {
      //gsl_matrix_view
      DMatrix Qi_sub = gsl_matrix_submatrix(Qi, i * d_size, j * d_size, d_size, d_size);
      //gsl_matrix_view
      DMatrix Qitemp_sub = gsl_matrix_submatrix(Qi_temp, i * d_size, j * d_size, d_size, d_size);
      //gsl_matrix_view
      DMatrix Vbeta_sub = gsl_matrix_submatrix(Vbeta, i * d_size, j * d_size, d_size, d_size);

      if (j < i) {
        //gsl_matrix_view
        DMatrix Vbeta_sym = gsl_matrix_submatrix(Vbeta, j * d_size, i * d_size, d_size, d_size);
        gsl_matrix_transpose_memcpy(&Vbeta_sub.matrix, &Vbeta_sym.matrix);
      } else {
        Qitemp_sub = matrix_mult(Qi_sub.matrix, UltVeh);
        Vbeta_sub = matrix_mult(UltVeh, Qitemp_sub);
      }
    }
  }

  // Copy beta to B, and Vbeta to se_B.
  for (size_t j = 0; j < B.shape[1]; j++) {
    for (size_t i = 0; i < B.shape[0]; i++) {
      B.set(i, j) = beta.accessor(j * d_size + i);
      se_B.set(i, j) = sqrt(Vbeta.accessor(j * d_size + i, j * d_size + i));
    }
  }

  return;
}

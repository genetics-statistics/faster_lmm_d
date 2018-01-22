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

void analyze_bimbam_mvlmm(const gsl_matrix *U, const gsl_vector *eval,
                          const gsl_matrix *UtW, const gsl_matrix *UtY) {
  debug_msg("entering");

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
  size_t msize = LMM_BATCH_SIZE;
  DMatrix Xlarge; // = gsl_matrix_alloc(U.shape[0], msize);
  DMatrix UtXlarge; // = gsl_matrix_alloc(U.shape[0], msize);
  gsl_matrix_set_zero(Xlarge);

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

  gsl_matrix_view X_sub = gsl_matrix_submatrix(X, 0, 0, c_size, n_size);
  gsl_matrix_view B_sub = gsl_matrix_submatrix(B, 0, 0, d_size, c_size);
  gsl_matrix_view xHi_all_sub =
      gsl_matrix_submatrix(xHi_all, 0, 0, d_size * c_size, d_size * n_size);

  gsl_matrix_transpose_memcpy(Y, UtY);

  gsl_matrix_transpose_memcpy(&X_sub.matrix, UtW);

  gsl_vector_view X_row = gsl_matrix_row(X, c_size);
  gsl_vector_set_zero(&X_row.vector);
  gsl_vector_view B_col = gsl_matrix_column(B, c_size);
  gsl_vector_set_zero(&B_col.vector);

  MphInitial(em_iter, em_prec, nr_iter, nr_prec, eval, &X_sub.matrix, Y, l_min,
             l_max, n_region, V_g, V_e, &B_sub.matrix);
  logl_H0 = MphEM('R', em_iter, em_prec, eval, &X_sub.matrix, Y, U_hat, E_hat,
                  OmegaU, OmegaE, UltVehiY, UltVehiBX, UltVehiU, UltVehiE, V_g,
                  V_e, &B_sub.matrix);
  logl_H0 = MphNR('R', nr_iter, nr_prec, eval, &X_sub.matrix, Y, Hi_all,
                  &xHi_all_sub.matrix, Hiy_all, V_g, V_e, Hessian, crt_a, crt_b,
                  crt_c);
  MphCalcBeta(eval, &X_sub.matrix, Y, V_g, V_e, UltVehiY, &B_sub.matrix,
              se_B_null);

  c = 0;
  Vg_remle_null.clear();
  Ve_remle_null.clear();
  for (size_t i = 0; i < d_size; i++) {
    for (size_t j = i; j < d_size; j++) {
      Vg_remle_null ~= V_g.accessor(i, j);
      Ve_remle_null ~= V_e.accessor(i, j);
      VVg_remle_null ~= Hessian.accessor(c, c);
      VVe_remle_null ~= Hessian.accessor(c + v_size, c + v_size);
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
      write(V_e(i, j), "\t");
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

  gsl_matrix_memcpy(V_g_null, V_g);
  gsl_matrix_memcpy(V_e_null, V_e);
  gsl_matrix_memcpy(B_null, B);

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

    ch_ptr = strtok_safe((char *)line.c_str(), " , \t");
    ch_ptr = strtok_safe(NULL, " , \t");
    ch_ptr = strtok_safe(NULL, " , \t");

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

        gsl_vector_set(x, c_phen, geno);
        gsl_vector_set(x_miss, c_phen, 1.0);
        x_mean += geno;
      }
      c_phen++;
    }

    x_mean /= to!double(ni_test - n_miss);

    for (size_t i = 0; i < ni_test; ++i) {
      if (gsl_vector_get(x_miss, i) == 0) {
        gsl_vector_set(x, i, x_mean);
      }
      geno = gsl_vector_get(x, i);
    }

    gsl_vector_view Xlarge_col = gsl_matrix_column(Xlarge, csnp % msize);
    gsl_vector_memcpy(&Xlarge_col.vector, x);
    csnp++;

    if (csnp % msize == 0 || csnp == t_last) {
      size_t l = 0;
      if (csnp % msize == 0) {
        l = msize;
      } else {
        l = csnp % msize;
      }

      gsl_matrix_view Xlarge_sub =
          gsl_matrix_submatrix(Xlarge, 0, 0, Xlarge.shape[0], l);
      gsl_matrix_view UtXlarge_sub =
          gsl_matrix_submatrix(UtXlarge, 0, 0, UtXlarge.shape[0], l);

      fast_dgemm("T", "N", 1.0, U, &Xlarge_sub.matrix, 0.0,
                 &UtXlarge_sub.matrix);

      gsl_matrix_set_zero(Xlarge);

      for (size_t i = 0; i < l; i++) {
        gsl_vector_view UtXlarge_col = gsl_matrix_column(UtXlarge, i);
        gsl_vector_memcpy(&X_row.vector, &UtXlarge_col.vector);

        // Initial values.
        gsl_matrix_memcpy(V_g, V_g_null);
        gsl_matrix_memcpy(V_e, V_e_null);
        gsl_matrix_memcpy(B, B_null);

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
          v_beta[i] = gsl_vector_get(beta, i);
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

/*
   This code is part of faster_lmm_d and published under the GPLv3
   License (see LICENSE.txt)

   Copyright © 2017 - 2018 Prasun Anand & Pjotr Prins
*/

module faster_lmm_d.vcm;

import core.stdc.stdlib : exit;
import core.stdc.time;

import std.algorithm;
import std.conv;
import std.exception;
import std.file;
import std.math;
import std.parallelism;
import std.algorithm: min, max, reduce, countUntil, canFind;
alias mlog = std.math.log;
import std.process;
import std.range;
import std.stdio;
alias fwrite = std.stdio.write;
import std.typecons;
import std.experimental.logger;
import std.string;

import faster_lmm_d.dmatrix;
import faster_lmm_d.gemma_lmm;
import faster_lmm_d.gemma_param;
import faster_lmm_d.helpers;
import faster_lmm_d.optmatrix;

import gsl.permutation;
import gsl.rng;
import gsl.randist;
import gsl.cdf;

class VC_PARAM {
  const DMatrix K;
  const DMatrix W;
  const DMatrix y;
  DMatrix P;
  DMatrix Py;
  DMatrix KPy_mat;
  DMatrix PKPy_mat;
  DMatrix Hessian;
  bool noconstrain;
};


void vc_WriteFile_qs(const DMatrix s_vec, const DMatrix q_vec,
                      const DMatrix qvar_vec, const DMatrix S_mat,
                      const DMatrix Svar_mat) {
  string file_str = path_out ~ "/" ~ file_out ~ ".qvec.txt";

  File outfile_q = File(file_str);

  for (size_t i = 0; i < s_vec.size; i++) {
    //outfile_q << s_vec.elements[i];
  }
  for (size_t i = 0; i < q_vec.size; i++) {
    //outfile_q << q_vec.elements[i];
  }
  for (size_t i = 0; i < qvar_vec.size; i++) {
    //outfile_q << qvar_vec.elements[i];
  }

  file_str = path_out ~ "/" ~ file_out ~ ".smat.txt";

  File outfile_s = File(file_str);

  for (size_t i = 0; i < S_mat.shape[0]; i++) {
    for (size_t j = 0; j < S_mat.shape[1]; j++) {
      //outfile_s << S_mat.accessor(i, j) << "\t";
    }
    //outfile_s << endl;
  }
  for (size_t i = 0; i < Svar_mat.shape[0]; i++) {
    for (size_t j = 0; j < Svar_mat.shape[1]; j++) {
      //outfile_s << Svar_mat.accessor(i, j) << "\t";
    }
    //outfile_s << endl;
  }

  return;
}

void UpdateParam(const DMatrix log_sigma2, VC_PARAM p) {
  size_t n1 = p.K.shape[0], n_vc = log_sigma2.size - 1, n_cvt = p.W.shape[1];

  gsl_matrix *WtHiWiWtHi = gsl_matrix_alloc(n_cvt, n1);

  double sigma2;

  // Calculate H = \sum_i^{k+1} \sigma_i^2 K_i.
  p.P = zeros_dmatrix(p.P.shape[0], p.P.shape[1]);
  for (size_t i = 0; i < n_vc + 1; i++) {
    if (i == n_vc) {
      K_temp = ones_dmatrix(n1, n1);
    } else {
      //gsl_matrix_const_view
      DMatrix K_sub = get_sub_dmatrix(p.K, 0, n1 * i, n1, n1);
      //gsl_matrix_memcpy(K_temp, &K_sub.matrix);
    }

    // When unconstrained, update on sigma2 instead of log_sigma2.
    if (p.noconstrain) {
      sigma2 = log_sigma2.elements[i];
    } else {
      sigma2 = exp(log_sigma2.elements[i]);
    }
    K_temp = K_temp * sigma2;
    p.P = p.P + K_temp;
  }

  // Calculate H^{-1}.
  p.P = p.P.inverse();

  DMatrix HiW = matrix_mult(p.P, p.W);
  DMatrix WtHiW = matrix_mult(p.W.T, HiW);

  DMatrix WtHiWi = WtHiW.inverse;

  DMatrix WtHiWiWtHi = matrix_mult(WtHiWi, HiW.T);
  //eigenlib_dgemm("N", "N", -1.0, HiW, WtHiWiWtHi, 1.0, p.P); // TODO

  // Calculate Py, KPy, PKPy.
  p.Py = matrix_mult(p.P, p.y);

  double d;
  for (size_t i = 0; i < n_vc + 1; i++) {
    //gsl_vector_view
    DMatrix KPy = get_col(p.KPy_mat, i);
    //gsl_vector_view
    DMatrix PKPy = get_col(p.PKPy_mat, i);

    if (i == n_vc) {
      //gsl_vector_memcpy(&KPy.vector, p.Py);
    } else {
      //gsl_matrix_const_view
      DMatrix K_sub = get_sub_dmatrix(p.K, 0, n1 * i, n1, n1);

      // Seems to be important to use gsl dgemv here instead of
      // eigenlib_dgemv; otherwise.
      KPy = matrix_mult(K_sub, p.Py);
    }

    PKPy = matrix_mult(p.P, KPy);

    // When phenotypes are not normalized well, then some values in
    // the following matrix maybe NaN; change that to 0; this seems to
    // only happen when eigenlib_dgemv was used above.
    for (size_t j = 0; j < p.KPy_mat.shape[0]; j++) {
      d = p.KPy_mat.accessor(j, i);
      if (isnan(d)) {
        p.KPy_mat.set(j, i, 0);
        writeln("nan appears in ", i, " ", j);
      }
      d = p.PKPy_mat.get(j, i);
      if (isnan(d)) {
        p.PKPy_mat.set(j, i, 0);
        writeln("nan appears in ", i, " ", j);
      }
    }
  }

  return;
}

// Below are functions for AI algorithm.
int LogRL_dev1(const DMatrix log_sigma2, void* params, DMatrix dev1) {
  VC_PARAM *p = cast(VC_PARAM *)params;

  size_t n1 = p.K.shape[0], n_vc = log_sigma2.size - 1;

  double tr, d;

  // Update parameters.
  UpdateParam(log_sigma2, p);

  // Calculate dev1=-0.5*trace(PK_i)+0.5*yPKPy.
  for (size_t i = 0; i < n_vc + 1; i++) {
    if (i == n_vc) {
      tr = 0;
      for (size_t l = 0; l < n1; l++) {
        tr += p.P.accessor(l, l);
      }
    } else {
      tr = 0;
      for (size_t l = 0; l < n1; l++) {
        //gsl_vector_view
        DMatrix P_row = get_row(p.P, l);
        //gsl_vector_const_view
        DMatrix K_col = get_col(p.K, n1 * i + l);
        d = vector_ddot(P_row, K_col);
        tr += d;
      }
    }

    //gsl_vector_view
    DMatrix KPy_i = get_col(p.KPy_mat, i);
    d = vector_ddot(p.Py, KPy_i);

    if (p.noconstrain) {
      d = (-0.5 * tr + 0.5 * d);
    } else {
      d = (-0.5 * tr + 0.5 * d) * exp(log_sigma2.elements[i]);
    }

    dev1.elements[i] = d;
  }

  return GSL_SUCCESS;
}

int LogRL_dev2(const DMatrix log_sigma2, void* params, DMatrix dev2) {
  VC_PARAM *p = cast(VC_PARAM *)params;

  size_t n_vc = log_sigma2.size - 1;

  double d, sigma2_i, sigma2_j;

  // Update parameters.
  UpdateParam(log_sigma2, p);

  // Calculate dev2 = 0.5(yPKPKPy).
  for (size_t i = 0; i < n_vc + 1; i++) {
    DMatrix KPy_i = gsl_matrix_column(p.KPy_mat, i);
    if (p.noconstrain) {
      sigma2_i = log_sigma2.elements[i];
    } else {
      sigma2_i = exp(log_sigma2.elements[i]);
    }

    for (size_t j = i; j < n_vc + 1; j++) {
      //gsl_vector_view
      DMatrix PKPy_j = get_col(p.PKPy_mat, j);

      d = vector_ddot(KPy_i, PKPy_j);
      if (p.noconstrain) {
        sigma2_j = log_sigma2.elements[j];
        d *= -0.5;
      } else {
        sigma2_j = exp(log_sigma2.elements[j]);
        d *= -0.5 * sigma2_i * sigma2_j;
      }

      dev2.set(i, j, d);
      if (j != i) {
        dev2.set(j, i, d);
      }
    }
  }

  //gsl_matrix_memcpy(p.Hessian, dev2);
  return GSL_SUCCESS;
}

int LogRL_dev12(const DMatrix log_sigma2, void* params, DMatrix dev1, DMatrix dev2) {
  VC_PARAM *p = cast(VC_PARAM *)params;

  size_t n1 = p.K.shape[0], n_vc = log_sigma2.size - 1;

  double tr, d, sigma2_i, sigma2_j;

  // Update parameters.
  UpdateParam(log_sigma2, p);

  for (size_t i = 0; i < n_vc + 1; i++) {
    if (i == n_vc) {
      tr = 0;
      for (size_t l = 0; l < n1; l++) {
        tr += gsl_matrix_get(p.P, l, l);
      }
    } else {
      tr = 0;
      for (size_t l = 0; l < n1; l++) {
        //gsl_vector_view 
        DMatrix P_row = get_row(p.P, l);
        //gsl_vector_const_view 
        DMatrix K_col = get_col(p.K, n1 * i + l);
        d = vector_ddot(P_row, K_col);
        tr += d;
      }
    }

    //gsl_vector_view
    DMatrix KPy_i = get_col(p.KPy_mat, i);
    d = vector_ddot(p.Py, KPy_i);

    if (p.noconstrain) {
      sigma2_i = log_sigma2.elements[i];
      d = (-0.5 * tr + 0.5 * d);
    } else {
      sigma2_i = exp(log_sigma2.elements[i]);
      d = (-0.5 * tr + 0.5 * d) * sigma2_i;
    }

    dev1.elements[i] = d;

    for (size_t j = i; j < n_vc + 1; j++) {
      //gsl_vector_view
      DMatrix PKPy_j = get_col(p.PKPy_mat, j);
      d = vector_ddot(KPy_i, PKPy_j);

      if (p.noconstrain) {
        sigma2_j = log_sigma2.elements[j];
        d *= -0.5;
      } else {
        sigma2_j = exp(log_sigma2.elements[j]);
        d *= -0.5 * sigma2_i * sigma2_j;
      }

      dev2.set(i, j, d);
      if (j != i) {
        dev2.set(j, i, d);
      }
    }
  }

  //gsl_matrix_memcpy(p->Hessian, dev2);

  return GSL_SUCCESS;
}

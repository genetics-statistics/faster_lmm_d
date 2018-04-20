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
import gsl.errno;

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
  string path_out, file_out;
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
  // To init
  DMatrix K_temp;

  size_t n1 = p.K.shape[0], n_vc = log_sigma2.size - 1, n_cvt = p.W.shape[1];

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
    K_temp = multiply_dmatrix_num(K_temp, sigma2);
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
      if (isNaN(d)) {
        p.KPy_mat.set(j, i, 0);
        writeln("nan appears in ", i, " ", j);
      }
      d = p.PKPy_mat.accessor(j, i);
      if (isNaN(d)) {
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
  //UpdateParam(log_sigma2, p);

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
  //UpdateParam(log_sigma2, p);

  // Calculate dev2 = 0.5(yPKPKPy).
  for (size_t i = 0; i < n_vc + 1; i++) {
    DMatrix KPy_i = get_col(p.KPy_mat, i);
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
  //UpdateParam(log_sigma2, p);

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

// Read cov file the first time, record mapRS2in, mapRS2var (in case
// var is not provided in the z file), store vec_n and vec_rs.
void ReadFile_cor(const string file_cor, const string[] setSnps,
                  string[] vec_rs, size_t[] vec_n,
                  double[] vec_cm, double[] vec_bp,
                  size_t[string] mapRS2in, double[string] mapRS2var) {
  writeln("entering ReadFile_cor");
  vec_rs = [];
  vec_n = [];
  mapRS2in = [];
  mapRS2var = [];

  File infile = File(file_cor);

  string rs, chr, a1, a0, pos, cm;
  double af = 0, var_x = 0, d_pos, d_cm;
  size_t n_total = 0, n_mis = 0, n_obs = 0, ni_total = 0;
  size_t ns_test = 0, ns_total = 0;

  HEADER header;

  // Header.
  safeGetline(infile, line).eof();
  ReadHeader_vc(line, header);

  if (header.n_col == 0) {
    if (header.nobs_col == 0 && header.nmis_col == 0) {
      writeln("error! missing sample size in the cor file.");
    } else {
      writeln("total sample size will be replaced by obs/mis sample size.");
    }
  }

  foreach(line; infile.byLine) {

    // do not read cor values this time; upto col_n-1.
    auto ch_ptr = line.split(" , \t");

    n_total = 0;
    n_mis = 0;
    n_obs = 0;
    af = 0;
    var_x = 0;
    d_cm = 0;
    d_pos = 0;
    for (size_t i = 0; i < header.coln - 1; i++) {
      //enforce(ch_ptr);
      if (header.rs_col != 0 && header.rs_col == i + 1) {
        rs = ch_ptr;
      }
      if (header.chr_col != 0 && header.chr_col == i + 1) {
        chr = ch_ptr;
      }
      if (header.pos_col != 0 && header.pos_col == i + 1) {
        pos = ch_ptr;
        d_pos = to!double(ch_ptr);
      }
      if (header.cm_col != 0 && header.cm_col == i + 1) {
        cm = ch_ptr;
        d_cm = to!double(ch_ptr);
      }
      if (header.a1_col != 0 && header.a1_col == i + 1) {
        a1 = ch_ptr;
      }
      if (header.a0_col != 0 && header.a0_col == i + 1) {
        a0 = ch_ptr;
      }

      if (header.n_col != 0 && header.n_col == i + 1) {
        n_total = to!int(ch_ptr);
      }
      if (header.nmis_col != 0 && header.nmis_col == i + 1) {
        n_mis = to!int(ch_ptr);
      }
      if (header.nobs_col != 0 && header.nobs_col == i + 1) {
        n_obs = to!int(ch_ptr);
      }

      if (header.af_col != 0 && header.af_col == i + 1) {
        af = to!double(ch_ptr);
      }
      if (header.var_col != 0 && header.var_col == i + 1) {
        var_x = to!double(ch_ptr);
      }

      ch_ptr = strtok(NULL, " , \t");
    }

    if (header.rs_col == 0) {
      rs = chr + ":" + pos;
    }

    if (header.n_col == 0) {
      n_total = n_mis + n_obs;
    }

    // Record rs, n.
    vec_rs ~= rs;
    vec_n ~= n_total;
    if (d_cm > 0) {
      vec_cm ~= d_cm;
    } else {
      vec_cm ~= d_cm;
    }
    if (d_pos > 0) {
      vec_bp ~= d_pos;
    } else {
      vec_bp ~= d_pos;
    }

    // Record mapRS2in and mapRS2var.
    if (setSnps.size() == 0 || setSnps.count(rs) != 0) {
      if (mapRS2in.count(rs) == 0) {
        mapRS2in[rs] = 1;

        if (header.var_col != 0) {
          mapRS2var[rs] = var_x;
        } else if (header.af_col != 0) {
          var_x = 2.0 * af * (1.0 - af);
          mapRS2var[rs] = var_x;
        } else {
        }

        ns_test++;

      } else {
        writeln("error! more than one snp has the same id ", rs, " in cor file?");
      }
    }

    // Record max pos.
    ni_total = max(ni_total, n_total);
    ns_total++;
  }

  return;
}


// Read beta file, store mapRS2var if var is provided here, calculate
// q and var_y.
void ReadFile_beta(const bool flag_priorscale, const string file_beta,
                   const size_t[string] mapRS2cat,
                   size_t[string] mapRS2in,
                   double[string] mapRS2var,
                   size_t[string] mapRS2nsamp, DMatrix q_vec,
                   DMatrix qvar_vec, DMatrix s_vec, size_t ni_total,
                   size_t ns_total) {
  writeln("entering ReadFile_beta");
  mapRS2nsamp = [];

  File infile = File(file_beta);

  string type;

  string rs, chr, a1, a0, pos, cm;
  double z = 0, beta = 0, se_beta = 0, chisq = 0, pvalue = 0, zsquare = 0,
         af = 0, var_x = 0;
  size_t n_total = 0, n_mis = 0, n_obs = 0;
  size_t ns_test = 0;
  ns_total = 0;
  ni_total = 0;

  double[] vec_q, vec_qvar, vec_s;
  for (size_t i = 0; i < q_vec.size; i++) {
    vec_q ~= 0.0;
    vec_qvar ~= 0.0;
    vec_s ~= 0.0;
  }

  // Read header.
  HEADER header;
  safeGetline(infile, line).eof();
  ReadHeader_vc(line, header);

  if (header.n_col == 0) {
    if (header.nobs_col == 0 && header.nmis_col == 0) {
      writeln("error! missing sample size in the beta file.");
    } else {
      writeln("total sample size will be replaced by obs/mis sample size.");
    }
  }

  if (header.z_col == 0 && (header.beta_col == 0 || header.sebeta_col == 0) &&
      header.chisq_col == 0 && header.p_col == 0) {
    writeln("error! missing z scores in the beta file.");
  }

  if (header.af_col == 0 && header.var_col == 0 && mapRS2var.size() == 0) {
    writeln("error! missing allele frequency in the beta file.");
  }

  foreach(line; infile.byLine) {

    // do not read cor values this time; upto col_n-1.
    auto ch_ptr = line.split(" , \t");

    z = 0;
    beta = 0;
    se_beta = 0;
    chisq = 0;
    pvalue = 0;
    n_total = 0;
    n_mis = 0;
    n_obs = 0;
    af = 0;
    var_x = 0;
    for (size_t i = 0; i < header.coln; i++) {
      enforce(ch_ptr);
      if (header.rs_col != 0 && header.rs_col == i + 1) {
        rs = ch_ptr;
      }
      if (header.chr_col != 0 && header.chr_col == i + 1) {
        chr = ch_ptr;
      }
      if (header.pos_col != 0 && header.pos_col == i + 1) {
        pos = ch_ptr;
      }
      if (header.cm_col != 0 && header.cm_col == i + 1) {
        cm = ch_ptr;
      }
      if (header.a1_col != 0 && header.a1_col == i + 1) {
        a1 = ch_ptr;
      }
      if (header.a0_col != 0 && header.a0_col == i + 1) {
        a0 = ch_ptr;
      }

      if (header.z_col != 0 && header.z_col == i + 1) {
        z = to!double(ch_ptr);
      }
      if (header.beta_col != 0 && header.beta_col == i + 1) {
        beta = to!double(ch_ptr);
      }
      if (header.sebeta_col != 0 && header.sebeta_col == i + 1) {
        se_beta = to!double(ch_ptr);
      }
      if (header.chisq_col != 0 && header.chisq_col == i + 1) {
        chisq = to!double(ch_ptr);
      }
      if (header.p_col != 0 && header.p_col == i + 1) {
        pvalue = to!double(ch_ptr);
      }

      if (header.n_col != 0 && header.n_col == i + 1) {
        n_total = to!int(ch_ptr);
      }
      if (header.nmis_col != 0 && header.nmis_col == i + 1) {
        n_mis = to!int(ch_ptr);
      }
      if (header.nobs_col != 0 && header.nobs_col == i + 1) {
        n_obs = to!int(ch_ptr);
      }

      if (header.af_col != 0 && header.af_col == i + 1) {
        af = to!double(ch_ptr);
      }
      if (header.var_col != 0 && header.var_col == i + 1) {
        var_x = to!double(ch_ptr);
      }

      ch_ptr = strtok(NULL, " , \t");
    }

    if (header.rs_col == 0) {
      rs = chr + ":" + pos;
    }

    if (header.n_col == 0) {
      n_total = n_mis + n_obs;
    }

    // Both z values and beta/se_beta have directions, while
    // chisq/pvalue do not.
    if (header.z_col != 0) {
      zsquare = z * z;
    } else if (header.beta_col != 0 && header.sebeta_col != 0) {
      z = beta / se_beta;
      zsquare = z * z;
    } else if (header.chisq_col != 0) {
      zsquare = chisq;
    } else if (header.p_col != 0) {
      zsquare = gsl_cdf_chisq_Qinv(pvalue, 1);
    } else {
      zsquare = 0;
    }

    // If the snp is also present in cor file, then do calculations.
    if ((header.var_col != 0 || header.af_col != 0 ||
         mapRS2var.count(rs) != 0) &&
        mapRS2in.count(rs) != 0 &&
        (mapRS2cat.size() == 0 || mapRS2cat.count(rs) != 0)) {
      if (mapRS2in.at(rs) > 1) {
        writeln("error! more than one snp has the same id ", rs, " in beta file?");
        break;
      }

      if (header.var_col == 0) {
        if (header.af_col != 0) {
          var_x = 2.0 * af * (1.0 - af);
        } else {
          var_x = mapRS2var.at(rs);
        }
      }

      if (flag_priorscale) {
        var_x = 1;
      }

      mapRS2in[rs]++;
      mapRS2var[rs] = var_x;
      mapRS2nsamp[rs] = n_total;

      if (mapRS2cat.length != 0) {
        vec_q[mapRS2cat.at(rs)] += (zsquare - 1.0) * var_x / to!double(n_total);
        vec_s[mapRS2cat.at(rs)] += var_x;
        vec_qvar[mapRS2cat.at(rs)] += var_x * var_x / to!double(n_total * n_total);
      } else {
        vec_q[0] += (zsquare - 1.0) * var_x / to!double(n_total);
        vec_s[0] += var_x;
        vec_qvar[0] += var_x * var_x / to!double(n_total * n_total);
      }

      ni_total = max(ni_total, n_total);
      ns_test++;
    }

    ns_total++;
  }

  for (size_t i = 0; i < q_vec.size; i++) {
    q_vec.elements[i] = vec_q[i];
    qvar_vec.elements[i] = 2.0 * vec_qvar[i];
    s_vec.elements[i] = vec_s[i];
  }

  return;
}

// Read cov file the first time, record mapRS2in, mapRS2var (in case
// var is not provided in the z file), store vec_n and vec_rs.
void ReadFile_cor(const string file_cor, const string[] setSnps,
                  string[] vec_rs, size_t[] vec_n,
                  double[] vec_cm, double[] vec_bp,
                  size_t[string] mapRS2in,
                  double[string] mapRS2var) {
  writeln("entering ReadFile_cor");
  vec_rs = [];
  vec_n = [];
  mapRS2in = [];
  mapRS2var = [];

  File infile = File(file_cor);

  string rs, chr, a1, a0, pos, cm;
  double af = 0, var_x = 0, d_pos, d_cm;
  size_t n_total = 0, n_mis = 0, n_obs = 0, ni_total = 0;
  size_t ns_test = 0, ns_total = 0;

  HEADER header;

  // Header.
  safeGetline(infile, line).eof();
  ReadHeader_vc(line, header);

  if (header.n_col == 0) {
    if (header.nobs_col == 0 && header.nmis_col == 0) {
      writeln("error! missing sample size in the cor file.");
    } else {
      writeln("total sample size will be replaced by obs/mis sample size.");
    }
  }

  foreach(line; infile.byLine) {

    // do not read cor values this time; upto col_n-1.
    auto ch_ptr = line.split(" , \t");

    n_total = 0;
    n_mis = 0;
    n_obs = 0;
    af = 0;
    var_x = 0;
    d_cm = 0;
    d_pos = 0;
    for (size_t i = 0; i < header.coln - 1; i++) {
      enforce(ch_ptr);
      if (header.rs_col != 0 && header.rs_col == i + 1) {
        rs = ch_ptr;
      }
      if (header.chr_col != 0 && header.chr_col == i + 1) {
        chr = ch_ptr;
      }
      if (header.pos_col != 0 && header.pos_col == i + 1) {
        pos = ch_ptr;
        d_pos = to!double(ch_ptr);
      }
      if (header.cm_col != 0 && header.cm_col == i + 1) {
        cm = ch_ptr;
        d_cm = to!double(ch_ptr);
      }
      if (header.a1_col != 0 && header.a1_col == i + 1) {
        a1 = ch_ptr;
      }
      if (header.a0_col != 0 && header.a0_col == i + 1) {
        a0 = ch_ptr;
      }

      if (header.n_col != 0 && header.n_col == i + 1) {
        n_total = to!int(ch_ptr);
      }
      if (header.nmis_col != 0 && header.nmis_col == i + 1) {
        n_mis = to!int(ch_ptr);
      }
      if (header.nobs_col != 0 && header.nobs_col == i + 1) {
        n_obs = to!int(ch_ptr);
      }

      if (header.af_col != 0 && header.af_col == i + 1) {
        af = to!double(ch_ptr);
      }
      if (header.var_col != 0 && header.var_col == i + 1) {
        var_x = to!double(ch_ptr);
      }

      ch_ptr = strtok(NULL, " , \t");
    }

    if (header.rs_col == 0) {
      rs = chr + ":" + pos;
    }

    if (header.n_col == 0) {
      n_total = n_mis + n_obs;
    }

    // Record rs, n.
    vec_rs ~= rs;
    vec_n.push_back(n_total);
    if (d_cm > 0) {
      vec_cm ~= d_cm;
    } else {
      vec_cm ~= d_cm;
    }
    if (d_pos > 0) {
      vec_bp ~= d_pos;
    } else {
      vec_bp ~= d_pos;
    }

    // Record mapRS2in and mapRS2var.
    if (setSnps.size() == 0 || setSnps.count(rs) != 0) {
      if (mapRS2in.count(rs) == 0) {
        mapRS2in[rs] = 1;

        if (header.var_col != 0) {
          mapRS2var[rs] = var_x;
        } else if (header.af_col != 0) {
          var_x = 2.0 * af * (1.0 - af);
          mapRS2var[rs] = var_x;
        } else {
        }

        ns_test++;

      } else {
        writeln("error! more than one snp has the same id ", rs, " in cor file?");
      }
    }

    // Record max pos.
    ni_total = max(ni_total, n_total);
    ns_total++;
  }

  return;
}

// Read beta file, store mapRS2var if var is provided here, calculate
// q and var_y.
void ReadFile_beta(const bool flag_priorscale, const string file_beta,
                   const size_t[string] mapRS2cat,
                   size_t[string] mapRS2in,
                   double[string] mapRS2var,
                   size_t[string] mapRS2nsamp, DMatrix q_vec,
                   DMatrix qvar_vec, DMatrix s_vec, DMatrix ni_total,
                   size_t ns_total) {
  writeln("entering ReadFile_beta");
  mapRS2nsamp = [];

  File infile = File(file_beta);

  string type;

  string rs, chr, a1, a0, pos, cm;
  double z = 0, beta = 0, se_beta = 0, chisq = 0, pvalue = 0, zsquare = 0,
         af = 0, var_x = 0;
  size_t n_total = 0, n_mis = 0, n_obs = 0;
  size_t ns_test = 0;
  ns_total = 0;
  ni_total = 0;

  double[] vec_q, vec_qvar, vec_s;
  for (size_t i = 0; i < q_vec.size; i++) {
    vec_q ~= 0.0;
    vec_qvar ~= 0.0;
    vec_s ~= 0.0;
  }

  // Read header.
  HEADER header;
  safeGetline(infile, line).eof();
  ReadHeader_vc(line, header);

  if (header.n_col == 0) {
    if (header.nobs_col == 0 && header.nmis_col == 0) {
      writeln("error! missing sample size in the beta file.");
    } else {
      writeln("total sample size will be replaced by obs/mis sample size.");
    }
  }

  if (header.z_col == 0 && (header.beta_col == 0 || header.sebeta_col == 0) &&
      header.chisq_col == 0 && header.p_col == 0) {
    writeln("error! missing z scores in the beta file.");
  }

  if (header.af_col == 0 && header.var_col == 0 && mapRS2var.length == 0) {
    writeln("error! missing allele frequency in the beta file.");
  }

 foreach(line; infile.byLine) {
    auto ch_ptr = line.split(" , \t");

    z = 0;
    beta = 0;
    se_beta = 0;
    chisq = 0;
    pvalue = 0;
    n_total = 0;
    n_mis = 0;
    n_obs = 0;
    af = 0;
    var_x = 0;
    for (size_t i = 0; i < header.coln; i++) {
      //enforce(ch_ptr);
      if (header.rs_col != 0 && header.rs_col == i + 1) {
        rs = ch_ptr;
      }
      if (header.chr_col != 0 && header.chr_col == i + 1) {
        chr = ch_ptr;
      }
      if (header.pos_col != 0 && header.pos_col == i + 1) {
        pos = ch_ptr;
      }
      if (header.cm_col != 0 && header.cm_col == i + 1) {
        cm = ch_ptr;
      }
      if (header.a1_col != 0 && header.a1_col == i + 1) {
        a1 = ch_ptr;
      }
      if (header.a0_col != 0 && header.a0_col == i + 1) {
        a0 = ch_ptr;
      }

      if (header.z_col != 0 && header.z_col == i + 1) {
        z = to!double(ch_ptr);
      }
      if (header.beta_col != 0 && header.beta_col == i + 1) {
        beta = to!double(ch_ptr);
      }
      if (header.sebeta_col != 0 && header.sebeta_col == i + 1) {
        se_beta = to!double(ch_ptr);
      }
      if (header.chisq_col != 0 && header.chisq_col == i + 1) {
        chisq = to!double(ch_ptr);
      }
      if (header.p_col != 0 && header.p_col == i + 1) {
        pvalue = to!double(ch_ptr);
      }

      if (header.n_col != 0 && header.n_col == i + 1) {
        n_total = to!int(ch_ptr);
      }
      if (header.nmis_col != 0 && header.nmis_col == i + 1) {
        n_mis = to!int(ch_ptr);
      }
      if (header.nobs_col != 0 && header.nobs_col == i + 1) {
        n_obs = to!int(ch_ptr);
      }

      if (header.af_col != 0 && header.af_col == i + 1) {
        af = to!double(ch_ptr);
      }
      if (header.var_col != 0 && header.var_col == i + 1) {
        var_x = to!double(ch_ptr);
      }

      ch_ptr = strtok(NULL, " , \t");
    }

    if (header.rs_col == 0) {
      rs = chr + ":" + pos;
    }

    if (header.n_col == 0) {
      n_total = n_mis + n_obs;
    }

    // Both z values and beta/se_beta have directions, while
    // chisq/pvalue do not.
    if (header.z_col != 0) {
      zsquare = z * z;
    } else if (header.beta_col != 0 && header.sebeta_col != 0) {
      z = beta / se_beta;
      zsquare = z * z;
    } else if (header.chisq_col != 0) {
      zsquare = chisq;
    } else if (header.p_col != 0) {
      zsquare = gsl_cdf_chisq_Qinv(pvalue, 1);
    } else {
      zsquare = 0;
    }

    // If the snp is also present in cor file, then do calculations.
    if ((header.var_col != 0 || header.af_col != 0 ||
         mapRS2var.count(rs) != 0) &&
        mapRS2in.count(rs) != 0 &&
        (mapRS2cat.size() == 0 || mapRS2cat.count(rs) != 0)) {
      if (mapRS2in.at(rs) > 1) {
        writeln("error! more than one snp has the same id ", rs, " in beta file?");
        break;
      }

      if (header.var_col == 0) {
        if (header.af_col != 0) {
          var_x = 2.0 * af * (1.0 - af);
        } else {
          var_x = mapRS2var.at(rs);
        }
      }

      if (flag_priorscale) {
        var_x = 1;
      }

      mapRS2in[rs]++;
      mapRS2var[rs] = var_x;
      mapRS2nsamp[rs] = n_total;

      if (mapRS2cat.size() != 0) {
        vec_q[mapRS2cat.at(rs)] += (zsquare - 1.0) * var_x / to!double(n_total);
        vec_s[mapRS2cat.at(rs)] += var_x;
        vec_qvar[mapRS2cat.at(rs)] +=
            var_x * var_x / to!double(n_total * n_total);
      } else {
        vec_q[0] += (zsquare - 1.0) * var_x / to!double(n_total);
        vec_s[0] += var_x;
        vec_qvar[0] += var_x * var_x / to!double(n_total * n_total);
      }

      ni_total = max(ni_total, n_total);
      ns_test++;
    }

    ns_total++;
  }

  for (size_t i = 0; i < q_vec.size; i++) {
    q_vec.elements[i] = vec_q[i];
    qvar_vec.elements[i] = 2.0 * vec_qvar[i];
    s_vec.elements[i] = vec_s[i];
  }

  return;
}


// Read covariance file the second time.
// Look for rs, n_mis+n_obs, var, window_size, cov.
// If window_cm/bp/ns is provided, then use these max values to
// calibrate estimates.
void ReadFile_cor(const string file_cor, const string[] vec_rs,
                  const size_t[] vec_n, const double[] vec_cm,
                  const double[] vec_bp,
                  const size_t[string] mapRS2cat,
                  const size_t[string] mapRS2in,
                  const double[string] mapRS2var,
                  const size_t[string] mapRS2nsamp, const size_t crt,
                  const double window_cm, const double window_bp,
                  const double window_ns, DMatrix S_mat,
                  DMatrix Svar_mat, DMatrix qvar_vec, size_t ni_total,
                  size_t ns_total, size_t ns_test, size_t ns_pair) {
  writeln("entering ReadFile_cor");
  File infile = File(file_cor);

  string rs1, rs2;
  double d1, d2, d3, cor, var1, var2;
  size_t n_nb, nsamp1, nsamp2, n12, bin_size = 10, bin;

  double[][] mat_S, mat_Svar, mat_tmp;
  double[] vec_qvar, vec_tmp;
  double[][][] mat3d_Sbin;

  for (size_t i = 0; i < S_mat->size1; i++) {
    vec_qvar ~= 0.0;
  }

  for (size_t i = 0; i < S_mat->size1; i++) {
    mat_S ~= vec_qvar;
    mat_Svar ~= vec_qvar;
  }

  for (size_t k = 0; k < bin_size; k++) {
    vec_tmp ~= 0.0;
  }
  for (size_t i = 0; i < S_mat->size1; i++) {
    mat_tmp ~= vec_tmp;
  }
  for (size_t i = 0; i < S_mat->size1; i++) {
    mat3d_Sbin ~= mat_tmp;
  }

  string rs, chr, a1, a0, type, pos, cm;
  size_t n_total = 0, n_mis = 0, n_obs = 0;
  double d_pos1, d_pos2, d_pos, d_cm1, d_cm2, d_cm;
  ns_test = 0;
  ns_total = 0;
  ns_pair = 0;
  ni_total = 0;

  // Header.
  HEADER header;

  safeGetline(infile, line).eof();
  ReadHeader_vc(line, header);

  while (!safeGetline(infile, line).eof()) {

    // Do not read cor values this time; upto col_n-1.
    d_pos1 = 0;
    d_cm1 = 0;
    ch_ptr = strtok_safe((char *)line.c_str(), " , \t");
    for (size_t i = 0; i < header.coln - 1; i++) {
      enforce(ch_ptr);
      if (header.rs_col != 0 && header.rs_col == i + 1) {
        rs = ch_ptr;
      }
      if (header.chr_col != 0 && header.chr_col == i + 1) {
        chr = ch_ptr;
      }
      if (header.pos_col != 0 && header.pos_col == i + 1) {
        pos = ch_ptr;
        d_pos1 = to!double(ch_ptr);
      }
      if (header.cm_col != 0 && header.cm_col == i + 1) {
        cm = ch_ptr;
        d_cm1 = to!double(ch_ptr);
      }
      if (header.a1_col != 0 && header.a1_col == i + 1) {
        a1 = ch_ptr;
      }
      if (header.a0_col != 0 && header.a0_col == i + 1) {
        a0 = ch_ptr;
      }

      if (header.n_col != 0 && header.n_col == i + 1) {
        n_total = to!int(ch_ptr);
      }
      if (header.nmis_col != 0 && header.nmis_col == i + 1) {
        n_mis = to!int(ch_ptr);
      }
      if (header.nobs_col != 0 && header.nobs_col == i + 1) {
        n_obs = to!int(ch_ptr);
      }

      ch_ptr = strtok(NULL, " , \t");
    }

    if (header.rs_col == 0) {
      rs = chr + ":" + pos;
    }

    if (header.n_col == 0) {
      n_total = n_mis + n_obs;
    }

    rs1 = rs;

    if ((mapRS2cat.length == 0 || mapRS2cat.count(rs1) != 0) &&
        mapRS2in.count(rs1) != 0 && mapRS2in.at(rs1) == 2) {
      var1 = mapRS2var.at(rs1);
      nsamp1 = mapRS2nsamp.at(rs1);
      d2 = var1 * var1;

      if (mapRS2cat.size() != 0) {
        mat_S[mapRS2cat.at(rs1)][mapRS2cat.at(rs1)] +=
            (1 - 1.0 / to!double(vec_n[ns_total])) * d2;
        mat_Svar[mapRS2cat.at(rs1)][mapRS2cat.at(rs1)] +=
            d2 * d2 / to!double(vec_n[ns_total] * vec_n[ns_total]);
        if (crt == 1) {
          mat3d_Sbin[mapRS2cat.at(rs1)][mapRS2cat.at(rs1)][0] +=
              (1 - 1.0 / to!double(vec_n[ns_total])) * d2;
        }
      } else {
        mat_S[0][0] += (1 - 1.0 / (double)vec_n[ns_total]) * d2;
        mat_Svar[0][0] +=
            d2 * d2 / to!double(vec_n[ns_total] * vec_n[ns_total]);
        if (crt == 1) {
          mat3d_Sbin[0][0][0] += (1 - 1.0 / to!double(vec_n[ns_total])) * d2;
        }
      }

      n_nb = 0;
      while (ch_ptr != NULL) {
        type = ch_ptr;
        if (type.compare("NA") != 0 && type.compare("na") != 0 &&
            type.compare("nan") != 0 && type.compare("-nan") != 0) {
          cor = to!double(ch_ptr);
          rs2 = vec_rs[ns_total + n_nb + 1];
          d_pos2 = vec_bp[ns_total + n_nb + 1];
          d_cm2 = vec_cm[ns_total + n_nb + 1];
          d_pos = abs(d_pos2 - d_pos1);
          d_cm = abs(d_cm2 - d_cm1);

          if ((mapRS2cat.length == 0 || mapRS2cat.count(rs2) != 0) &&
              mapRS2in.count(rs2) != 0 && mapRS2in.at(rs2) == 2) {
            var2 = mapRS2var.at(rs2);
            nsamp2 = mapRS2nsamp.at(rs2);
            d1 = cor * cor -
                 1.0 / to!double(min(vec_n[ns_total], vec_n[ns_total + n_nb + 1]));
            d2 = var1 * var2;
            d3 = cor * cor / to!double(nsamp1 * nsamp2);
            n12 = min(vec_n[ns_total], vec_n[ns_total + n_nb + 1]);

            // Compute bin.
            if (crt == 1) {
              if (window_cm != 0 && d_cm1 != 0 && d_cm2 != 0) {
                bin =
                    min(to!int(floor(d_cm / window_cm * bin_size)), to!int(bin_size));
              } else if (window_bp != 0 && d_pos1 != 0 && d_pos2 != 0) {
                bin = min(to!int(floor(d_pos / window_bp * bin_size)),
                          to!int(bin_size));
              } else if (window_ns != 0) {
                bin = min(to!int(floor(((double)n_nb + 1) / window_ns * bin_size)),
                          to!int(bin_size));
              }
            }

            if (mapRS2cat.length != 0) {
              if (mapRS2cat.at(rs1) == mapRS2cat.at(rs2)) {
                vec_qvar[mapRS2cat.at(rs1)] += 2 * d3 * d2;
                mat_S[mapRS2cat.at(rs1)][mapRS2cat.at(rs2)] += 2 * d1 * d2;
                mat_Svar[mapRS2cat.at(rs1)][mapRS2cat.at(rs2)] +=
                    2 * d2 * d2 / to!double(n12 * n12);
                if (crt == 1) {
                  mat3d_Sbin[mapRS2cat.at(rs1)][mapRS2cat.at(rs2)][bin] +=
                      2 * d1 * d2;
                }
              } else {
                mat_S[mapRS2cat.at(rs1)][mapRS2cat.at(rs2)] += d1 * d2;
                mat_Svar[mapRS2cat.at(rs1)][mapRS2cat.at(rs2)] +=
                    d2 * d2 / to!double(n12 * n12);
                if (crt == 1) {
                  mat3d_Sbin[mapRS2cat.at(rs1)][mapRS2cat.at(rs2)][bin] +=
                      d1 * d2;
                }
              }
            } else {
              vec_qvar[0] += 2 * d3 * d2;
              mat_S[0][0] += 2 * d1 * d2;
              mat_Svar[0][0] += 2 * d2 * d2 / to!double(n12 * n12);

              if (crt == 1) {
                mat3d_Sbin[0][0][bin] += 2 * d1 * d2;
              }
            }
            ns_pair++;
          }
        }

        ch_ptr = strtok(NULL, " , \t");
        n_nb++;
      }
      ni_total = max(ni_total, n_total);
      ns_test++;
    }

    ns_total++;
  }

  // Use S_bin to fit a rational function y=1/(a+bx)^2, where
  // x=seq(0.5,bin_size-0.5,by=1) and then compute a correlation
  // factor as a percentage.
  double a, b, x, y, n, var_y, var_x, mean_y, mean_x, cov_xy, crt_factor;
  if (crt == 1) {
    for (size_t i = 0; i < S_mat.shape[0]; i++) {
      for (size_t j = i; j < S_mat.shape[1]; j++) {

        // Correct mat_S.
        n = 0;
        var_y = 0;
        var_x = 0;
        mean_y = 0;
        mean_x = 0;
        cov_xy = 0;
        for (size_t k = 0; k < bin_size; k++) {
          if (j == i) {
            y = mat3d_Sbin[i][j][k];
          } else {
            y = mat3d_Sbin[i][j][k] + mat3d_Sbin[j][i][k];
          }
          x = k + 0.5;
          write(y, ", ");
          if (y > 0) {
            y = 1 / sqrt(y);
            mean_x += x;
            mean_y += y;
            var_x += x * x;
            var_y += y * y;
            cov_xy += x * y;
            n++;
          }
        }
        write("\n");

        if (n >= 5) {
          mean_x /= n;
          mean_y /= n;
          var_x /= n;
          var_y /= n;
          cov_xy /= n;
          var_x -= mean_x * mean_x;
          var_y -= mean_y * mean_y;
          cov_xy -= mean_x * mean_y;
          b = cov_xy / var_x;
          a = mean_y - b * mean_x;
          crt_factor = a / (b * (bin_size + 0.5)) + 1;
          if (i == j) {
            mat_S[i][j] *= crt_factor;
          } else {
            mat_S[i][j] *= crt_factor;
            mat_S[j][i] *= crt_factor;
          }
          write(crt_factor, "\n");

          // Correct qvar.
          if (i == j) {
            vec_qvar[i] *= crt_factor;
          }
        }
      }
    }
  }

  // Save to gsl_vector and gsl_matrix: qvar_vec, S_mat, Svar_mat.
  for (size_t i = 0; i < S_mat->size1; i++) {
    d1 = gsl_vector_get(qvar_vec, i) + 2 * vec_qvar[i];
    gsl_vector_set(qvar_vec, i, d1);
    for (size_t j = 0; j < S_mat->size2; j++) {
      if (i == j) {
        S_mat.set(i, j, mat_S[i][i]);
        Svar_mat.set(i, j, 2.0 * mat_Svar[i][i] * ns_test *
                                           ns_test / (2.0 * ns_pair));
      } else {
        S_mat.set(i, j, mat_S[i][j] + mat_S[j][i]);
        Svar_mat.set(i, j, 2.0 * (mat_Svar[i][j] + mat_Svar[j][i]) *
                                           ns_test * ns_test / (2.0 * ns_pair));
      }
    }
  }

  return;
}

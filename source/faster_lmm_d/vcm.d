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

// Read header to determine which column contains which item.
bool ReadHeader_vc(const string line, HEADER header) {
  debug_msg("entering");
  string[] rs_ptr = ["rs",   "RS",    "snp",   "SNP",  "snps",
                     "SNPS", "snpid", "SNPID", "rsid", "RSID"];
  set<string> rs_set(rs_ptr, rs_ptr + 10);
  string chr_ptr[] = {"chr", "CHR"};
  set<string> chr_set(chr_ptr, chr_ptr + 2);
  string pos_ptr[] = {
      "ps", "PS", "pos", "POS", "base_position", "BASE_POSITION", "bp", "BP"};
  set<string> pos_set(pos_ptr, pos_ptr + 8);
  string cm_ptr[] = {"cm", "CM"};
  set<string> cm_set(cm_ptr, cm_ptr + 2);
  string a1_ptr[] = {"a1", "A1", "allele1", "ALLELE1"};
  set<string> a1_set(a1_ptr, a1_ptr + 4);
  string a0_ptr[] = {"a0", "A0", "allele0", "ALLELE0"};
  set<string> a0_set(a0_ptr, a0_ptr + 4);

  string z_ptr[] = {"z", "Z", "z_score", "Z_SCORE", "zscore", "ZSCORE"};
  set<string> z_set(z_ptr, z_ptr + 6);
  string beta_ptr[] = {"beta", "BETA", "b", "B"};
  set<string> beta_set(beta_ptr, beta_ptr + 4);
  string sebeta_ptr[] = {"se_beta", "SE_BETA", "se", "SE"};
  set<string> sebeta_set(sebeta_ptr, sebeta_ptr + 4);
  string chisq_ptr[] = {"chisq", "CHISQ", "chisquare", "CHISQUARE"};
  set<string> chisq_set(chisq_ptr, chisq_ptr + 4);
  string p_ptr[] = {"p", "P", "pvalue", "PVALUE", "p-value", "P-VALUE"};
  set<string> p_set(p_ptr, p_ptr + 6);

  string n_ptr[] = {"n", "N", "ntotal", "NTOTAL", "n_total", "N_TOTAL"};
  set<string> n_set(n_ptr, n_ptr + 6);
  string nmis_ptr[] = {"nmis", "NMIS", "n_mis", "N_MIS", "n_miss", "N_MISS"};
  set<string> nmis_set(nmis_ptr, nmis_ptr + 6);
  string nobs_ptr[] = {"nobs", "NOBS", "n_obs", "N_OBS"};
  set<string> nobs_set(nobs_ptr, nobs_ptr + 4);

  string af_ptr[] = {"af",
                     "AF",
                     "maf",
                     "MAF",
                     "f",
                     "F",
                     "allele_freq",
                     "ALLELE_FREQ",
                     "allele_frequency",
                     "ALLELE_FREQUENCY"};
  set<string> af_set(af_ptr, af_ptr + 10);
  string var_ptr[] = {"var", "VAR"};
  set<string> var_set(var_ptr, var_ptr + 2);

  string ws_ptr[] = {"window_size", "WINDOW_SIZE", "ws", "WS"};
  set<string> ws_set(ws_ptr, ws_ptr + 4);
  string cor_ptr[] = {"cor", "COR", "r", "R"};
  set<string> cor_set(cor_ptr, cor_ptr + 4);

  header.rs_col = 0;
  header.chr_col = 0;
  header.pos_col = 0;
  header.a1_col = 0;
  header.a0_col = 0;
  header.z_col = 0;
  header.beta_col = 0;
  header.sebeta_col = 0;
  header.chisq_col = 0;
  header.p_col = 0;
  header.n_col = 0;
  header.nmis_col = 0;
  header.nobs_col = 0;
  header.af_col = 0;
  header.var_col = 0;
  header.ws_col = 0;
  header.cor_col = 0;
  header.coln = 0;

  char *ch_ptr;
  string type;
  size_t n_error = 0;

  ch_ptr = line.split(" , \t");
  while (ch_ptr != NULL) {
    type = ch_ptr;
    if (rs_set.count(type) != 0) {
      if (header.rs_col == 0) {
        header.rs_col = header.coln + 1;
      } else {
        writeln("error! more than two rs columns in the file.");
        n_error++;
      }
    } else if (chr_set.count(type) != 0) {
      if (header.chr_col == 0) {
        header.chr_col = header.coln + 1;
      } else {
        writeln("error! more than two chr columns in the file.");
        n_error++;
      }
    } else if (pos_set.count(type) != 0) {
      if (header.pos_col == 0) {
        header.pos_col = header.coln + 1;
      } else {
        writeln("error! more than two pos columns in the file.");
        n_error++;
      }
    } else if (cm_set.count(type) != 0) {
      if (header.cm_col == 0) {
        header.cm_col = header.coln + 1;
      } else {
        writeln("error! more than two cm columns in the file.");
        n_error++;
      }
    } else if (a1_set.count(type) != 0) {
      if (header.a1_col == 0) {
        header.a1_col = header.coln + 1;
      } else {
        writeln("error! more than two allele1 columns in the file.");
        n_error++;
      }
    } else if (a0_set.count(type) != 0) {
      if (header.a0_col == 0) {
        header.a0_col = header.coln + 1;
      } else {
        writeln("error! more than two allele0 columns in the file.");
        n_error++;
      }
    } else if (z_set.count(type) != 0) {
      if (header.z_col == 0) {
        header.z_col = header.coln + 1;
      } else {
        writeln("error! more than two z columns in the file.");
        n_error++;
      }
    } else if (beta_set.count(type) != 0) {
      if (header.beta_col == 0) {
        header.beta_col = header.coln + 1;
      } else {
        writeln("error! more than two beta columns in the file.");
        n_error++;
      }
    } else if (sebeta_set.count(type) != 0) {
      if (header.sebeta_col == 0) {
        header.sebeta_col = header.coln + 1;
      } else {
        writeln("error! more than two se_beta columns in the file.");
        n_error++;
      }
    } else if (chisq_set.count(type) != 0) {
      if (header.chisq_col == 0) {
        header.chisq_col = header.coln + 1;
      } else {
        writeln("error! more than two z columns in the file.");
        n_error++;
      }
    } else if (p_set.count(type) != 0) {
      if (header.p_col == 0) {
        header.p_col = header.coln + 1;
      } else {
        writeln("error! more than two p columns in the file.");
        n_error++;
      }
    } else if (n_set.count(type) != 0) {
      if (header.n_col == 0) {
        header.n_col = header.coln + 1;
      } else {
        writeln("error! more than two n_total columns in the file.");
        n_error++;
      }
    } else if (nmis_set.count(type) != 0) {
      if (header.nmis_col == 0) {
        header.nmis_col = header.coln + 1;
      } else {
        writeln("error! more than two n_mis columns in the file.");
        n_error++;
      }
    } else if (nobs_set.count(type) != 0) {
      if (header.nobs_col == 0) {
        header.nobs_col = header.coln + 1;
      } else {
        writeln("error! more than two n_obs columns in the file.");
        n_error++;
      }
    } else if (ws_set.count(type) != 0) {
      if (header.ws_col == 0) {
        header.ws_col = header.coln + 1;
      } else {
        writeln("error! more than two window_size columns in the file.");
        n_error++;
      }
    } else if (af_set.count(type) != 0) {
      if (header.af_col == 0) {
        header.af_col = header.coln + 1;
      } else {
        writeln("error! more than two af columns in the file.");
        n_error++;
      }
    } else if (cor_set.count(type) != 0) {
      if (header.cor_col == 0) {
        header.cor_col = header.coln + 1;
      } else {
        writeln("error! more than two cor columns in the file.");
        n_error++;
      }
    } else {
    }

    ch_ptr = strtok(NULL, " , \t");
    header.coln++;
  }

  if (header.cor_col != 0 && header.cor_col != header.coln) {
    writeln("error! the cor column should be the last column.");
    n_error++;
  }

  if (header.rs_col == 0) {
    if (header.chr_col != 0 && header.pos_col != 0) {
      writeln("missing an rs column. rs id will be replaced by chr:pos");
    } else {
      writeln("error! missing an rs column.");
      n_error++;
    }
  }

  if (n_error == 0) {
    return true;
  } else {
    return false;
  }
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
        mat_S[0][0] += (1 - 1.0 / to!double(vec_n[ns_total]) * d2;
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
    d1 = qvar_vec.elements[i] + 2 * vec_qvar[i];
    qvar_vec.elements[i] = d1;
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


// Use the new method to calculate variance components with summary
// statistics first, use a function CalcS to compute S matrix (where
// the diagonal elements are part of V(q) ), and then use bootstrap to
// compute the variance for S, use a set of genotypes, phenotypes, and
// individual ids, and snp category label.
void CalcVCss(const DMatrix Vq, const DMatrix S_mat,
              const DMatrix Svar_mat, const DMatrix q_vec,
              const DMatrix s_vec, const double df, double[] v_pve,
              double[] v_se_pve, double pve_total, double se_pve_total,
              double[] v_sigma2, double[] v_se_sigma2,
              double[] v_enrich, double[] v_se_enrich) {
  size_t n_vc = S_mat.shape[0];

  DMatrix Si_mat; // = gsl_matrix_alloc(n_vc, n_vc);
  DMatrix Var_mat; // = gsl_matrix_alloc(n_vc, n_vc);
  DMatrix tmp_mat1; // = gsl_matrix_alloc(n_vc, n_vc);
  DMatrix VarEnrich_mat; // = gsl_matrix_alloc(n_vc, n_vc);
  DMatrix qvar_mat; // = gsl_matrix_alloc(n_vc, n_vc);

  DMatrix pve_plus; // = gsl_vector_alloc(n_vc + 1);
  DMatrix tmp; // = gsl_vector_alloc(n_vc + 1);
  DMatrix enrich; // = gsl_vector_alloc(n_vc);
  DMatrix se_pve; // = gsl_vector_alloc(n_vc);
  DMatrix se_sigma2persnp; // = gsl_vector_alloc(n_vc);
  DMatrix se_enrich; // = gsl_vector_alloc(n_vc);

  double d;

  // Calculate S^{-1}q.
  DMatrix Si_mat = S_mat.inverse();

  // Calculate sigma2snp and pve.
  pve = matrix_mult(Si_mat, q_vec);
  DMatrix sigma2persnp; //gsl_vector_memcpy(sigma2persnp, pve);
  sigma2persnp = divide_dmatrix(sigma2persnp, s_vec);

  // Get qvar_mat.
  //gsl_matrix_memcpy(qvar_mat, Vq);
  qvar_mat = slow_matrix_multiply(qvar_mat, 1.0 / (df * df));

  // Calculate variance for these estimates.
  for (size_t i = 0; i < n_vc; i++) {
    for (size_t j = i; j < n_vc; j++) {
      d = Svar_mat.accessor(i, j);
      d *= pve.elements[i] * pve.elements[j];

      d += qvar_mat.accessor(i, j);
      Var_mat.set(i, j, d);
      if (i != j) {
        Var_mat.set(j, i, d);
      }
    }
  }

  DMatrix tmp_mat = matrix_mult(Si_mat, Var_mat);
  DMatrix Var_mat = matrix_mult(tmp_mat, Si_mat);

  for (size_t i = 0; i < n_vc; i++) {
    d = sqrt(Var_mat.accessor(i, i));
    se_pve.elements[i] = d;
    d /= s_vec.elements[i];
    se_sigma2persnp.elements[i] = d;
  }

  // Compute pve_total, se_pve_total.
  pve_total = 0;
  se_pve_total = 0;
  for (size_t i = 0; i < n_vc; i++) {
    pve_total += pve.elements[i];

    for (size_t j = 0; j < n_vc; j++) {
      se_pve_total += gsl_matrix_get(Var_mat, i, j);
    }
  }
  se_pve_total = sqrt(se_pve_total);

  // Compute enrichment and its variance.
  double s_pve = 0, s_snp = 0;
  for (size_t i = 0; i < n_vc; i++) {
    s_pve += gsl_vector_get(pve, i);
    s_snp += gsl_vector_get(s_vec, i);
  }
  gsl_vector_memcpy(enrich, sigma2persnp);
  gsl_vector_scale(enrich, s_snp / s_pve);

  //gsl_matrix_set_identity(tmp_mat);
  DMatrix tmp_mat = ones_dmatrix(tmp_mat.shape[0], tmp_mat.shape[1])

  double d1;
  for (size_t i = 0; i < n_vc; i++) {
    d = pve.elements[i] / s_pve;
    d1 = s_vec.elements[i];
    for (size_t j = 0; j < n_vc; j++) {
      if (i == j) {
        tmp_mat.set(i, j, (1 - d) / d1 * s_snp / s_pve);
      } else {
        tmp_mat.set(i, j, -1 * d / d1 * s_snp / s_pve);
      }
    }
  }
  tmp_mat1 = matrix_mult(tmp_mat, Var_mat);
  VarEnrich_mat = matrix_mult(tmp_mat1, tmp_mat.T);

  for (size_t i = 0; i < n_vc; i++) {
    d = sqrt(VarEnrich_mat.accessor(i, i));
    se_enrich.elements[i] = d;
  }

  writeln("pve = ");
  for (size_t i = 0; i < n_vc; i++) {
    write(pve.elements[i], " ");
  }
  write("\n");

  write("se(pve) = ")s
  for (size_t i = 0; i < n_vc; i++) {
    write(se_pve.elements[i], " ");
  }
  write("\n");

  write("sigma2 per snp = ");
  for (size_t i = 0; i < n_vc; i++) {
    write(sigma2persnp.elements[i], " ");
  }
  write("\n");

  write("se(sigma2 per snp) = ");
  for (size_t i = 0; i < n_vc; i++) {
    write(se_sigma2persnp.elements[i], " ");
  }
  write("\n");

  write("enrichment = ");
  for (size_t i = 0; i < n_vc; i++) {
    write(enrich.elements[i], " ");
  }
  write("\n");

  write("se(enrichment) = ");
  for (size_t i = 0; i < n_vc; i++) {
    write(se_enrich.elements[i], " ");
  }
  write("\n");

  // Save data.
  v_pve = [];
  v_se_pve = [];
  v_sigma2 = [];
  v_se_sigma2 = [];
  v_enrich = [];
  v_se_enrich = [];
  for (size_t i = 0; i < n_vc; i++) {
    d = pve.elements[i];
    v_pve ~= d;
    d = se_pve.elements[i];
    v_se_pve ~= d;

    d = sigma2persnp.elements[i];
    v_sigma2 ~= d;
    d = se_sigma2persnp.elements[i];
    v_se_sigma2 ~= d;

    d = enrich.elements[i];
    v_enrich ~= d;
    d = se_enrich.elements[i];
    v_se_enrich ~= d;
  }

  return;
}

// Ks are not scaled.
void CalcVChe(const DMatrix K, const DMatrix W,
                  const DMatrix y) {
  size_t n1 = K.shape[0], n2 = K.shape[1];
  size_t n_vc = n2 / n1;

  double r = to!double(n1) / to!double(n1 - W->size2);
  double var_y, var_y_new;
  double d, tr, s, v;
  double[] traceG_new;

  // New matrices/vectors.
  DMatrix K_scale; // = gsl_matrix_alloc(n1, n2);
  DMatrix y_scale; // = gsl_vector_alloc(n1);
  DMatrix Kry; // = gsl_matrix_alloc(n1, n_vc);
  DMatrix yKrKKry; // = gsl_matrix_alloc(n_vc, n_vc * (n_vc + 1));
  DMatrix KKry; // = gsl_vector_alloc(n1);

  // Old matrices/vectors.
  DMatrix pve; // = gsl_vector_alloc(n_vc);
  DMatrix se_pve; // = gsl_vector_alloc(n_vc);
  DMatrix q_vec; // = gsl_vector_alloc(n_vc);
  DMatrix qvar_mat; // = gsl_matrix_alloc(n_vc, n_vc);
  DMatrix tmp_mat; // = gsl_matrix_alloc(n_vc, n_vc);
  DMatrix S_mat; // = gsl_matrix_alloc(n_vc, n_vc);
  DMatrix Si_mat; // = gsl_matrix_alloc(n_vc, n_vc);
  DMatrix Var_mat; // = gsl_matrix_alloc(n_vc, n_vc);

  // Center and scale K by W.
  for (size_t i = 0; i < n_vc; i++) {
    gsl_matrix_view Kscale_sub =
        gsl_matrix_submatrix(K_scale, 0, n1 * i, n1, n1);
    gsl_matrix_const_view K_sub =
        gsl_matrix_const_submatrix(K, 0, n1 * i, n1, n1);
    gsl_matrix_memcpy(&Kscale_sub.matrix, &K_sub.matrix);

    CenterMatrix(&Kscale_sub.matrix, W);
    d = ScaleMatrix(&Kscale_sub.matrix);
    traceG_new.push_back(d);
  }

  // Center y by W, and standardize it to have variance 1 (t(y)%*%y/n=1).
  gsl_vector_memcpy(y_scale, y);
  CenterVector(y_scale, W);

  var_y = VectorVar(y);
  var_y_new = VectorVar(y_scale);

  StandardizeVector(y_scale);

  // Compute Kry, which is used for confidence interval; also compute
  // q_vec (*n^2).
  for (size_t i = 0; i < n_vc; i++) {
    //gsl_matrix_const_view
    DMatrix Kscale_sub = get_sub_dmatrix(K_scale, 0, n1 * i, n1, n1);
    //gsl_vector_view
    DMatrix Kry_col = get_col(Kry, i);

    //gsl_vector_memcpy(&Kry_col.vector, y_scale);
    // Note: there is a scaling factor
    //gsl_blas_dgemv(CblasNoTrans, 1.0, &Kscale_sub.matrix, y_scale, -1.0 * r, &Kry_col.vector);

    d = vector_ddot(Kry_col, y_scale);
    q_vec.elements[i] = d;
  }

  // Compute yKrKKry, which is used later for confidence interval.
  for (size_t i = 0; i < n_vc; i++) {
    //gsl_vector_const_view 
    DMatrix Kry_coli = get_col(Kry, i);
    for (size_t j = i; j < n_vc; j++) {
      //gsl_vector_const_view
      DMatrix Kry_colj = get_col(Kry, j);
      for (size_t l = 0; l < n_vc; l++) {
        //gsl_matrix_const_view
        DMatrix Kscale_sub = get_sub_dmatrix(K_scale, 0, n1 * l, n1, n1);
        KKry = matrix_mult(Kscale_sub, Kry_coli);
        d = vector_ddot(Kry_colj, KKry);
        yKrKKry.set(i, l * n_vc + j, d);
        if (i != j) {
          yKrKKry.set(j, l * n_vc + i, d);
        }
      }
      d = vector_ddot(Kry_coli, Kry_colj);
      yKrKKry.set(i, n_vc * n_vc + j, d);
      if (i != j) {
        yKrKKry.set(j, n_vc * n_vc + i, d);
      }
    }
  }

  // Compute Sij (*n^2).
  for (size_t i = 0; i < n_vc; i++) {
    for (size_t j = i; j < n_vc; j++) {
      tr = 0;
      for (size_t l = 0; l < n1; l++) {
        //gsl_vector_const_view
        DMatrix Ki_col = get_col(K_scale, i * n1 + l);
        //gsl_vector_const_view
        DMatrix Kj_col = get_col(K_scale, j * n1 + l);
        d = vector_ddot(Ki_col, Kj_col);
        tr += d;
      }

      tr = tr - r * (double)n1;
      S_mat.set(i, j, tr);
      if (i != j) {
        S_mat.set(j, i, tr);
      }
    }
  }

  // Compute S^{-1}q.
  DMatrix Si_mat = S_mat.inverse();

  // Compute pve (on the transformed scale).
  pve = matrix_mult(Si_mat, q_vec);

  // Compute q_var (*n^4).
  qvar_mat = zeros_dmatrix(q_var.shape[0], q_var.shape[1]);
  s = 1;
  for (size_t i = 0; i < n_vc; i++) {
    d = pve.elements[i];
    //gsl_matrix_view
    DMatrix yKrKKry_sub = get_sub_dmatrix(yKrKKry, 0, i * n_vc, n_vc, n_vc);
    //gsl_matrix_memcpy(tmp_mat, &yKrKKry_sub.matrix);
    tmp_mat = multiply_dmatrix_num(tmp_mat, d);
    qvar_mat += tmp_mat;
    s -= d;
  }
  //gsl_matrix_view 
  DMatrix yKrKKry_sub = get_sub_dmatrix(yKrKKry, 0, n_vc * n_vc, n_vc, n_vc);
  //gsl_matrix_memcpy(tmp_mat, &yKrKKry_sub.matrix);
  tmp_mat = multiply_dmatrix_num(tmp_mat, s);
  qvar_mat += tmp_mat;

  qvar_mat = multiply_dmatrix_num(qvar_mat, 2.0);

  // Compute S^{-1}var_qS^{-1}.
  tmp_mat = matrix_mult(Si_mat, qvar_mat);
  Var_mat = matrix_mult(tmp_mat, Si_mat);

  // Transform pve back to the original scale and save data.
  v_pve = [];
  v_se_pve = [];
  v_sigma2 = [];
  v_se_sigma2 = [];

  s = 1.0, v = 0, pve_total = 0, se_pve_total = 0;
  for (size_t i = 0; i < n_vc; i++) {
    d = pve.elements[i];
    v_sigma2 ~= d * var_y_new / traceG_new[i];
    v_pve ~= d * (var_y_new / traceG_new[i]) * (v_traceG[i] / var_y);
    s -= d;
    pve_total += d * (var_y_new / traceG_new[i]) * (v_traceG[i] / var_y);

    d = sqrt(Var_mat.get(i, i));
    v_se_sigma2 ~= d * var_y_new / traceG_new[i];
    v_se_pve ~= d * (var_y_new / traceG_new[i]) * (v_traceG[i] / var_y);

    for (size_t j = 0; j < n_vc; j++) {
      v += Var_mat.get(i, j);
      se_pve_total += Var_mat.get(i, j) *
                      (var_y_new / traceG_new[i]) * (v_traceG[i] / var_y) *
                      (var_y_new / traceG_new[j]) * (v_traceG[j] / var_y);
    }
  }
  v_sigma2 ~= s * r * var_y_new;
  v_se_sigma2 ~= sqrt(v) * r * var_y_new;
  se_pve_total = sqrt(se_pve_total);

  write("sigma2 = ");
  for (size_t i = 0; i < n_vc + 1; i++) {
    write(v_sigma2[i], " ");
  }
  write("\n");

  cout << "se(sigma2) = ";
  for (size_t i = 0; i < n_vc + 1; i++) {
    write(v_se_sigma2[i], " ");
  }
  write("\n");
  
  cout << "pve = ";
  for (size_t i = 0; i < n_vc; i++) {
    write(v_pve[i], " ");
  }
  write("\n");

  cout << "se(pve) = ";
  for (size_t i = 0; i < n_vc; i++) {
    write(v_se_pve[i], " ");
  }
  write("\n");

  if (n_vc > 1) {
    writeln("total pve = ", pve_total);
    writeln("se(total pve) = ", se_pve_total);
  }

  return;
}

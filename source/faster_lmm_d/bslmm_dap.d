module faster_lmm_d.bslmm_dap;

import std.algorithm: min, max, reduce, countUntil;
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

import faster_lmm_d.bslmm;
import faster_lmm_d.dmatrix;
import faster_lmm_d.gemma_io;
import faster_lmm_d.gemma_param;
import faster_lmm_d.logistic;
import faster_lmm_d.optmatrix;

// Read hyp file.
void ReadFile_hyb(const string file_hyp, ref double[] vec_sa2,
                  ref double[] vec_sb2,  ref double[] vec_wab) {
  vec_sa2 = [];
  vec_sb2 = [];
  vec_wab = [];

  File infile = File(file_hyp);

  foreach(line; infile.byLine){
    auto chr = line.split("\t");
    vec_sa2 ~= to!double(chr[2]);
    vec_sb2 ~= to!double(chr[3]);
    vec_wab ~= to!double(chr[4]);
  }

  return;
}

// Read bf file.
void ReadFile_bf(const string file_bf, string[] vec_rs, double[][][] BF) {
  //BF.clear();
  vec_rs = [];

  File infile = File(file_bf);

  string rs, block;
  double[] vec_bf;
  double[][] mat_bf;

  size_t bf_size = 0, flag_block;

  size_t t = 0;
  foreach(line; infile.byLine){
    flag_block = 0;

    auto ch_ptr = line.split(" \t");
    vec_rs ~= to!string(ch_ptr[0]);

    if (t == 0) {
      block = to!string(ch_ptr[1]);
    } else {
      if (ch_ptr[1] != block) {
        flag_block = 1;
        block = to!string(ch_ptr[1]);
      }
    }

    //TODO : CHECK
    foreach(chr; ch_ptr[2..$])
      vec_bf ~= to!double(chr);

    if (t == 0) {
      bf_size = vec_bf.length;
    } else {
      if (bf_size != vec_bf.length) {
        writeln("error! unequal row size in bf file.");
      }
    }

    if (flag_block == 0) {
      mat_bf ~= vec_bf;
    } else {
      BF ~= mat_bf;
      mat_bf = [];
    }
    vec_bf = [];

    t++;
  }

  return;
}

// Read category files.
// Read both continuous and discrete category file, record mapRS2catc.

struct MapRS2catc_t{
  string rs;
  double[] cat;
}

struct MapRS2catd_t{
  string rs;
  int[] cat;
}

void ReadFile_cat(const string file_cat, const string[] vec_rs,
                  ref DMatrix Ac, ref DMatrix_int Ad, ref DMatrix_int dlevel,
                  ref size_t kc, ref size_t kd) {
  File infile = File(file_cat);

  string rs, chr, a1, a0, pos, cm;

  // Read header.
  HEADER header;
  string header_line = infile.readln();
  ReadHeader_io(header_line, header);

  // Use the header to determine the number of categories.
  kc = header.catc_col.length;
  kd = header.catd_col.length;

  // set up storage and mapper
  MapRS2catc_t[] mapRS2catc;
  MapRS2catd_t[] mapRS2catd;
  double[] catc;
  int[] catd;

  // Read the following lines to record mapRS2cat.
  foreach(line; infile.byLine){
    auto ch_ptr = line.split(" \t");

    if (header.rs_col == 0) {
      rs = chr ~ ":" ~ pos;
    }

    catc = [];
    catd = [];

    for (size_t i = 0; i < header.coln; i++) {
      if (header.rs_col != 0 && header.rs_col == i + 1) {
        rs = to!string(ch_ptr[i]);
      } else if (header.chr_col != 0 && header.chr_col == i + 1) {
        chr = to!string(ch_ptr[i]);
      } else if (header.pos_col != 0 && header.pos_col == i + 1) {
        pos = to!string(ch_ptr[i]);
      } else if (header.cm_col != 0 && header.cm_col == i + 1) {
        cm = to!string(ch_ptr[i]);
      } else if (header.a1_col != 0 && header.a1_col == i + 1) {
        a1 = to!string(ch_ptr[i]);
      } else if (header.a0_col != 0 && header.a0_col == i + 1) {
        a0 = to!string(ch_ptr[i]);
      } else if (header.catc_col.length != 0 &&
                 header.catc_col.count(i + 1) != 0) {
        catc ~= to!double(ch_ptr[i]);
      } else if (header.catd_col.length != 0 &&
                 header.catd_col.count(i + 1) != 0) {
        catd ~= to!int(ch_ptr[i]);
      } else {
      }
    }

    //if (mapRS2catc.count(rs) == 0 && kc > 0) {
    //  mapRS2catc[rs] = catc;
    //}
    //if (mapRS2catd.count(rs) == 0 && kd > 0) {
    //  mapRS2catd[rs] = catd;
    //}
  }

  // Load into Ad and Ac.
  if (kc > 0) {
    Ac = zeros_dmatrix(vec_rs.length, kc);
    for (size_t i = 0; i < vec_rs.length; i++) {
      //if (mapRS2catc.count(vec_rs[i]) != 0) {
      //  for (size_t j = 0; j < kc; j++) {
      //    Ac.set(i, j, mapRS2catc[vec_rs[i]][j]);
      //  }
      //} else {
      //  for (size_t j = 0; j < kc; j++) {
      //    Ac.set(i, j, 0);
      //  }
      //}
    }
  }

  if (kd > 0) {
    Ad = zeros_dmatrix_int(vec_rs.length, kd);

    for (size_t i = 0; i < vec_rs.length; i++) {
      //if (mapRS2catd.count(vec_rs[i]) != 0) {
      //  for (size_t j = 0; j < kd; j++) {
      //    Ad.set(i, j, mapRS2catd[vec_rs[i]][j]);
      //  }
      //} else {
      //  for (size_t j = 0; j < kd; j++) {
      //    Ad.set(i, j, 0);
      //  }
      //}
    }

    //dlevel = gsl_vector_int_alloc(kd);
    int[int] rcd;
    int val;
    for (size_t j = 0; j < kd; j++) {
      rcd.clear();
      for (size_t i = 0; i < Ad.shape[0]; i++) {
        val = Ad.accessor(i, j);
        rcd[val] = 1;
      }
      dlevel.elements[j] = to!int(rcd.length);
    }
  }

  return;
}

void WriteResult(const DMatrix Hyper, const DMatrix BF) {
  string file_bf, file_hyp, path_out, file_out;
  int[] indicator_snp; // TODO
  size_t ns_total;
  SNPINFO[] snpInfo;
  file_bf = path_out ~ "/" ~ file_out;
  file_bf ~= ".bf.txt";
  file_hyp = path_out ~ "/" ~ file_out;
  file_hyp ~= ".hyp.txt";

  File outfile_bf = File(file_bf);
  File outfile_hyp = File(file_hyp);

  outfile_hyp.writeln("h \t
                    rho \t
                    sa2 \t
                    sb2 \t
                    weight");
  for (size_t i = 0; i < Hyper.shape[0]; i++) {
    for (size_t j = 0; j < Hyper.shape[1]; j++) {
      outfile_hyp.write(Hyper.accessor(i, j), "\t");
    }
    outfile_hyp.write("\n");
  }

  outfile_bf.writeln("chr \t
                    rs \t
                    ps \t
                    n_miss");
  for (size_t i = 0; i < BF.shape[1]; i++) {
    outfile_bf.write("\tBF",  i + 1);
  }
  outfile_bf.write("\n");

  size_t t = 0;
  for (size_t i = 0; i < ns_total; ++i) {
    if (indicator_snp[i] == 0) {
      continue;
    }

    outfile_bf.write(snpInfo[i].chr, "\t", snpInfo[i].rs_number, "\t",
                     snpInfo[i].base_position, "\t", snpInfo[i].n_miss);

    for (size_t j = 0; j < BF.shape[1]; j++) {
      outfile_bf.write("\t", BF.accessor(t, j));
    }
    outfile_bf.write("\n");

    t++;
  }

  return;
}

void WriteResult(const string[] vec_rs,
                 const DMatrix Hyper, const DMatrix pip,
                 const DMatrix coef) {
  string file_gamma, file_hyp, file_coef, path_out, file_out;
  file_gamma = path_out ~ "/" ~ file_out;
  file_gamma ~= ".gamma.txt";
  file_hyp = path_out ~ "/" ~ file_out;
  file_hyp ~= ".hyp.txt";
  file_coef = path_out ~ "/" ~ file_out;
  file_coef ~= ".coef.txt";

  File outfile_gamma = File(file_gamma);
  File outfile_hyp = File(file_hyp);
  File outfile_coef = File(file_coef);

  outfile_hyp.writeln("h \t 
                     rho \t
                     sa2 \t
                     sb2 \t
                     weight");
  for (size_t i = 0; i < Hyper.shape[0]; i++) {
    for (size_t j = 0; j < Hyper.shape[1]; j++) {
      outfile_hyp.write(Hyper.accessor(i, j), "\t");
    }
    outfile_hyp,write("\n");
  }

  outfile_gamma.write("rs \t gamma");
  for (size_t i = 0; i < vec_rs.length; ++i) {
    outfile_gamma.writeln(vec_rs[i], "\t", pip.elements[i]);
  }

  outfile_coef.writeln("coef");
  for (size_t i = 0; i < coef.size; i++) {
    outfile_coef.writeln(coef.elements[i]);
  }

  return;
}

double CalcMarginal(const DMatrix Uty, const DMatrix K_eval,
                    const double sigma_b2, const double tau) {
  DMatrix weight_Hi = zeros_dmatrix(1, Uty.size);
  size_t ni_test;

  double logm = 0.0;
  double d, uy, Hi_yy = 0, logdet_H = 0.0;
  for (size_t i = 0; i < ni_test; ++i) {
    d = K_eval.elements[i] * sigma_b2;
    d = 1.0 / (d + 1.0);
    weight_Hi.elements[i] = d;

    logdet_H -= mlog(d);
    uy = Uty.elements[i];
    Hi_yy += d * uy * uy;
  }

  // Calculate likelihood.
  logm = -0.5 * logdet_H - 0.5 * tau * Hi_yy + 0.5 * mlog(tau) * to!double(ni_test);

  return logm;
}

double CalcMarginal(const DMatrix UtXgamma, const DMatrix Uty,
                    const DMatrix K_eval, const double sigma_a2,
                    const double sigma_b2, const double tau) {
  size_t ni_test;
  double logm = 0.0;
  double d, uy, P_yy = 0, logdet_O = 0.0, logdet_H = 0.0;

  DMatrix UtXgamma_eval = zeros_dmatrix(UtXgamma.shape[0], UtXgamma.shape[1]);
  DMatrix Omega = zeros_dmatrix(UtXgamma.shape[1], UtXgamma.shape[1]);
  DMatrix XtHiy = zeros_dmatrix(1, UtXgamma.shape[1]);
  DMatrix beta_hat = zeros_dmatrix(1, UtXgamma.shape[1]);
  DMatrix weight_Hi = zeros_dmatrix(1, UtXgamma.shape[0]);

  UtXgamma_eval = UtXgamma.dup_dmatrix;

  logdet_H = 0.0;
  P_yy = 0.0;
  for (size_t i = 0; i < ni_test; ++i) {
    //gsl_vector_view 
    DMatrix UtXgamma_row = get_row(UtXgamma_eval, i);
    d = K_eval.elements[i] * sigma_b2;
    d = 1.0 / (d + 1.0);
    weight_Hi.elements[i] = d;

    logdet_H -= mlog(d);
    uy = Uty.elements[i];
    P_yy += d * uy * uy;
    UtXgamma_row = multiply_dmatrix_num(UtXgamma_row, d);
  }

  // Calculate Omega.
  Omega = identity_dmatrix(Omega.shape[0], Omega.shape[1]);

  //lapack_dgemm((char *)"T", (char *)"N", sigma_a2, UtXgamma_eval, UtXgamma, 1.0, Omega);

  // Calculate beta_hat.
  XtHiy = matrix_mult(UtXgamma_eval.T, Uty);

  //logdet_O = CholeskySolve(Omega, XtHiy, beta_hat);

  beta_hat = multiply_dmatrix_num(beta_hat, sigma_a2);

  d = vector_ddot(XtHiy, beta_hat);
  P_yy -= d;

  logm = -0.5 * logdet_H - 0.5 * logdet_O - 0.5 * tau * P_yy +
         0.5 * mlog(tau) * to!double(ni_test);

  return logm;
}

double CalcPrior(HYPBSLMM cHyp) {
  size_t ns_test;
  double logprior = 0;
  logprior = (to!double(cHyp.n_gamma) - 1.0) * cHyp.logp +
      (to!double(ns_test) - to!double(cHyp.n_gamma)) * mlog(1.0 - exp(cHyp.logp));
  return logprior;
}

// Where A is the ni_test by n_cat matrix of annotations.
void DAP_CalcBF(const DMatrix U, const DMatrix UtX,
                const DMatrix Uty, const DMatrix K_eval,
                const DMatrix y) {
  double trace_G; // TODO
  size_t h_ngrid, rho_ngrid, ns_test, ni_test;
  double h_min, h_max, rho_min, rho_max;
  // Set up BF.
  double tau, h, rho, sigma_a2, sigma_b2, d;
  size_t ns_causal = 10;
  size_t n_grid = h_ngrid * rho_ngrid;
  double[] vec_sa2, vec_sb2, logm_null;

  DMatrix BF = zeros_dmatrix(ns_test, n_grid);
  DMatrix Xgamma = zeros_dmatrix(ni_test, 1);
  DMatrix Hyper = zeros_dmatrix(n_grid, 5);

  // Compute tau by using yty.
  tau = vector_ddot(Uty, Uty);
  tau = to!double(ni_test) / tau;

  // Set up grid values for sigma_a2 and sigma_b2 based on an
  // approximately even grid for h and rho, and a fixed number
  // of causals.
  size_t ij = 0;
  for (size_t i = 0; i < h_ngrid; i++) {
    h = h_min + (h_max - h_min) * to!double(i) / (to!double(h_ngrid) - 1);
    for (size_t j = 0; j < rho_ngrid; j++) {
      rho = rho_min + (rho_max - rho_min) * to!double(j) / (to!double(rho_ngrid) - 1);

      sigma_a2 = h * rho / ((1 - h) * to!double(ns_causal));
      sigma_b2 = h * (1.0 - rho) / (trace_G * (1 - h));

      vec_sa2 ~= sigma_a2;
      vec_sb2 ~= sigma_b2;
      logm_null ~= CalcMarginal(Uty, K_eval, 0.0, tau);

      Hyper.set(ij, 0, h);
      Hyper.set(ij, 1, rho);
      Hyper.set(ij, 2, sigma_a2);
      Hyper.set(ij, 3, sigma_b2);
      Hyper.set(ij, 4, 1 / to!double(n_grid));
      ij++;
    }
  }

  // Compute BF factors.
  writeln("Calculating BF...");
  for (size_t t = 0; t < ns_test; t++) {
    //gsl_vector_view
    DMatrix Xgamma_col = get_col(Xgamma, 0);
    //gsl_vector_const_view 
    DMatrix X_col = get_col(UtX, t);
    Xgamma_col =X_col.dup_dmatrix;

    for (size_t k = 0; k < n_grid; k++) {
      sigma_a2 = vec_sa2[k];
      sigma_b2 = vec_sb2[k];

      d = CalcMarginal(Xgamma, Uty, K_eval, sigma_a2, sigma_b2, tau);
      d -= logm_null[k];
      d = exp(d);

      BF.set(t, k, d);
    }
  }

  // Save results.
  WriteResult(Hyper, BF);

  return;
}

void single_ct_regression(const DMatrix_int Xd,
                          const DMatrix_int dlevel,
                          const DMatrix pip_vec, ref DMatrix coef,
                          DMatrix prior_vec) {

  double[int] sum_pip;
  double[int] sum;

  int levels = dlevel.elements[0];

  for (int i = 0; i < levels; i++) {
    sum_pip[i] = sum[i] = 0;
  }

  for (size_t i = 0; i < Xd.shape[0]; i++) {
    int cat = Xd.accessor(i, 0);
    sum_pip[cat] += pip_vec.elements[i];
    sum[cat] += 1;
  }

  for (size_t i = 0; i < Xd.shape[0]; i++) {
    int cat = Xd.accessor(i, 0);
    prior_vec.elements[i] = sum_pip[cat] / sum[cat];
  }

  for (int i = 0; i < levels; i++) {
    double new_prior = sum_pip[i] / sum[i];
    coef.elements[i] = mlog(new_prior / (1 - new_prior));
  }

  return;
}

// Where A is the ni_test by n_cat matrix of annotations.
void DAP_EstimateHyper(
    const size_t kc, const size_t kd, const string[] vec_rs,
    const double[] vec_sa2, const double[] vec_sb2,
    const double[] wab, const double[][][] BF,
    ref DMatrix Ac, ref DMatrix_int Ad, ref DMatrix_int dlevel) {
  // clock_t time_start;

  // Set up BF.
  double h, rho, sigma_a2, sigma_b2, d, s, logm, logm_save;
  size_t t1, t2;
  size_t n_grid = wab.length, ns_test = vec_rs.length;

  DMatrix prior_vec = zeros_dmatrix(1, ns_test);
  DMatrix Hyper = zeros_dmatrix(n_grid, 5);
  DMatrix pip = zeros_dmatrix(1, ns_test);
  DMatrix coef = zeros_dmatrix(1, kc + kd + 1);

  // Perform the EM algorithm.
  double[] vec_wab, vec_wab_new;

  // Initial values.
  for (size_t t = 0; t < ns_test; t++) {
    prior_vec.elements[t] = to!double(BF.length) / to!double(ns_test);
  }
  for (size_t ij = 0; ij < n_grid; ij++) {
    vec_wab ~= wab[ij];
    vec_wab_new ~= wab[ij];
  }

  // EM iteration.
  size_t it = 0;
  double dif = 1;
  while (it < 100 && dif > 1e-3) {

    // Update E_gamma.
    t1 = 0, t2 = 0;
    for (size_t b = 0; b < BF.length; b++) {
      s = 1;
      for (size_t m = 0; m < BF[b].length; m++) {
        d = 0;
        for (size_t ij = 0; ij < n_grid; ij++) {
          d += vec_wab_new[ij] * BF[b][m][ij];
        }
        d *= prior_vec.elements[t1] / (1 - prior_vec.elements[t1]);

        pip.elements[t1] = d;
        s += d;
        t1++;
      }

      for (size_t m = 0; m < BF[b].length; m++) {
        d = pip.elements[t2] / s;
        pip.elements[t2] = d;
        t2++;
      }
    }

    // Update E_wab.
    s = 0;
    for (size_t ij = 0; ij < n_grid; ij++) {
      vec_wab_new[ij] = 0;

      t1 = 0;
      for (size_t b = 0; b < BF.length; b++) {
        d = 1;
        for (size_t m = 0; m < BF[b].length; m++) {
          d += prior_vec.elements[t1] / (1 - prior_vec.elements[t1]) * vec_wab[ij] * BF[b][m][ij];
          t1++;
        }
        vec_wab_new[ij] += mlog(d);
      }

      s = max(s, vec_wab_new[ij]);
    }

    d = 0;
    for (size_t ij = 0; ij < n_grid; ij++) {
      vec_wab_new[ij] = exp(vec_wab_new[ij] - s);
      d += vec_wab_new[ij];
    }

    for (size_t ij = 0; ij < n_grid; ij++) {
      vec_wab_new[ij] /= d;
    }

    // Update coef, and pi.
    if (kc == 0 && kd == 0) {

      // No annotation.
      s = 0;
      for (size_t t = 0; t < pip.size; t++) {
        s += pip.elements[t1];
      }
      s = s / to!double(pip.size);
      for (size_t t = 0; t < pip.size; t++) {
        prior_vec.elements[t] = s;
      }

      coef.elements[0] = mlog(s / (1 - s));
    } else if (kc == 0 && kd != 0) {

      // Only discrete annotations.
      if (kd == 1) {
        single_ct_regression(Ad, dlevel, pip, coef, prior_vec);
      } else {
        logistic_cat_fit(coef, Ad, dlevel, pip, 0, 0);
        logistic_cat_pred(coef, Ad, dlevel, prior_vec);
      }
    } else if (kc != 0 && kd == 0) {

      // Only continuous annotations.
      logistic_cont_fit(coef, Ac, pip, 0, 0);
      logistic_cont_pred(coef, Ac, prior_vec);
    } else if (kc != 0 && kd != 0) {

      // Both continuous and categorical annotations.
      logistic_mixed_fit(coef, Ad, dlevel, Ac, pip, 0, 0);
      logistic_mixed_pred(coef, Ad, dlevel, Ac, prior_vec);
    }

    // Compute marginal likelihood.
    logm = 0;

    t1 = 0;
    for (size_t b = 0; b < BF.length; b++) {
      d = 1;
      s = 0;
      for (size_t m = 0; m < BF[b].length; m++) {
        s += mlog(1 - prior_vec.elements[t1]);
        for (size_t ij = 0; ij < n_grid; ij++) {
          d += prior_vec.elements[t1] /
               (1 - prior_vec.elements[t1]) * vec_wab[ij] * BF[b][m][ij];
        }
      }
      logm += mlog(d) + s;
      t1++;
    }

    if (it > 0) {
      dif = logm - logm_save;
    }
    logm_save = logm;
    it++;

    writeln("iteration = ", it, "; marginal likelihood = ", logm);
  }

  // Update h and rho that correspond to w_ab.
  for (size_t ij = 0; ij < n_grid; ij++) {
    sigma_a2 = vec_sa2[ij];
    sigma_b2 = vec_sb2[ij];

    d = exp(coef.elements[coef.size] - 1) /
        (1 + exp(coef.elements[coef.size] - 1));
    h = (d * to!double(ns_test) * sigma_a2 + 1 * sigma_b2) /
        (1 + d * to!double(ns_test) * sigma_a2 + 1 * sigma_b2);
    rho = d * to!double(ns_test) * sigma_a2 /
          (d * to!double(ns_test) * sigma_a2 + 1 * sigma_b2);

    Hyper.set(ij, 0, h);
    Hyper.set(ij, 1, rho);
    Hyper.set(ij, 2, sigma_a2);
    Hyper.set(ij, 3, sigma_b2);
    Hyper.set(ij, 4, vec_wab_new[ij]);
  }

  // Obtain beta and alpha parameters.

  // Save results.
  WriteResult(vec_rs, Hyper, pip, coef);

  // Free matrices and vectors.
  return;
}

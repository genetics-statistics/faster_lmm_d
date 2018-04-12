/*
   This code is part of faster_lmm_d and published under the GPLv3
   License (see LICENSE.txt)

   Copyright © 2017 - 2018 Prasun Anand & Pjotr Prins
*/

module faster_lmm_d.gemma_param;

import faster_lmm_d.dmatrix;
import std.conv;
import std.exception;
import std.file;
import std.math;
import std.stdio;
import std.algorithm: min, max, reduce;
alias mlog = std.math.log;


struct SNPINFO{
  double cM;
  string chr;
  double maf;
  size_t n_nb;          // Number of neighbours on the right hand side.
  size_t n_idv;         // Number of non-missing individuals.
  size_t n_miss;
  string a_minor;
  string a_major;
  string rs_number;
  double missingness;
  long   base_position;
  size_t file_position; // SNP location in file.

  this(string chr, string rs_number, double cM, long base_position, string a_minor,
        string a_major, size_t n_miss, double missingness, double maf, size_t n_idv,
        size_t n_nb, size_t file_position){
    this.cM            = cM;
    this.chr           = chr;
    this.maf           = maf;
    this.n_nb          = n_nb;
    this.n_idv         = n_idv;
    this.n_miss        = n_miss;
    this.a_minor       = a_minor;
    this.a_major       = a_major;
    this.rs_number     = rs_number;
    this.missingness   = missingness;
    this.base_position = base_position;
    this.file_position = file_position;
  }
}

// Results for LMM.
struct SUMSTAT {
  double beta;         // REML estimator for beta.
  double se;           // SE for beta.
  double lambda_remle; // REML estimator for lambda.
  double p_wald;       // p value from a Wald test.
  string indicator;

  this(double beta, double se, double lambda_remle, double p_wald, string indicator){
    this.beta = beta;
    this.se = se;
    this.lambda_remle = lambda_remle;
    this.p_wald = p_wald;
    this.indicator = indicator;
  }
};

struct Param{
  bool mode_check = true;   // run data checks (slower)
  bool mode_strict = false; // exit on some data checks
  bool mode_silence;
  bool mode_debug = false;
  bool faster_lmm_d = false;
  uint issue; // enable tests for issue on github tracker

  int a_mode; // Analysis mode, 1/2/3/4 for Frequentist tests
  int k_mode; // Kinship read mode: 1: n by n matrix, 2: id/id/k_value;
  //vector<size_t> p_column; // Which phenotype column needs analysis.
  size_t d_pace;           // Display pace

  string file_bfile, file_mbfile;
  string file_geno, file_mgeno;
  string file_pheno;

  // LMM-related parameters.
  string loco;
  double l_min;
  double l_max;
  size_t n_region;
  double l_mle_null, l_remle_null;
  double logl_mle_H0, logl_remle_H0;
  double pve_null, pve_se_null, pve_total, se_pve_total;
  double vg_remle_null, ve_remle_null, vg_mle_null, ve_mle_null;

  double trace_G;

  bool error;

  // Number of individuals.
  size_t ni_total, ni_test, ni_cvt, ni_study, ni_ref;
  size_t ni_max = 0; // -nind switch for testing purposes

  size_t ng_total, ng_test;   // Number of genes.
  size_t ni_control, ni_case; // Number of controls and number of cases.
  size_t ni_subsample;        // Number of subsampled individuals.
  size_t n_cvt;               // Number of covariates.
  size_t n_cat;               // Number of continuous categories.
  size_t n_ph;                // Number of phenotypes.
  size_t n_vc;                // Number of variance components
                              // (including the diagonal matrix).

  // Indicator for individuals (phenotypes): 0 missing, 1
  // available for analysis
  string indicator_idv_file;

  // Sequence indicator for SNPs: 0 ignored because of (a) maf,
  // (b) miss, (c) non-poly; 1 available for analysis.
  string indicator_snp_file;

  void ReadFiles(){}
  void CheckData(){}
  void WriteVector(DMatrix a, string b){}

}

struct mapRS{
  void clear(){}
}

void check_data() {

  int a_mode, n_vc, d_pace, DEFAULT_PACE;
  int[] indicator_cvt, indicator_gxe, indicator_weight, indicator_idv, indicator_read, indicator_bv;
  int[][] indicator_pheno;
  string file_beta, file_cor, file_gene, file_study, file_mstudy, file_position, file_bf, file_mk, file_epm, loco;
  size_t ni_test, ni_cvt, ni_total, n_cvt, np_obs, np_miss, ni_case, ni_control, ns_test, ns_total,n_mh, n_ph, ng_total;
  double[] v_pve;
  double[][] pheno;
  double h_min, h_max, s_min, s_max, logp_scale, logp_min, h_scale, rho_scale;
  mapRS[] mapRS2wcat;

  bool error;

  if ((a_mode == 66 || a_mode == 67) && (v_pve.length != n_vc)) {
    writeln("error! the number of pve estimates does not equal to
            the number of categories in the cat file:", v_pve.length,
            " ", n_vc);
    error = true;
  }

  if (indicator_cvt.length != 0 &&
      indicator_cvt.length != indicator_idv.length) {
    error = true;
    writeln("error! number of rows in the covariates file do not
             match the number of individuals. ", indicator_cvt.length);
    return;
  }
  if (indicator_gxe.length != 0 &&
      indicator_gxe.length != indicator_idv.length) {
    error = true;
    writeln("error! number of rows in the gxe file do not match the number
             of individuals. ");
    return;
  }
  if (indicator_weight.length != 0 &&
      indicator_weight.length != indicator_idv.length) {
    error = true;
    writeln("error! number of rows in the weight file do not match
             the number of individuals. ");
    return;
  }

  if (indicator_read.length != 0 &&
      indicator_read.length != indicator_idv.length) {
    error = true;
    writeln("error! number of rows in the total read file do not
             match the number of individuals. ");
    return;
  }

  // Calculate ni_total and ni_test, and set indicator_idv to 0
  // whenever indicator_cvt=0, and calculate np_obs and np_miss.
  ni_total = indicator_idv.length;

  ni_test = 0;
  foreach(idv;  indicator_idv) {
    if (idv == 0) {
      continue;
    }
    ni_test++;
  }

  ni_cvt = 0;
  foreach (cvt; indicator_cvt) {
    if (cvt == 0) {
      continue;
    }
    ni_cvt++;
  }

  np_obs = 0;
  np_miss = 0;
  for (size_t i = 0; i < indicator_pheno.length; i++) {
    if (indicator_cvt.length != 0) {
      if (indicator_cvt[i] == 0) {
        continue;
      }
    }

    if (indicator_gxe.length != 0) {
      if (indicator_gxe[i] == 0) {
        continue;
      }
    }

    if (indicator_weight.length != 0) {
      if (indicator_weight[i] == 0) {
        continue;
      }
    }

    for (size_t j = 0; j < indicator_pheno[i].length; j++) {
      if (indicator_pheno[i][j] == 0) {
        np_miss++;
      } else {
        np_obs++;
      }
    }
  }

  if (ni_test == 0 && file_cor == "" && file_mstudy == "" &&
      file_study == "" && file_beta == "" && file_bf == "") {
    error = true;
    writeln("error! number of analyzed individuals equals 0.");
    return;
  }

  if (a_mode == 43) {
    if (ni_cvt == ni_test) {
      error = true;
      writeln("error! no individual has missing phenotypes.");
      return;
    }
    if ((np_obs + np_miss) != (ni_cvt * n_ph)) {
      error = true;
      writeln("error! number of phenotypes do not match the
               summation of missing and observed phenotypes.");
      return;
    }
  }

  // Output some information.
  if (file_cor == "" && file_mstudy == "" && file_study == "" &&
      a_mode != 15 && a_mode != 27 && a_mode != 28) {
    writeln("## number of total individuals = ", ni_total);
    if (a_mode == 43) {
      writeln("## number of analyzed individuals = ", ni_cvt);
      writeln("## number of individuals with full phenotypes = ", ni_test);
    } else {
      writeln("## number of analyzed individuals = ", ni_test);
    }
    writeln("## number of covariates = ", n_cvt);
    writeln("## number of phenotypes = ", n_ph);
    if (a_mode == 43) {
      writeln("## number of observed data = ", np_obs);
      writeln("## number of missing data = ",  np_miss);
    }
    if (file_gene != "") {
      writeln("## number of total genes = ", ng_total);
    }
    else if (file_epm == "" && a_mode != 43 && a_mode != 5) {
      if (loco != "")
        writeln("## leave one chromosome out (LOCO) = ", loco);
      writeln("## number of total SNPs/var          = ", ns_total);
      // uncomment later
      //if (setSnps.length)
      //  writeln("## number of considered SNPS       = ", setSnps.length);
      //if (setKSnps.length)
      //  writeln("## number of SNPS for K            = ", setKSnps.length);
      //if (setGWASnps.length)
      //  writeln("## number of SNPS for GWAS         = ", setGWASnps.length);
      //  writeln("## number of analyzed SNPs         = ", ns_test);
      }
      else {}
  }

  // Set d_pace to 1000 for gene expression.
  if (file_gene != "" && d_pace == DEFAULT_PACE) {
    d_pace = 1000;
  }

  // For case-control studies, count # cases and # controls.
  int flag_cc = 0;
  if (a_mode == 13) {
    ni_case = 0;
    ni_control = 0;
    for (size_t i = 0; i < indicator_idv.length; i++) {
      if (indicator_idv[i] == 0) {
        continue;
      }

      if (pheno[i][0] == 0) {
        ni_control++;
      } else if (pheno[i][0] == 1) {
        ni_case++;
      } else {
        flag_cc = 1;
      }
    }
    writeln("## number of cases = ", ni_case);
    writeln("## number of controls = ", ni_control);
  }

  if (flag_cc == 1) {
    writeln("Unexpected non-binary phenotypes for
            case/control analysis. Use default (BSLMM) analysis instead.");
    a_mode = 11;
  }

  // Set parameters for BSLMM and check for predict.
  if (a_mode == 11 || a_mode == 12 || a_mode == 13 || a_mode == 14) {
    if (a_mode == 11) {
      n_mh = 1;
    }
    if (logp_min == 0) {
      logp_min = -1.0 * mlog(to!double(ns_test));
    }

    if (h_scale == -1) {
      h_scale = min(1.0, 10.0 / sqrt(to!double(ns_test)));
    }
    if (rho_scale == -1) {
      rho_scale = min(1.0, 10.0 / sqrt(to!double(ns_test)));
    }
    if (logp_scale == -1) {
      logp_scale = min(1.0, 5.0 / sqrt(to!double(ns_test)));
    }

    if (h_min == -1) {
      h_min = 0.0;
    }
    if (h_max == -1) {
      h_max = 1.0;
    }

    if (s_max > ns_test) {
      s_max = ns_test;
      writeln("s_max is re-set to the number of analyzed SNPs.");
    }
    if (s_max < s_min) {
      writeln("error! maximum s value must be larger than the
               minimal value. current values = ", s_max, " and ", s_min);
      error = true;
    }
  } else if (a_mode == 41 || a_mode == 42) {
    if (indicator_bv.length != 0) {
      if (indicator_idv.length != indicator_bv.length) {
        writeln("error! number of rows in the
                 phenotype file does not match that in the
                 estimated breeding value file:", indicator_idv.length,
                 "\t", indicator_bv.length);
        error = true;
      } else {
        size_t flag_bv = 0;
        for (size_t i = 0; i < (indicator_bv).length; ++i) {
          if (indicator_idv[i] != indicator_bv[i]) {
            flag_bv++;
          }
        }
        if (flag_bv != 0) {
          writeln("error! individuals with missing value in the
                   phenotype file does not match that in the
                   estimated breeding value file: ", flag_bv);
          error = true;
        }
      }
    }
  }

  if (a_mode == 62 && file_beta != "" && mapRS2wcat.length == 0) {
    writeln("vc analysis with beta files requires -wcat file.");
    error = true;
  }
  if (a_mode == 67 && mapRS2wcat.length == 0) {
    writeln("ci analysis with beta files requires -wcat file.");
    error = true;
  }

  // File_mk needs to contain more than one line.
  if (n_vc == 1 && file_mk != "") {
    writeln("error! -mk file should contain more than one line.");
    error = true;
  }

  return;
}


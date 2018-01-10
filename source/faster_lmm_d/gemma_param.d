/*
   This code is part of faster_lmm_d and published under the GPLv3
   License (see LICENSE.txt)

   Copyright © 2017 - 2018 Prasun Anand & Pjotr Prins
*/

module faster_lmm_d.gemma_param;

import faster_lmm_d.dmatrix;

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

struct Kinship_param{
  DMatrix cvt;
  int[] indicator_idv;
  int[] indicator_snp;
  int[] indicator_weight;
}

struct mapRS{
  void clear(){}
}

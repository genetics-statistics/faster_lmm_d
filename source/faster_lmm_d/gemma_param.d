/*
   This code is part of faster_lmm_d and published under the GPLv3
   License (see LICENSE.txt)

   Copyright © 2017 Prasun Anand & Pjotr Prins
*/

module faster_lmm_d.gemma_param.d;

import faster_lmm_d.dmatrix;

struct SNPINFO {
  string chr;
  string rs_number;
  double cM;
  uint base_position;
  string a_minor;
  string a_major;
  size_t n_miss;
  double missingness;
  double maf;
  size_t n_idv;         // Number of non-missing individuals.
  size_t n_nb;          // Number of neighbours on the right hand side.
  size_t file_position; // SNP location in file.
};

// Results for LMM.
struct SUMSTAT {
  double beta;         // REML estimator for beta.
  double se;           // SE for beta.
  double lambda_remle; // REML estimator for lambda.
  double lambda_mle;   // MLE estimator for lambda.
  double p_wald;       // p value from a Wald test.
  double p_lrt;        // p value from a likelihood ratio test.
  double p_score;      // p value from a score test.
};

// Results for mvLMM.
struct MPHSUMSTAT {
  DMatrix v_beta;  // REML estimator for beta.
  double p_wald;          // p value from a Wald test.
  double p_lrt;           // p value from a likelihood ratio test.
  double p_score;         // p value from a score test.
  DMatrix v_Vg;    // Estimator for Vg, right half.
  DMatrix v_Ve;    // Estimator for Ve, right half.
  DMatrix v_Vbeta; // Estimator for Vbeta, right half.
};

// Hyper-parameters for BSLMM.
struct HYPBSLMM {
  double h;
  double pve;
  double rho;
  double pge;
  double logp;
  size_t n_gamma;
};

// Header struct.
struct HEADER {
  size_t rs_col;
  size_t chr_col;
  size_t pos_col;
  size_t cm_col;
  size_t a1_col;
  size_t a0_col;
  size_t z_col;
  size_t beta_col;
  size_t sebeta_col;
  size_t chisq_col;
  size_t p_col;
  size_t n_col;
  size_t nmis_col;
  size_t nobs_col;
  size_t ncase_col;
  size_t ncontrol_col;
  size_t af_col;
  size_t var_col;
  size_t ws_col;
  size_t cor_col;
  size_t coln; // Number of columns.
  //set<size_t> catc_col;
  //set<size_t> catd_col;
};

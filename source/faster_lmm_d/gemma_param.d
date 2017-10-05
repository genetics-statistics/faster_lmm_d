/*
   This code is part of faster_lmm_d and published under the GPLv3
   License (see LICENSE.txt)

   Copyright © 2017 Prasun Anand & Pjotr Prins
*/

module faster_lmm_d.gemma_param;

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
  string file_anno; // Optional.
  string file_gxe;  // Optional.
  string file_cvt;  // Optional.
  string file_cat, file_mcat;
  string file_catc, file_mcatc;
  string file_var;
  string file_beta;
  string file_cor;
  string file_kin, file_mk;
  string file_ku, file_kd;
  string file_study, file_mstudy;
  string file_ref, file_mref;
  string file_weight, file_wsnp, file_wcat;
  string file_out;
  string file_bf, file_hyp;
  string path_out;

  string file_epm;     // Estimated parameter file.
  string file_ebv;     // Estimated breeding value file.
  string file_log;     // Log file containing mean estimate.
  string file_read;    // File containing total number of reads.
  string file_gene;    // Gene expression file.
  string file_snps;    // File containing analyzed SNPs or genes.
  string file_ksnps;   // File SNPs for computing K
  string file_gwasnps; // File SNPs for computing GWAS

  // WJA added.
  string file_oxford;

  // QC-related parameters.
  double miss_level;
  double maf_level;
  double hwe_level;
  double r2_level;

  // LMM-related parameters.
  string loco;
  double l_min;
  double l_max;
  size_t n_region;
  double l_mle_null, l_remle_null;
  double logl_mle_H0, logl_remle_H0;
  double pve_null, pve_se_null, pve_total, se_pve_total;
  double vg_remle_null, ve_remle_null, vg_mle_null, ve_mle_null;
  //DMatrix Vg_remle_null, Ve_remle_null, Vg_mle_null, Ve_mle_null;
  //DMatrix VVg_remle_null, VVe_remle_null, VVg_mle_null;
  //DMatrix VVe_mle_null;
  //DMatrix beta_remle_null, se_beta_remle_null, beta_mle_null;
  //DMatrix se_beta_mle_null;
  double p_nr;
  double em_prec, nr_prec;
  size_t em_iter, nr_iter;
  size_t crt;
  double pheno_mean; // Phenotype mean from BSLMM fitting or prediction.

  // For fitting multiple variance components.
  // The first 3 are of size (n_vc), and the next 2 are of size n_vc+1.
  bool noconstrain;
  DMatrix v_traceG;
  DMatrix v_pve;
  DMatrix v_se_pve;

  DMatrix v_sigma2;
  DMatrix v_se_sigma2;
  DMatrix v_enrich;
  DMatrix v_se_enrich;
  DMatrix v_beta;
  DMatrix v_se_beta;

  // BSLMM/MCMC-related parameters.
  double h_min, h_max, h_scale;          // Priors for h.
  double rho_min, rho_max, rho_scale;    // Priors for rho.
  double logp_min, logp_max, logp_scale; // Priors for log(pi).
  size_t h_ngrid, rho_ngrid;
  size_t s_min, s_max; // Min & max. number of gammas.
  size_t w_step;       // # warm up/burn in iter.
  size_t s_step;       // # sampling iterations.
  size_t r_pace;       // Record pace.
  size_t w_pace;       // Write pace.
  size_t n_accept;     // Number of acceptance.
  size_t n_mh;         // # MH steps in each iter.
  double geo_mean;     // Mean of geometric dist.
  uint randseed;
  double trace_G;

  //HYPBSLMM cHyp_initial;  <= Param

  // VARCOV-related parameters.
  double window_cm;
  size_t window_bp;
  size_t window_ns;

  // vc-related parameters.
  size_t n_block;

  // Summary statistics.
  bool error;

  // Number of individuals.
  size_t ni_total, ni_test, ni_cvt, ni_study, ni_ref;
  size_t ni_max = 0; // -nind switch for testing purposes

  // Number of observed and missing phenotypes.
  size_t np_obs, np_miss;

  // Number of SNPs.
  size_t ns_total, ns_test, ns_study, ns_ref;

  size_t ng_total, ng_test;   // Number of genes.
  size_t ni_control, ni_case; // Number of controls and number of cases.
  size_t ni_subsample;        // Number of subsampled individuals.
  size_t n_cvt;               // Number of covariates.
  size_t n_cat;               // Number of continuous categories.
  size_t n_ph;                // Number of phenotypes.
  size_t n_vc;                // Number of variance components
                              // (including the diagonal matrix).
  double time_total;          // Record total time.
  double time_G;              // Time spent on reading files the
                              // second time and calculate K.
  double time_eigen;          // Time spent on eigen-decomposition.
  double time_UtX;            // Time spent on calculating UX and Uy.
  double time_UtZ;            // Time calculating UtZ for probit BSLMM.
  double time_opt;            // Time on optimization iterations/MCMC.
  double time_Omega;          // Time spent on calculating Omega.
  double time_hyp;            // Time sampling hyperparameters in PMM.
  double time_Proposal;       // Time spent on constructing the
                              // proposal distribution (i.e. the
                              // initial LMM or LM analysis).

  // Data.
  // Vector recording all phenotypes (NA replaced with -9).
  DMatrix pheno;

  // Vector recording all covariates (NA replaced with -9).
  DMatrix cvt;

  // Vector recording all covariates (NA replaced with -9).
  DMatrix gxe;

  // Vector recording weights for the individuals, which is
  // useful for animal breeding studies.
  DMatrix weight;

  // Matrix recording when a phenotype is missing for an
  // individual; 0 missing, 1 available.
  DMatrix indicator_pheno;

  // Indicator for individuals (phenotypes): 0 missing, 1
  // available for analysis
  int[] indicator_idv;

  // Sequence indicator for SNPs: 0 ignored because of (a) maf,
  // (b) miss, (c) non-poly; 1 available for analysis.
  DMatrix indicator_snp;

  // Sequence indicator for SNPs: 0 ignored because of (a) maf,
  // (b) miss, (c) non-poly; 1 available for analysis.
  DMatrix mindicator_snp;

  // Indicator for covariates: 0 missing, 1 available for
  // analysis.
  int[] indicator_cvt;

  // Indicator for gxe: 0 missing, 1 available for analysis.
  DMatrix indicator_gxe;

  // Indicator for weight: 0 missing, 1 available for analysis.
  DMatrix indicator_weight;

  // Indicator for estimated breeding value file: 0 missing, 1
  // available for analysis.
  DMatrix indicator_bv;

  // Indicator for read file: 0 missing, 1 available for analysis.
  DMatrix indicator_read;
  DMatrix vec_read; // Total number of reads.
  DMatrix vec_bv;   // Breeding values.
  DMatrix est_column;

  mapRS mapID2num;             // Map small ID to number, 0 to n-1.
  mapRS mapRS2chr;          // Map rs# to chromosome location.
  mapRS mapRS2bp;         // Map rs# to base position.
  mapRS mapRS2cM;           // Map rs# to cM.
  mapRS mapRS2est;          // Map rs# to parameters.
  mapRS mapRS2cat;          // Map rs# to category number.
  mapRS mapRS2catc; // Map rs# to cont. cat's.
  mapRS mapRS2wsnp;         // Map rs# to SNP weights.
  mapRS mapRS2wcat; // Map rs# to SNP cat weights.

  //vector<SNPINFO> snpInfo;          // Record SNP information.
  //vector<vector<SNPINFO>> msnpInfo; // Record SNP information.
  //set<string> setSnps;              // Set of snps for analysis (-snps).
  //set<string> setKSnps;             // Set of snps for K (-ksnps and LOCO)
  //set<string> setGWASnps;           // Set of snps for GWA (-gwasnps and LOCO)

  void ReadFiles(){}
  void CheckData(){}
  void WriteMatrix(DMatrix a, string b){}
  void WriteVar(string a){}
  void WriteVector(DMatrix a, string b){}
  void CopyCvtPhen(DMatrix a, DMatrix b, int c){}
  void ObtainWeight(string[] a, mapRS b){}
  void CalcS(mapRS a, mapRS b, DMatrix c, DMatrix d, DMatrix e, DMatrix f, DMatrix g, DMatrix h){}
  void CalcKin(DMatrix a){}
  void UpdateSNP(mapRS a){}
  void UpdateWeight(int, mapRS, ulong, DMatrix, mapRS){}

}

struct mapRS{
  void clear(){}
}

struct LM{
  void CopyFromParam(Param cpar){}
  void AnalyzeGene(DMatrix a, DMatrix b){}
  void AnalyzePlink(DMatrix a, DMatrix b){}
  void Analyzebgen(DMatrix a, DMatrix b){}
  void AnalyzeBimbam(DMatrix a, DMatrix b){}

  //void ReadFiles(){}
  //void CheckData(){}
  //void WriteMatrix(DMatrix a, string b){}
  //void WriteVar(string a){}
  //void WriteVector(DMatrix a, string b){}
  //void CopyCvtPhen(DMatrix a, DMatrix b, int c){}
  //void ObtainWeight(){}
  //void CalcS(){}

  void WriteFiles(){}
  void CopyToParam(Param cpar){}
}

void ReadFile_snps_header(string a, string[] b){}

void Calcq(mapRS a, mapRS b, DMatrix c, DMatrix d, DMatrix e, DMatrix f, DMatrix g, DMatrix h){}
void Calcq(ulong a, ulong[] b, ulong[] c, double[] d, double[] e, DMatrix f, DMatrix g, DMatrix h){}
void ReadFile_kin(string a, int[] b, mapRS c, int d, bool e, DMatrix f){}
void ReadFile_beta(string a, mapRS b, mapRS c, ulong[] d, ulong[] e, double[] f, double[] g, ulong h, ulong i, ulong j){}
void validate_K(DMatrix a, bool b, bool c){}
void setSnps_beta(){}

struct PRDT{
  size_t a_mode;
  size_t d_pace;

  string file_bfile;
  string file_geno;
  string file_out;
  string path_out;

  DMatrix indicator_pheno;
  int[] indicator_cvt;
  int[] indicator_idv;
  SNPINFO[] snpInfo;
  mapRS mapRS2est;

  size_t n_ph;
  size_t np_obs, np_miss;
  size_t ns_total;
  size_t ns_test;

  double time_eigen;

  // Main functions.
  void CopyFromParam(Param cPar);
  void CopyToParam(Param cPar);
  //void WriteFiles(DMatrix y_prdt);
  void WriteFiles(DMatrix Y_full);
  void AddBV(DMatrix G, const DMatrix u_hat, DMatrix y_prdt);
  void AnalyzeBimbam(DMatrix y_prdt);
  void AnalyzePlink(DMatrix y_prdt);
  void MvnormPrdt(const DMatrix Y_hat, const DMatrix H,
                  DMatrix Y_full);
}
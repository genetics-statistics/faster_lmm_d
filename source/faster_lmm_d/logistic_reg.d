/*
  This code is part of faster_lmm_d and published under the GPLv3
  License (see LICENSE.txt)

  Copyright © 2017 - 2018 Prasun Anand & Pjotr Prins
*/

/*
  Hail's  logistic regression algorithm  has been used initially to build
  the underlying code. We are extremely grateful to hail team for their work.

  The logistic regression algorithm in Faster-LMM-D modifies the algorithm from hail to take 
  advantage of GPUs. 

  URL : https://github.com/hail-is/hail
*/

module faster_lmm_d.logistic_reg;

import faster_lmm_d.dmatrix;
import faster_lmm_d.helpers;
import faster_lmm_d.optmatrix;
import faster_lmm_d.logistic_reg;
import faster_lmm_d.gemma_io;
import faster_lmm_d.gemma_kinship;
import faster_lmm_d.gemma_param;

import core.stdc.stdlib : exit;

import std.algorithm: min, max, reduce, maxElement;
import std.algorithm.iteration:map;
import std.conv;
import std.exception;
import std.math;
import std.process;
alias mlog = std.math.log;
import std.stdio;
import std.string;
import std.typecons;

import gsl.cdf;

void logistic_reg_run(const string option_snps, const string option_pheno, const string option_covar,
            const string option_geno, const string option_bfile){
  writeln("In logistic regression");
  run_test();

  auto pheno = ReadFile_pheno(option_pheno, [1]);

  size_t n_cvt;
  int[] indicator_cvt;

  writeln("reading covar " , option_covar);
  double[][] cvt = readfile_cvt(option_covar, indicator_cvt, n_cvt);

  auto indicators = process_cvt_phen(pheno.indicator_pheno, cvt, indicator_cvt, n_cvt);

  size_t ni_test = indicators.ni_test;
  writeln(ni_test);
  size_t ni_total = indicators.indicator_idv.length;

  string[] setSnps = ReadFile_snps(option_snps);
  //writeln(setSnps);
  string[string] mapRS2chr;
  size_t[string] mapRS2bp; 
  double[string] mapRS2cM;
  ReadFile_anno(option_snps, mapRS2chr, mapRS2bp, mapRS2cM);

  size_t n_ph = 1;
  string option_gene;

  SUMSTAT[] sumStat;

  // set covariates matrix W and phenotype matrix Y
  // an intercept should be included in W
  writeln(ni_test);
  DMatrix Y = zeros_dmatrix(ni_test, n_ph);
  DMatrix W = zeros_dmatrix(ni_test, n_cvt);
  CopyCvtPhen(W, Y, indicators.indicator_idv, indicators.indicator_cvt, cvt, pheno.pheno, n_ph, n_cvt, 0);

  auto geno_result = ReadFile_geno1(option_geno, ni_total, W, indicators.indicator_idv, setSnps, mapRS2chr, mapRS2bp, mapRS2cM);
  int[] indicator_snp = geno_result.indicator_snp;
  SNPINFO[] snpInfo = geno_result.snpInfo;
  writeln(snpInfo.length);

  //logistic_analyze_bimbam_batched(option_geno);
  writeln(Y);
  foreach(yi; Y.elements){
    if(yi != 1 && yi != 0){
      writeln(yi);
      writeln("For logistic regression, phenotype must be bool or numeric with all present values equal to 0 or 1");
      exit(0);
    }
  }

  LogisticRegressionFit dummy;  

  auto null_fit = model_fit(W, Y, dummy);

  logistic_analyze_bimbam(option_geno, null_fit, W, Y, indicators.indicator_idv, indicator_snp, ni_test, ni_total);
}

void logistic_analyze_bimbam(const string file_geno, const LogisticRegressionFit null_fit,
                             const DMatrix W, const DMatrix y,
                             const int[] indicator_idv, const int[] indicator_snp,
                             const size_t ni_test, const size_t ni_total) {

  writeln("entered lm_analyze_bimbam");
  
  SUMSTAT[] sumStat;
  SNPINFO[] snpInfo;

  writeln(file_geno);

  auto pipe = pipeShell("gunzip -c " ~ file_geno);
  File input = pipe.stdout;

  int lm_mode = 0;
  double beta = 0, se = 0, p_wald = 0, p_lrt = 0, p_score = 0;
  int n_miss, c_phen;
  double geno, x_mean;

  // Calculate some basic quantities.
  double df = to!double(W.shape[0]) - to!double(W.shape[1]) - 1.0;

  DMatrix x = zeros_dmatrix(ni_test, 1);
  DMatrix x_miss = zeros_dmatrix(1, ni_test);

  // Start reading genotypes and analyze.
  int t = 0;
  foreach (line; input.byLine) {
   
    if (indicator_snp[t] == 0) {
      t++;
      continue;
    }

    auto ch_ptr = to!string(line).split(",")[3..$];

    x_mean = 0.0;
    c_phen = 0;
    n_miss = 0;
    x_miss = zeros_dmatrix(1, W.shape[0]);
    for (size_t i = 0; i < ni_total; ++i) {
      if (indicator_idv[i] == 0) {
        continue;
      }

      if (ch_ptr[0] == "NA") {
        x_miss.elements[c_phen] = 0.0;
        n_miss++;
      } else {
        geno = to!double(ch_ptr[i].strip());

        x.elements[c_phen] = geno;
        x_miss.elements[c_phen] = 1.0;
        x_mean += geno;
      }
      c_phen++;
    }

    x_mean /= to!double(ni_test - n_miss);

    for (size_t i = 0; i < ni_test; ++i) {
      if (x_miss.elements[i] == 0) {
        x.elements[i] = x_mean;
      }
    }
    DMatrix X = horizontally_stack(W, x);
    WaldStats wald_stats = lm_wald_test(X, y, null_fit);
    // Calculate statistics.
    writeln(wald_stats);
  }
}

struct LogisticRegressionFit{

  int nIter;
  bool is_init = false;
  bool converged, exploded;
  double logLkhd;
  DMatrix b, score, fisher;

  this(DMatrix b, DMatrix score, DMatrix fisher, double logLkhd, int nIter, bool converged, bool exploded){
    this.is_init = true;
    this.b = b;
    this.score = score;
    this.fisher = fisher;
    this.logLkhd = logLkhd;
    this.nIter = nIter;
    this.converged = converged;
    this.exploded = exploded;
  }
}

alias Tuple!(DMatrix, "beta", DMatrix, "standard_error", DMatrix, "z_stat", DMatrix, "p_value") WaldStats;
alias Tuple!(DMatrix, "b", double, "chi2", double, "p") LikelihoodRatioStats;
alias Tuple!(double, "chi2", double, "p") ScoreStats;

  
WaldStats lm_wald_test(const DMatrix X, const DMatrix y, const LogisticRegressionFit null_fit){
  writeln("in WaldTest");

  auto fit = model_fit(X, y, null_fit);

  WaldStats waldStats;
  if(fit.converged) {
    DMatrix se = sqrt_dmatrix((inverse(fit.fisher)).get_diagonal);
    DMatrix z = fit.b / se.T;
    DMatrix p = zeros_dmatrix(z.shape[0], z.shape[1]);
    foreach(i; 0..p.size){
      p.elements[i] = 2 * gsl_cdf_ugaussian_P(-1*(abs(z.elements[i])));
    }
    waldStats = WaldStats(fit.b, se, z, p);
  } 

  return waldStats;
}

LikelihoodRatioStats likelihood_ratio_test(const DMatrix X, const DMatrix y, const LogisticRegressionFit null_fit){

  size_t m = X.cols;
  size_t m0 = null_fit.b.size;

  auto fit = model_fit(X, y, null_fit);

  LikelihoodRatioStats lrStats;
  if (fit.converged) {
    double chi2 = 2 * (fit.logLkhd - null_fit.logLkhd);
    double p = 1 - gsl_cdf_chisq_P(chi2, m - m0);

    lrStats = LikelihoodRatioStats(fit.b, chi2, p);
  }

  return lrStats;
}

ScoreStats score_test(const DMatrix X, const DMatrix y, const LogisticRegressionFit nullFit){ 
  
  size_t m = X.cols;
  DMatrix b = zeros_dmatrix(1, m);
  DMatrix score = zeros_dmatrix(1, m);
  DMatrix fisher = zeros_dmatrix(m, m);

  auto m0 = nullFit.b.size;

  DMatrix X0 = get_sub_dmatrix(X, 0, 0, X.rows, m0);
  DMatrix X1 = get_sub_dmatrix(X, 0, m0,  X.rows, X.cols - m0);

 foreach(i; 0..m0){ b.elements[i] = nullFit.b.elements[i]; }

  auto mu = sigmoid(matrix_mult(X, b.T));
  DMatrix score_r0 = dup_dmatrix(nullFit.score);
  DMatrix score_r1 = matrix_mult(X1.T, (y - mu));            

  foreach(i; 0..m0){ score.elements[i] = score_r0.elements[i]; }
  foreach(i; m0..score.size){ score.elements[i] = score_r1.elements[i-m0]; }

  DMatrix mu_prod = mu * subtract_num_dmatrix(1, mu);
  DMatrix mu_X1 = zeros_dmatrix(X1.rows, X1.cols);

  foreach(i; 0..X1.cols){ set_col2(mu_X1, i, X1.get_col(i) * mu_prod); }

  DMatrix fisher_r0_r0 = dup_dmatrix(nullFit.fisher);
  DMatrix fisher_r0_r1 = matrix_mult(X0.T, mu_X1);
  DMatrix fisher_r1_r0 = fisher_r0_r1.T;
  DMatrix fisher_r1_r1 = matrix_mult(X1.T, mu_X1);

  set_sub_dmatrix2(fisher, 0,  0,  m0,                    m0,               fisher_r0_r0);
  set_sub_dmatrix2(fisher, 0,  m0, m0,                    fisher.rows - m0, fisher_r0_r1);
  set_sub_dmatrix2(fisher, m0, 0,  fisher.cols - m0     , m0,               fisher_r1_r0);
  set_sub_dmatrix2(fisher, m0, m0, fisher.cols - m0,      fisher.rows - m0, fisher_r1_r1);

  auto chi2 = vector_ddot(score, fisher.solve(score.T));
  auto p = 1 - gsl_cdf_chisq_P(chi2, m - m0);
  return ScoreStats(chi2, p);
}

DMatrix bInterceptOnly(const DMatrix y, const size_t n, const size_t m){
  writeln("in bInterceptOnly");
  double sumY = sum(y.elements);
  DMatrix b = zeros_dmatrix(1, m);
  double avg = sum(y.elements) / n;
  b.elements[0] = mlog(avg / (1 - avg));
  return b;
}

LogisticRegressionFit model_fit(const DMatrix X, const DMatrix y, const LogisticRegressionFit optNullFit, 
                                const int maxIter = 25, const double tol = 1e-6){
  writeln("in model_fit");

  assert(y.size == X.rows);
  size_t n = X.rows;
  size_t m = X.cols;

  DMatrix b = zeros_dmatrix(1, m);
  DMatrix mu = zeros_dmatrix(1, n);
  DMatrix score = zeros_dmatrix(1, m);
  DMatrix fisher = zeros_dmatrix(m, m);

  if(!optNullFit.is_init){
    b = bInterceptOnly(y, X.rows, X.cols);
    mu = sigmoid(matrix_mult(X,b.T));
    score = matrix_mult(X.T, (y - mu));
    
    DMatrix temp = mu * subtract_num_dmatrix(1, mu);
    DMatrix muX = zeros_dmatrix(X.rows, X.cols);

    foreach(i; 0..X.cols){
      set_col2(muX, i, X.get_col(i) * temp);
    }

    fisher = matrix_mult(X.T, muX);
  }
  else{
    writeln("fitting non-null model");
    size_t m0 = optNullFit.b.size;

    DMatrix X0 = get_sub_dmatrix(X, 0, 0, X.rows, m0);
    DMatrix X1 = get_sub_dmatrix(X, 0, m0,  X.rows, X.cols - m0);

    foreach(i; 0..m0){ b.elements[i] = optNullFit.b.elements[i]; }

    mu = sigmoid(matrix_mult(X, b.T));
    
    DMatrix score_r0 = dup_dmatrix(optNullFit.score);
    DMatrix score_r1 = matrix_mult(X1.T, (y - mu));            

    foreach(i; 0..m0){ score.elements[i] = score_r0.elements[i]; }

    foreach(i; m0..score.size){ score.elements[i] = score_r1.elements[i-m0]; }

    DMatrix mu_prod = mu * subtract_num_dmatrix(1, mu);
    DMatrix mu_X1 = zeros_dmatrix(X1.rows, X1.cols);

    foreach(i; 0..X1.cols){ set_col2(mu_X1, i, X1.get_col(i) * mu_prod); }

    DMatrix fisher_r0_r0 = dup_dmatrix(optNullFit.fisher);
    DMatrix fisher_r0_r1 = matrix_mult(X0.T, mu_X1);
    DMatrix fisher_r1_r0 = fisher_r0_r1.T;
    DMatrix fisher_r1_r1 = matrix_mult(X1.T, mu_X1);

    set_sub_dmatrix2(fisher, 0,  0,  m0,                    m0,               fisher_r0_r0);
    set_sub_dmatrix2(fisher, 0,  m0, m0,                    fisher.rows - m0, fisher_r0_r1);
    set_sub_dmatrix2(fisher, m0, 0,  fisher.cols - m0     , m0,               fisher_r1_r0);
    set_sub_dmatrix2(fisher, m0, m0, fisher.cols - m0,      fisher.rows - m0, fisher_r1_r1);
  }

  int iter = 1;
  bool converged = false;
  bool exploded = false;
  
  DMatrix deltaB = zeros_dmatrix(3, 1);

  while (!converged && !exploded && iter <= maxIter) {
    deltaB = fisher.solve(score.T);

    if (maxElement(abs_dmatrix(deltaB).elements) < tol) {
      converged = true;
    } else {
      iter += 1;
      b = b + deltaB;
      mu = sigmoid(matrix_mult(X, b.T));
      score = matrix_mult(X.T, (y - mu)).T;
      
      DMatrix mu_prod = mu * subtract_num_dmatrix(1, mu);
      DMatrix zeros = zeros_dmatrix(X.rows, X.cols);
      foreach(i; 0..X.cols){ set_col2(zeros, i, X.get_col(i) * mu_prod); }
      
      fisher = matrix_mult(X.T, zeros);
    }
  }
  
  double logLkhd = sum((log_dmatrix((y * mu) + (subtract_num_dmatrix(1, y) * subtract_num_dmatrix(1, mu)))).elements);

  return LogisticRegressionFit(b, score, fisher, logLkhd, iter, converged, exploded);
}

void run_test(){
  DMatrix y = DMatrix([6,1] ,[ 0, 0, 1, 1, 1, 1]);

  DMatrix C = DMatrix([6,2],
    [0.0, -1.0,
     2.0,  3.0,
     1.0,  5.0,
    -2.0,  0.0,
    -2.0, -4.0,
     4.0,  3.0]);

  DMatrix ones = ones_dmatrix(6, 1);

  LogisticRegressionFit nullFit;  

  auto fit = model_fit(ones, y, nullFit);
  
  check_null_fit_results(fit);

  DMatrix X = horizontally_stack(ones, C);

  WaldStats wald_stats = lm_wald_test(X, y, fit);
  check_wald_test_results(wald_stats);
  LikelihoodRatioStats lrStats = likelihood_ratio_test(X, y, fit);
  check_likelihood_ratio_test_results(lrStats);

  ScoreStats score = score_test(X, y, fit);
  check_score_stats_test_results(score);
}


void check_null_fit_results(const LogisticRegressionFit fit){
  enforce(modDiff(fit.b.elements[0], 0.693147) < 0.001);
  enforce(modDiff(fit.logLkhd,          -3.81908501) < 0.001);
}

void check_wald_test_results(const WaldStats wald_stats){
  enforce(modDiff(wald_stats.beta.elements[0],           0.7245034) < 0.001);
  enforce(modDiff(wald_stats.standard_error.elements[0], 0.9396654) < 0.001);
  enforce(modDiff(wald_stats.z_stat.elements[0],         0.7710228) < 0.001);
  enforce(modDiff(wald_stats.p_value.elements[0],        0.4406934) < 0.001);
  
  enforce(modDiff(wald_stats.beta.elements[1],          -0.3585773) < 0.001);
  enforce(modDiff(wald_stats.standard_error.elements[1], 0.6246568) < 0.001);
  enforce(modDiff(wald_stats.z_stat.elements[1],        -0.5740389) < 0.001);
  enforce(modDiff(wald_stats.p_value.elements[1],        0.5659415) < 0.001);
  
  enforce(modDiff(wald_stats.beta.elements[2],           0.1922622) < 0.001);
  enforce(modDiff(wald_stats.standard_error.elements[2], 0.4559844) < 0.001);
  enforce(modDiff(wald_stats.z_stat.elements[2],         0.4216421) < 0.001);
  enforce(modDiff(wald_stats.p_value.elements[2],        0.6732863) < 0.001);

  writeln("LogisticRegression Wald tests pass!");
}

void check_likelihood_ratio_test_results(const LikelihoodRatioStats lr_stats){
  enforce(modDiff(lr_stats.chi2, 0.35153606)<0.001);
  enforce(modDiff(lr_stats.p,    0.8388125392)<0.001);
  writeln("LogisticRegression Likelihood Ratio tests pass!");
}

void check_score_stats_test_results(const ScoreStats score_stats){
  enforce(modDiff(score_stats.chi2, 0.346648)<0.001);
  enforce(modDiff(score_stats.p,    0.8408652791)<0.001);
  writeln("Score Stats tests pass!");
}

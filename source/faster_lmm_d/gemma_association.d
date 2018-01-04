/*
   This code is part of faster_lmm_d and published under the GPLv3
   License (see LICENSE.txt)

   Copyright © 2017 - 2018 Prasun Anand & Pjotr Prins
*/

module faster_lmm_d.gemma_association;

import core.stdc.stdlib : exit;

import std.conv;
import std.exception;
import std.file;
import std.math;
import std.parallelism;
import std.algorithm: min, max, reduce;
alias mlog = std.math.log;
import std.process;
import std.range;
import std.stdio;
import std.typecons;
import std.experimental.logger;
import std.string;

import faster_lmm_d.dmatrix;
import faster_lmm_d.gemma_param;
import faster_lmm_d.gemma_lmm;
import faster_lmm_d.helpers;
import faster_lmm_d.optmatrix;

import gsl.permutation;
import gsl.cdf;
import gsl.errno;
import gsl.math;
import gsl.min;
import gsl.roots;

void analyze_bimbam(Param cPar, const DMatrix U, const DMatrix eval, const DMatrix UtW, const DMatrix Uty,
                    const DMatrix W, const DMatrix y, const size_t n_cvt, const size_t ni_total, const size_t LMM_BATCH_SIZE = 1000) {

  DMatrix UT = U.T;
  DMatrix indicator_idv = read_matrix_from_file2(cPar.indicator_idv_file);
  DMatrix indicator_snp = read_matrix_from_file2(cPar.indicator_snp_file);


  string filename = cPar.file_geno;
  auto pipe = pipeShell("gunzip -c " ~ filename);
  File input = pipe.stdout;

  SUMSTAT[] sumStat;

  double lambda_mle=0, lambda_remle=0, beta=0, se=0, p_wald=0;
  double p_lrt=0, p_score=0;
  int n_miss, c_phen;
  double geno, x_mean;

  double logl_H1=0.0;
  const size_t ni_test = UtW.shape[0];
  const size_t n_region = cPar.n_region;
  const int a_mode = 1;
  const double l_min = cPar.l_min;
  const double l_mle_null = cPar.l_mle_null;
  const double l_max = cPar.l_max;
  const double logl_mle_H0 = cPar.logl_mle_H0;

  size_t n_index=(n_cvt+2+1)*(n_cvt+2)/2;

  writeln("ni_test =======> ", ni_test);
  writeln("ni_total =======> ", ni_total);
  writeln("n_region =======> ", n_region);
  writeln("l_mle_null =======> ", l_mle_null);
  writeln("l_remle_null =======> ", cPar.l_remle_null);
  writeln("l_max =======> ", l_max);
  writeln("l_min =======> ", l_min);
  writeln("logl_mle_H0 =======> ", logl_mle_H0);
  writeln("n_index =======> ", n_index);

  DMatrix x = zeros_dmatrix(U.shape[0],1);

  const DMatrix ab = zeros_dmatrix(1, n_index);

  size_t msize=10000;
  DMatrix Xlarge = zeros_dmatrix(U.shape[0], msize);
  DMatrix UtXlarge = zeros_dmatrix(U.shape[0], msize);


  //start reading genotypes and analyze
  size_t c=0, t_last=0;
  foreach (element; indicator_snp.elements) {
    if (element ==0) {continue;}
    t_last++;
  }

  int t = 0;


  DMatrix Uab = calc_Uab(UtW, Uty, U.shape[1], n_index);

  foreach (line ; input.byLine) {
    if (indicator_snp.elements[t]==0) {
      t++;
      continue;
    }

    auto chr = to!string(line).split(",")[3..$];

    x_mean=0.0; c_phen=0; n_miss=0;
    DMatrix x_miss = zeros_dmatrix(1, U.shape[0]);
    foreach ( i; 0..ni_total) {
      auto ch_ptr = to!string(chr[i].strip());
      if (indicator_idv.elements[i]==0) {continue;}

      if (ch_ptr == "NA") {
        x_miss.elements[c_phen] = 0.0;
        n_miss++;
      }
      else {
        geno=to!double(ch_ptr);
        x.elements[c_phen] = geno;
        x_miss.elements[c_phen] = 1.0;
        x_mean += geno;
      }
      c_phen++;
    }

    x_mean/= to!double(ni_test-n_miss);

    foreach (i; 0..ni_test) {
      if ( x_miss.elements[i] == 0) {
        x.elements[i] = x_mean;
      }
    }

    set_col2(Xlarge, c%msize, x);

    c++;

    size_t index_snp = 0;

    if (c % msize==0 || c==t_last) {
      size_t l=0;
      if (c%msize==0) {l=msize;} else {l=c%msize;}

      DMatrix Xlarge_sub = get_sub_dmatrix(Xlarge, 0, 0, Xlarge.shape[0], l);
      DMatrix UtXlarge_sub = matrix_mult(UT, Xlarge_sub);
      UtXlarge = set_sub_dmatrix(UtXlarge, 0, 0, UtXlarge.shape[0], l, UtXlarge_sub);
      const DMatrix UtXlargeT = UtXlarge.T;

      version(PARALLEL) {
        auto tsps = new SUMSTAT[l];
        auto items = iota(0,l).array;

        foreach (ref snp; taskPool.parallel(items,100)) {
          const DMatrix Utx = get_row(UtXlargeT, snp);
          const Uab_new = calc_Uab(UtW, Uty, Utx, Uab);
          loglikeparam param1 = loglikeparam(false, ni_test, n_cvt, eval, Uab_new, ab, 0);

          tsps[snp].lambda_remle = calc_lambda ('R', cast(void *)&param1, l_min, l_max, n_region).lambda;
          auto score = calc_RL_Wald(ni_test, tsps[snp].lambda_remle, param1);
          tsps[snp].beta = score.beta;
          tsps[snp].se = score.se;
          tsps[snp].p_wald = score.p_wald;
        }
        sumStat ~= tsps;
      }
      else{
        foreach (snp; 0..l) {
          const DMatrix Utx = get_row(UtXlargeT, snp);
          const Uab_new = calc_Uab(UtW, Uty, Utx, Uab);

          loglikeparam param1 = loglikeparam(false, ni_test, n_cvt, eval, Uab_new, ab, 0);

          lambda_remle = calc_lambda ('R', cast(void *)&param1, l_min, l_max, n_region).lambda;
          auto score = calc_RL_Wald(ni_test, lambda_remle, param1);
          sumStat ~= SUMSTAT(score.beta, score.se, lambda_remle, score.p_wald);
        }
      }
      Xlarge = zeros_dmatrix(U.shape[0], msize);
    }
    t++;
  }

  writeln(sumStat);
  check_assoc_result(sumStat);
  return;
}

void analyze_bimbam_batched(Param cPar, const DMatrix U, const DMatrix eval, const DMatrix UtW, const DMatrix Uty,
                    const DMatrix W, const DMatrix y, const size_t n_cvt, const size_t ni_total, const size_t LMM_BATCH_SIZE = 1000) {

  DMatrix UT = U.T;
  DMatrix indicator_idv = read_matrix_from_file2(cPar.indicator_idv_file);
  DMatrix indicator_snp = read_matrix_from_file2(cPar.indicator_snp_file);

  string filename = cPar.file_geno;
  auto pipe = pipeShell("gunzip -c " ~ filename);
  File input = pipe.stdout;

  SUMSTAT[] sumStat;

  double lambda_mle=0, lambda_remle=0, beta=0, se=0, p_wald=0;
  double p_lrt=0, p_score=0;
  int n_miss, c_phen;
  double geno, x_mean;

  double logl_H1=0.0;
  const size_t ni_test = UtW.shape[0];
  const size_t n_region = cPar.n_region;
  const int a_mode = 1;
  const double l_min = cPar.l_min;
  const double l_mle_null = cPar.l_mle_null;
  const double l_max = cPar.l_max;
  const double logl_mle_H0 = cPar.logl_mle_H0;

  size_t n_index=(n_cvt+2+1)*(n_cvt+2)/2;

  writeln("ni_test =======> ", ni_test);
  writeln("ni_total =======> ", ni_total);
  writeln("n_region =======> ", n_region);
  writeln("l_mle_null =======> ", l_mle_null);
  writeln("l_remle_null =======> ", cPar.l_remle_null);
  writeln("l_max =======> ", l_max);
  writeln("l_min =======> ", l_min);
  writeln("logl_mle_H0 =======> ", logl_mle_H0);
  writeln("n_index =======> ", n_index);

  DMatrix x = zeros_dmatrix(U.shape[0],1);

  const DMatrix ab = zeros_dmatrix(1, n_index);

  // Create a large matrix.
  size_t msize=1000;
  DMatrix Xlarge = zeros_dmatrix(U.shape[0], msize);
  DMatrix UtXlarge = zeros_dmatrix(U.shape[0], msize);


  //start reading genotypes and analyze
  size_t c=0, t_last=0;
  foreach (element; indicator_snp.elements) {
    if (element ==0) {continue;}
    t_last++;
  }

  int t = 0;


  DMatrix Uab = calc_Uab(UtW, Uty, U.shape[1], n_index);

  auto task_pool = new TaskPool(totalCPUs);

  foreach (line ; input.byLine) {
    if (indicator_snp.elements[t]==0) {
      t++;
      continue;
    }

    auto chr = to!string(line).split(",")[3..$];

    x_mean=0.0; c_phen=0; n_miss=0;
    DMatrix x_miss = zeros_dmatrix(1, U.shape[0]);
    foreach ( i; 0..ni_total) {
      auto ch_ptr = to!string(chr[i].strip());
      if (indicator_idv.elements[i]==0) {continue;}

      if (ch_ptr == "NA") {
        x_miss.elements[c_phen] = 0.0;
        n_miss++;
      }
      else {
        geno=to!double(ch_ptr);
        x.elements[c_phen] = geno;
        x_miss.elements[c_phen] = 1.0;
        x_mean += geno;
      }
      c_phen++;
    }

    x_mean/= to!double(ni_test-n_miss);

    foreach (i; 0..ni_test) {
      if ( x_miss.elements[i] == 0) {
        x.elements[i] = x_mean;
      }
    }

    set_col2(Xlarge, c%msize, x);

    c++;

    if (c % msize==0 || c==t_last) {
      size_t l=0;
      if (c%msize==0) {l=msize;} else {l=c%msize;}
      version(PARALLEL){
        auto taskk = task(&compute_assoc, Xlarge, UtXlarge, l, UtW, Uty, Uab, ab, eval, UT, ni_test, n_cvt, n_region, l_max, l_min);
        task_pool.put(taskk);
      }
      else{
        compute_assoc(Xlarge, UtXlarge, l, UtW, Uty, Uab, ab, eval, UT, ni_test, n_cvt, n_region, l_max, l_min);
      }
      Xlarge = zeros_dmatrix(U.shape[0], msize);
    }
    t++;
  }
  task_pool.finish(true);
  return;
}

void compute_assoc( const DMatrix Xlarge, const DMatrix UtXlarge, const size_t l, const DMatrix UtW,
                      const DMatrix Uty, const DMatrix Uab, const DMatrix ab, const DMatrix eval,
                      const DMatrix UT, const size_t ni_test, const size_t n_cvt, const size_t n_region,
                      const double l_max, const double l_min){

  double[] elements = new double[l];
  double[] Uab_large_elements;

  DMatrix Xlarge_sub = get_sub_dmatrix(Xlarge, 0, 0, Xlarge.shape[0], l);
  DMatrix UtXlarge_sub = matrix_mult(UT, Xlarge_sub);
  DMatrix UtXlarge_new = set_sub_dmatrix(UtXlarge, 0, 0, UtXlarge.shape[0], l, UtXlarge_sub);

  foreach (snp; 0..l) {
    const DMatrix Utx = get_col(UtXlarge_new, snp);
    const Uab_new = calc_Uab(UtW, Uty, Utx, Uab);
    Uab_large_elements ~= Uab_new.elements;

    loglikeparam param1 = loglikeparam(false, ni_test, n_cvt, eval, Uab_new, ab, 0);
    elements[snp] = calc_lambda ('R', cast(void *)&param1, l_min, l_max, n_region).lambda;
  }
  DMatrix Uab_large = DMatrix([l*Uab.shape[1] , Uab.shape[0]], Uab_large_elements);
  loglikeparam param2 = loglikeparam(false, ni_test, n_cvt, eval, Uab_large, ab, 0);
  writeln(calc_RL_Wald_batched(ni_test, elements, param2)[0]);
}

void check_assoc_result(SUMSTAT[] snps){

  enforce(modDiff(snps[0].beta, -0.0778866 ) < 0.001);
  enforce(modDiff(snps[0].se, 0.061935) < 0.001);
  enforce(modDiff(snps[0].lambda_remle, 4.31799) < 0.001);
  enforce(modDiff(snps[0].p_wald, 0.208762) < 0.001);

  enforce(modDiff(snps[$-1].beta, 0.0684089 ) < 0.001);
  enforce(modDiff(snps[$-1].se, 0.0462648 ) < 0.001);
  enforce(modDiff(snps[$-1].lambda_remle, 4.31939 ) < 0.001);
  enforce(modDiff(snps[$-1].p_wald, 0.139461) < 0.001);

  writeln("LMM Association tests pass.");
}

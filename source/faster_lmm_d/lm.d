/*
   This code is part of faster_lmm_d and published under the GPLv3
   License (see LICENSE.txt)

   Copyright © 2017 - 2018 Prasun Anand & Pjotr Prins
*/

module faster_lmm_d.lm;

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
import faster_lmm_d.gemma;
import faster_lmm_d.gemma_io;
import faster_lmm_d.gemma_kinship;
import faster_lmm_d.gemma_lmm;
import faster_lmm_d.gemma_param;
import faster_lmm_d.helpers;
import faster_lmm_d.optmatrix;

import gsl.permutation;
import gsl.rng;
import gsl.randist;
import gsl.cdf;

void lm_run(string option_snps, string option_kinship, string option_pheno, string option_covar, string option_geno, string option_bfile, size_t lm_mode){
  writeln("entered lm_run");

  writeln("reading pheno " , option_pheno);
  auto pheno = ReadFile_pheno(option_pheno, [1]);

  size_t n_cvt;
  int[] indicator_cvt;

  writeln("reading covar " , option_covar);
  double[][] cvt = readfile_cvt(option_covar, indicator_cvt, n_cvt);
  writeln(cvt);

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
  DMatrix Y = zeros_dmatrix(ni_test, n_ph);
  DMatrix W = zeros_dmatrix(ni_test, n_cvt);
  CopyCvtPhen(W, Y, indicators.indicator_idv, indicators.indicator_cvt, cvt, pheno.pheno, n_ph, n_cvt, 0);

  auto geno_result = ReadFile_geno1(option_geno, ni_total, W, indicators.indicator_idv, setSnps, mapRS2chr, mapRS2bp, mapRS2cM);
  int[] indicator_snp = geno_result.indicator_snp;
  SNPINFO[] snpInfo = geno_result.snpInfo;

  // Fit LM or mvLM
  if (n_ph == 1) {
    //Lm.CopyFromParam(cPar);

    DMatrix Y_col = get_col(Y, 0);

    if (option_gene != "") {
      lm_analyze_gene(option_geno, W, Y_col); // y is the predictor, not the phenotype
    } else if (option_bfile != "") {
      lm_analyze_plink(option_geno, W, Y_col);
    } else {
      sumStat = lm_analyze_bimbam(option_geno, W, Y_col, indicators.indicator_idv, indicator_snp, ni_test, ni_total);
    }

    lm_write_files(sumStat, option_geno, snpInfo, indicator_snp, ni_test, lm_mode);
  }
}

void lm_write_files(SUMSTAT[] sumStat, const string file_gene, SNPINFO[] snpInfo, int[] indicator_snp, size_t ni_test, size_t lm_mode) {
  // define later
  string path_out = "";
  string file_out = "";
  string file_str = path_out ~ "/" ~ file_out ~ ".assoc.txt";

  if (file_gene == "") {
    fwrite("geneID\t");

    if (lm_mode == 1) {
      fwrite("beta\tse\tp_wald\n");
    } else if (lm_mode == 2) {
      fwrite("p_lrt\n");
    } else if (lm_mode == 3) {
      fwrite("beta\tse\tp_score\n");
    } else if (lm_mode == 4) {
      fwrite("beta\tse\tp_wald\tp_lrt\tp_score\n");
    } else {
    }

    for (size_t t = 0; t < sumStat.length; ++t) {
      fwrite(snpInfo[t].rs_number, "\t");

      if (lm_mode == 1) {
        fwrite(sumStat[t].beta, "\t", sumStat[t].se, "\t", sumStat[t].p_wald, "\n");
      } else if (lm_mode == 2) {
        fwrite(sumStat[t].p_lrt, "\n");
      } else if (lm_mode == 3) {
        fwrite(sumStat[t].beta, "\t", sumStat[t].se, "\t", sumStat[t].p_score, "\n");
      } else if (lm_mode == 4) {
        fwrite(sumStat[t].beta, "\t", sumStat[t].se, "\t", sumStat[t].p_wald, "\t",
              sumStat[t].p_lrt, "\t", sumStat[t].p_score);
      } else {
      }
    }
  } else {
    fwrite("chr",
          "\t",
          "rs",
          "\t",
          "ps",
          "\t",
          "n_mis",
          "\t",
          "n_obs",
          "\t",
          "allele1",
          "\t",
          "allele0",
          "\t",
          "af",
          "\t");

    if (lm_mode == 1) {
      fwrite("beta",
            "\t",
            "se",
            "\t",
            "p_wald", 
            "\n");
    } else if (lm_mode == 2) {
      fwrite("p_lrt", "\n");
    } else if (lm_mode == 3) {
      fwrite("beta",
            "\t",
            "se",
            "\t",
            "p_score",
            "\n");
    } else if (lm_mode == 4) {
      fwrite("beta",
            "\t",
            "se",
            "\t",
            "p_wald",
            "\t",
            "p_lrt",
            "\t",
            "p_score",
            "\n");
    } else {
    }

    size_t t = 0;
    for (size_t i = 0; i < snpInfo.length; ++i) {
      if (indicator_snp[i] == 0) {
        continue;
      }

      fwrite(snpInfo[i].chr, "\t", snpInfo[i].rs_number, "\t", 
            snpInfo[i].base_position, "\t", snpInfo[i].n_miss, "\t",
            ni_test - snpInfo[i].n_miss, "\t", snpInfo[i].a_minor, "\t",
            snpInfo[i].a_major, "\t", snpInfo[i].maf,"\t");

      if (lm_mode == 1) {
        fwrite(sumStat[t].beta, "\t", sumStat[t].se, "\t", sumStat[t].p_wald, "\n");
      } else if (lm_mode == 2) {
        fwrite(sumStat[t].p_lrt, "\n");
      } else if (lm_mode == 3) {
        fwrite(sumStat[t].beta, "\t", sumStat[t].se, "\t", sumStat[t].p_score, "\n");
      } else if (lm_mode == 4) {
        fwrite(sumStat[t].beta, "\t", sumStat[t].se, "\t", sumStat[t].p_wald, "\t",
              sumStat[t].p_lrt, "\t", sumStat[t].p_score, "\n");
      } else {
      }
      t++;
    }
  }

  return;
}

void CalcvPv(const DMatrix WtWi, const DMatrix Wty,
             const DMatrix Wtx, const DMatrix y, const DMatrix x,
             ref double xPwy, ref double xPwx) {
  size_t c_size = Wty.size;
  double d;

  xPwx = vector_ddot(x, x);
  xPwy = vector_ddot(x, y);
  DMatrix WtWiWtx = matrix_mult(WtWi, Wtx);

  d = vector_ddot(WtWiWtx, Wtx);
  xPwx -= d;

  d = vector_ddot(WtWiWtx, Wty);
  xPwy -= d;

  return;
}

void CalcvPv(const DMatrix WtWi, const DMatrix Wty, const DMatrix y, ref double yPwy) {
  yPwy = vector_ddot(y, y);
  DMatrix WtWiWty = matrix_mult(WtWi, Wty);

  double d = vector_ddot(WtWiWty, Wty);
  yPwy -= d;

  return;
}

// Calculate p-values and beta/se in a linear model.
void LmCalcP(const size_t lm_mode, const double yPwy, const double xPwy,
             const double xPwx, const double df, const size_t n_size,
             ref double beta, ref double se, ref double p_wald, ref double p_lrt,
             ref double p_score) {
  double yPxy = yPwy - xPwy * xPwy / xPwx;
  double se_wald, se_score;

  beta = xPwy / xPwx;

  se_wald = sqrt(yPxy / (df * xPwx));
  se_score = sqrt(yPwy / (to!double(n_size) * xPwx));
  p_wald = gsl_cdf_fdist_Q(beta * beta / (se_wald * se_wald), 1.0, df);
  p_score = gsl_cdf_fdist_Q(beta * beta / (se_score * se_score), 1.0, df);
  p_lrt = gsl_cdf_chisq_Q(to!double(n_size) * (mlog(yPwy) - mlog(yPxy)), 1);

  if (lm_mode == 3) {
    se = se_score;
  } else {
    se = se_wald;
  }

  return;
}

void lm_analyze_gene(const string file_gene, const DMatrix W, const DMatrix x) {
  writeln("entered lm_analyze_gene");
  size_t lm_mode;
  size_t ni_test, ng_total;
  int[] indicator_idv, indicator_snp;
  SUMSTAT[] sumStat;
  SNPINFO[] snpInfo;

  writeln("entering lm_analyze_gene");
  File infile = File(file_gene);

  string line;

  double beta = 0, se = 0, p_wald = 0, p_lrt = 0, p_score = 0;
  int c_phen;
  string rs; // Gene id.
  double d;

  // Calculate some basic quantities.
  double yPwy, xPwy, xPwx;
  double df = to!double(W.shape[0]) - to!double(W.shape[1]) - 1.0;

  DMatrix y; // = gsl_vector_alloc(W,shape[0]);

  DMatrix WtW = matrix_mult(W.T, W);
  DMatrix WtWi = WtW.inverse();

  DMatrix Wtx =  matrix_mult(W.T, x);
  CalcvPv(WtWi, Wtx, x, xPwx);

  // Header.
  infile.readln();

  for (size_t t = 0; t < ng_total; t++) {
    line = infile.readln();
   
    auto ch_ptr = line.split("\t");
    rs = ch_ptr[0];

    c_phen = 0;
    for (size_t i = 0; i < indicator_idv.length; ++i) {
      if (indicator_idv[i] == 0) {
        continue;
      }

      d = to!double(ch_ptr[1]);
      y.elements[c_phen] = d;

      c_phen++;
    }

    // Calculate statistics.
    DMatrix Wty = matrix_mult(W.T, y);
    CalcvPv(WtWi, Wtx, Wty, x, y, xPwy, yPwy);
    LmCalcP(lm_mode, yPwy, xPwy, xPwx, df, W.shape[0], beta, se, p_wald,
            p_lrt, p_score);

    // Store summary data.
    SUMSTAT SNPs = SUMSTAT(beta, se, 0.0, 0.0, p_wald, p_lrt, p_score, -0.0);
    sumStat ~= SNPs;
  }

  return;
}

SUMSTAT[] lm_analyze_bimbam(const string file_geno, const DMatrix W, const DMatrix y, int[] indicator_idv, int[] indicator_snp, size_t ni_test, size_t ni_total) {

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
  double yPwy, xPwy, xPwx;
  double df = to!double(W.shape[0]) - to!double(W.shape[1]) - 1.0;

  DMatrix x = zeros_dmatrix(ni_total, 1);
  DMatrix x_miss = zeros_dmatrix(1, ni_total);

  DMatrix WtW = matrix_mult(W.T, W);
  DMatrix WtWi = WtW.inverse();

  writeln(WtWi);

  DMatrix Wty = matrix_mult(W.T, y);
  CalcvPv(WtWi, Wty, y, yPwy);

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
      geno = x.elements[i];
    }

    // Calculate statistics.
    DMatrix Wtx = matrix_mult(W.T, x);

    CalcvPv(WtWi, Wty, Wtx, y, x, xPwy, xPwx);
    LmCalcP(lm_mode - 50, yPwy, xPwy, xPwx, df, W.shape[0], beta, se, p_wald, p_lrt, p_score);

    // Store summary data.
    SUMSTAT SNPs = SUMSTAT(beta, se, 0.0, 0.0, p_wald, p_lrt, p_score, -0.0);
    sumStat ~= SNPs;
    t++;
  }

  return sumStat;
}

void lm_analyze_plink(const string file_bfile, const DMatrix W, const DMatrix y) {
  
  writeln("entered lm_analyze_plink");

  size_t ni_total, ni_test;
  int lm_mode;
  int[] indicator_snp, indicator_idv;
  SUMSTAT[] sumStat;
  SNPINFO[] snpInfo;

  string file_bed = file_bfile ~ ".bed";
  File infile = File(file_bed);

  char ch;
  int[] b;

  double beta = 0, se = 0, p_wald = 0, p_lrt = 0, p_score = 0;
  int n_bit, n_miss, ci_total, ci_test;
  double geno, x_mean;

  // Calculate some basic quantities.
  double yPwy, xPwy, xPwx;
  double df = to!double(W.shape[0]) - to!double(W.shape[1]) - 1.0;

  DMatrix x; // = gsl_vector_alloc(W->size1);

  gsl_permutation *pmt = gsl_permutation_alloc(W.shape[1]);

  DMatrix WtW = matrix_mult(W.T, W);
  DMatrix WtWi = WtW.inverse;

  DMatrix Wty = matrix_mult(W.T, y);
  CalcvPv(WtWi, Wty, y, yPwy);

  // Calculate n_bit and c, the number of bit for each SNP.
  if (ni_total % 4 == 0) {
    n_bit = to!int(ni_total) / 4;
  } else {
    n_bit = to!int(ni_total) / 4 + 1;
  }

  // Print the first three magic numbers.
  for (int i = 0; i < 3; ++i) {
    //infile.read(ch, 1);  // TODO: FIXME
    //b = ch;
  }

  for(size_t t = 0; t < snpInfo.length; ++t) {
    if (indicator_snp[t] == 0) {
      continue;
    }

    // n_bit, and 3 is the number of magic numbers.
    infile.seek(t * n_bit + 3);

    // Read genotypes.
    x_mean = 0.0;
    n_miss = 0;
    ci_total = 0;
    ci_test = 0;
    for (int i = 0; i < n_bit; ++i) {
      //infile.read(ch, 1);
      //b = ch;  // TODO: FIXME

      // Minor allele homozygous: 2.0; major: 0.0;
      for (size_t j = 0; j < 4; ++j) {
        if ((i == (n_bit - 1)) && ci_total == ni_total) {
          break;
        }
        if (indicator_idv[ci_total] == 0) {
          ci_total++;
          continue;
        }

        if (b[2 * j] == 0) {
          if (b[2 * j + 1] == 0) {
            x.elements[ci_test] = 2;
            x_mean += 2.0;
          } else {
            x.elements[ci_test] = 1;
            x_mean += 1.0;
          }
        } else {
          if (b[2 * j + 1] == 1) {
            x.elements[ci_test] = 0;
          } else {
            x.elements[ci_test] = -9;
            n_miss++;
          }
        }

        ci_total++;
        ci_test++;
      }
    }

    x_mean /= to!double(ni_test - n_miss);

    for (size_t i = 0; i < ni_test; ++i) {
      geno = x.elements[i];
      if (geno == -9) {
        x.elements[i] = x_mean;
        geno = x_mean;
      }
    }

    // Calculate statistics.

    DMatrix Wtx = matrix_mult(W.T, x);
    CalcvPv(WtWi, Wty, Wtx, y, x, xPwy, xPwx);
    LmCalcP(lm_mode, yPwy, xPwy, xPwx, df, W.shape[0], beta, se, p_wald, p_lrt, p_score);

    // store summary data
    SUMSTAT SNPs = SUMSTAT(beta, se, 0.0, 0.0, p_wald, p_lrt, p_score, -0.0);
    sumStat ~= SNPs;
  }

  gsl_permutation_free(pmt);
  return;
}

// Make sure that both y and X are centered already.
void MatrixCalcLmLR(const DMatrix X, const DMatrix y,
                    double[size_t] pos_loglr) {
  double yty, xty, xtx, log_lr;
  yty = vector_ddot(y, y);

  for (size_t i = 0; i < X.shape[1]; ++i) {
    //gsl_vector_const_view
    DMatrix X_col = get_col(X, i);
    xtx = vector_ddot(X_col, X_col);
    xty = vector_ddot(X_col, y);

    log_lr = 0.5 * to!double(y.size) * (mlog(yty) - mlog(yty - xty * xty / xtx));
    // TODO
    //pos_loglr ~= (make_pair(i, log_lr));
  }

  return;
}

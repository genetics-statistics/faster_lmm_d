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
import std.zlib;

import faster_lmm_d.dmatrix;
import faster_lmm_d.gemma;
import faster_lmm_d.gemma_io;
import faster_lmm_d.gemma_kinship;
import faster_lmm_d.gemma_lmm;
import faster_lmm_d.gemma_param;
import faster_lmm_d.helpers;
import faster_lmm_d.optmatrix;

import gsl.cdf;

void lm_run(const string option_snps, const string option_pheno, const string option_covar,
            const string option_geno, const string option_bfile, const size_t lm_mode){
  writeln("entered lm_run");

  writeln("reading pheno " , option_pheno);
  auto pheno = ReadFile_pheno(option_pheno, [1]);

  size_t n_cvt;
  int[] indicator_cvt;

  writeln("reading covar " , option_covar);
  double[][] cvt = readfile_cvt(option_covar, indicator_cvt, n_cvt);
  //writeln(cvt);

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

  //auto geno_result = ReadFile_geno1(option_geno, ni_total, W, indicators.indicator_idv, setSnps, mapRS2chr, mapRS2bp, mapRS2cM);
  auto geno_result = ReadFile_bgen(option_geno, ni_total, W, indicators.indicator_idv, setSnps, mapRS2chr, mapRS2bp, mapRS2cM);
  int[] indicator_snp = geno_result.indicator_snp;
  SNPINFO[] snpInfo = geno_result.snpInfo;
  writeln(snpInfo.length);

  DMatrix WtW = matrix_mult(W.T, W);
  DMatrix WtWi = WtW.inverse();

  // Fit LM or mvLM
  if (n_ph == 1) {
    //Lm.CopyFromParam(cPar);

    DMatrix Y_col = get_col(Y, 0);

    if (option_gene != "") {
      lm_analyze_gene(option_geno, W, Y_col); // y is the predictor, not the phenotype
    } else if (option_bfile != "") {
      lm_analyze_plink(option_geno, W, Y_col);
    } else {
      DMatrix Wty = matrix_mult(W.T, Y_col);
      sumStat = lm_analyze_bimbam(option_geno, W, Y_col, WtWi, Wty, indicators.indicator_idv, indicator_snp, ni_test, ni_total);
      lm_analyze_bgen(option_geno, W, Y_col, WtWi, Wty, indicators.indicator_idv, indicator_snp, ni_test, ni_total);
    }

    //lm_write_files(sumStat, option_geno, snpInfo, indicator_snp, ni_test, lm_mode);
  }
}

void lm_write_files(SUMSTAT[] sumStat, const string file_gene, const SNPINFO[] snpInfo,
                    const int[] indicator_snp, const size_t ni_test, const size_t lm_mode) {
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

alias Tuple!(const double, "x", const double, "y")vPvResult;

vPvResult CalcvPv(const DMatrix WtWi, const DMatrix Wty,
             const DMatrix Wtx, const DMatrix y, const DMatrix x) {
  size_t c_size = Wty.size;
  double d;

  double xPwx = vector_ddot(x, x);
  double xPwy = vector_ddot(x, y);

  DMatrix WtWiWtx = matrix_mult(WtWi, Wtx);

  d = vector_ddot(WtWiWtx, Wtx);
  xPwx -= d;

  d = vector_ddot(WtWiWtx, Wty);
  xPwy -= d;

  return vPvResult(xPwx, xPwy);
}

double CalcvPv(const DMatrix WtWi, const DMatrix Wty, const DMatrix y) {
  double yPwy = vector_ddot(y, y);
  DMatrix WtWiWty = matrix_mult(WtWi, Wty);

  double d = vector_ddot(WtWiWty, Wty);
  yPwy -= d;

  return yPwy;
}

// Calculate p-values and beta/se in a linear model.
alias Tuple!(const double, "beta", const double, "se", const double, "p_wald",
       const double, "p_lrt", const double, "p_score")Pvals;

Pvals LmCalcP(const size_t lm_mode, const double yPwy, const double xPwy,
             const double xPwx, const double df, const size_t n_size) {
  double yPxy = yPwy - xPwy * xPwy / xPwx;
  double beta = xPwy / xPwx;
  double se_wald = sqrt(yPxy / (df * xPwx));
  double se_score = sqrt(yPwy / (to!double(n_size) * xPwx));
  double p_wald = gsl_cdf_fdist_Q(beta * beta / (se_wald * se_wald), 1.0, df);
  double p_score = gsl_cdf_fdist_Q(beta * beta / (se_score * se_score), 1.0, df);
  double p_lrt = gsl_cdf_chisq_Q(to!double(n_size) * (mlog(yPwy) - mlog(yPxy)), 1);
  double se = (lm_mode == 3 ? se_score : se_wald);
  return Pvals(beta, se, p_wald, p_lrt, p_score);
}

void LmCalcP2(const DMatrix WtWi, const DMatrix Wty,
              const DMatrix Wtx_collect, const DMatrix y, const DMatrix x_collect,
              const size_t lm_mode, const double yPwy, const double df, const size_t n_size) {

  size_t c_size = Wty.size;
  DMatrix xPwx = matrix_mult(x_collect, x_collect.T).get_diagonal;
  DMatrix xPwy = matrix_mult(x_collect, y);
  DMatrix WtWiWtx = matrix_mult(WtWi, Wtx_collect.T);

  DMatrix d = matrix_mult(WtWiWtx.T, Wtx_collect.T).get_diagonal; //check shape
  xPwx = xPwx - d; // subtract_dmatrix
  DMatrix d2 = matrix_mult(WtWiWtx.T, Wty);
  xPwy = subtract_dmatrix(xPwy, d2); // subtract_dmatrix_num

  DMatrix yPxy = subtract_num_dmatrix(yPwy, (xPwy * (xPwy / xPwx))); // matrix
  DMatrix beta_matrix = xPwy / xPwx; // beta
  DMatrix xPwx_se = multiply_dmatrix_num(xPwx, n_size);
  DMatrix wald = yPxy / multiply_dmatrix_num(xPwx, df);

  SUMSTAT[] sumStat;
  foreach(i, beta; beta_matrix.elements){
    double se_score = sqrt(yPwy / xPwx_se.elements[i]);
    double se_wald = sqrt(wald.elements[i]);
    double p_wald = gsl_cdf_fdist_Q(beta * beta / (se_wald * se_wald), 1.0, df);
    double p_score = gsl_cdf_fdist_Q(beta * beta / (se_score * se_score), 1.0, df);
    double p_lrt = gsl_cdf_chisq_Q(to!double(n_size) * (mlog(yPwy) - mlog(yPxy.elements[i])), 1);
    double se = (lm_mode == 3 ? se_score : se_wald);
    sumStat ~= SUMSTAT(beta, se, 0.0, 0.0, p_wald, p_lrt, p_score, -0.0);
  }
  writeln(sumStat[0]);
  //return sumStat;
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
  double df = to!double(W.shape[0]) - to!double(W.shape[1]) - 1.0;

  DMatrix y; // = gsl_vector_alloc(W,shape[0]);

  DMatrix WtW = matrix_mult(W.T, W);
  DMatrix WtWi = WtW.inverse();

  DMatrix Wtx =  matrix_mult(W.T, x);
  double xPwx = CalcvPv(WtWi, Wtx, x);

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
    auto vPv = CalcvPv(WtWi, Wtx, Wty, x, y);
    Pvals p = LmCalcP(lm_mode, vPv.y, vPv.x, xPwx, df, W.shape[0]);

    // Store summary data.
    SUMSTAT SNPs = SUMSTAT(p.beta, p.se, 0.0, 0.0, p.p_wald, p.p_lrt, p.p_score, -0.0);
    sumStat ~= SNPs;
  }

  return;
}

SUMSTAT[] lm_analyze_bimbam(const string file_geno, const DMatrix W, const DMatrix y,
                            const DMatrix WtWi, const DMatrix Wty,
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

  double yPwy = CalcvPv(WtWi, Wty, y);

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

    // Calculate statistics.
    DMatrix Wtx = matrix_mult(W.T, x);

    auto vPv = CalcvPv(WtWi, Wty, Wtx, y, x);
    auto p = LmCalcP(lm_mode, yPwy, vPv.y, vPv.x, df, W.shape[0]);

    // Store summary data.
    SUMSTAT SNPs = SUMSTAT(p.beta, p.se, 0.0, 0.0, p.p_wald, p.p_lrt, p.p_score, -0.0);
    sumStat ~= SNPs;
    t++;
  }

  if(file_geno == "data/gemma/BXD_geno.txt.gz"){
    check_lm_results(sumStat);
  }

  return sumStat;
}

SUMSTAT[] lm_analyze_bimbam_batched(const string file_geno, const DMatrix W, const DMatrix y,
                                    const DMatrix WtWi, const DMatrix Wty,
                                    const int[] indicator_idv, const int[] indicator_snp,
                                    const size_t ni_test, const size_t ni_total) {

  writeln("entered lm_analyze_bimbam");
  version(PARALLEL){
    auto task_pool = new TaskPool(totalCPUs);
  }

  size_t msize=5000;

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

  double yPwy = CalcvPv(WtWi, Wty, y);

  // Start reading genotypes and analyze.
  int t = 0;
  size_t c = 0;
  size_t batch_count = 0;
  size_t total_snps = indicator_snp.length;
  size_t expected_batch_count = total_snps / msize;
  size_t t_last = total_snps;
  DMatrix x_collect = DMatrix([msize, ni_test],[]);
  DMatrix Wtx_collect = DMatrix([msize, W.shape[1]], []);
  if(expected_batch_count == batch_count){ msize = total_snps % msize;}
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

    for (size_t i = 0; i < ni_test; ++i) {
      if (x_miss.elements[i] == 0) {
        x.elements[i] = x_mean;
      }
    }

    // Calculate statistics.
    DMatrix Wtx = matrix_mult(W.T, x);

    x_collect.elements ~= x.elements;
    Wtx_collect.elements ~= Wtx.elements;
    c++;

    if(c%5000 == 0 || c==t_last){
      version(PARALLEL){
        auto taskk = task(&LmCalcP2, WtWi, Wty, Wtx_collect, y, x_collect, lm_mode,  yPwy, df, W.shape[0]);
        task_pool.put(taskk);
      }
      else{
        LmCalcP2(WtWi, Wty, Wtx_collect, y, x_collect, lm_mode,  yPwy, df, W.shape[0]);
      }
      batch_count++;
      if(expected_batch_count == batch_count){ msize = total_snps % msize;}
      x_collect = DMatrix([msize , ni_test],[]);
      Wtx_collect = DMatrix([msize, W.shape[1]],[]);
    }
    t++;
  }

  //sumStat = LmCalcP2(WtWi, Wty, Wtx_collect,  y, x_collect_mat,lm_mode,  yPwy, df, W.shape[0]);
  version(PARALLEL){
    task_pool.finish(true);
  }
  if(file_geno == "data/gemma/BXD_geno.txt.gz"){
    check_lm_results(sumStat);
  }

  return sumStat;
}

void lm_analyze_bgen( const string file_oxford, const DMatrix W, const DMatrix y,
                      const DMatrix WtWi, const DMatrix Wty,
                      const int[] indicator_idv, const int[] indicator_snp,
                      const size_t ni_test, const size_t ni_total){

  string file_bgen = file_oxford ~ ".bgen";
  File infile = File(file_bgen);

  SUMSTAT[] sumStat;

  string line;
  char *ch_ptr;

  int lm_mode = 0;
  double beta=0, se=0, p_wald=0, p_lrt=0, p_score=0;
  int n_miss, c_phen;
  double geno, x_mean;

  //calculate some basic quantities
  double yPwy, xPwy, xPwx;
  double df= to!double(W.shape[0] - W.shape[1] - 1.0);

  DMatrix x =  zeros_dmatrix(1, W.shape[0]);
  DMatrix x_miss = zeros_dmatrix(1, W.shape[1]);

  DMatrix WtW = zeros_dmatrix(W.shape[1], W.shape[1]);
  DMatrix Wtx = zeros_dmatrix(1, W.shape[1]);

  WtW = matrix_mult(W.T, W);

  yPwy = CalcvPv(WtWi, Wty, y);

  // read in header
  uint* bgen_snp_block_offset;
  uint* bgen_header_length;
  uint* bgen_nsamples;
  uint* bgen_nsnps;
  uint* bgen_flags;
  writeln("ALL SET!");

  //infile.read(reinterpret_cast<char*>(&bgen_snp_block_offset),4);
  bgen_snp_block_offset = cast(uint*)infile.rawRead(new char[4]);

  //infile.read(reinterpret_cast<char*>(&bgen_header_length),4);
  bgen_header_length = cast(uint*)infile.rawRead(new char[4]);
  bgen_snp_block_offset -= 4;

  //infile.read(reinterpret_cast<char*>(&bgen_nsnps),4);
  bgen_nsnps = cast(uint*)infile.rawRead(new char[4]);
  bgen_snp_block_offset -= 4;

  //infile.read(reinterpret_cast<char*>(&bgen_nsamples),4);
  bgen_nsamples = cast(uint*)infile.rawRead(new char[4]);
  bgen_snp_block_offset-=4;

  //infile.ignore( 4 + bgen_header_length - 20);
  size_t ignore = 4 + cast(size_t)bgen_header_length - 20; // check
  infile.rawRead( new char[ignore]);

  bgen_snp_block_offset -= ignore;

  //infile.read(reinterpret_cast<char*>(&bgen_flags),4);
  bgen_flags = cast(uint*)infile.rawRead(new char[4]);
  bgen_snp_block_offset -= 4;

  bool CompressedSNPBlocks; // check TODO  = bgen_flags&0x1;
//  bool LongIds=bgen_flags&0x4; // Prev. commented

  //infile.ignore(bgen_snp_block_offset);
  infile.rawRead(new char[cast(size_t)bgen_snp_block_offset]);

  double bgen_geno_prob_AA,
         bgen_geno_prob_AB,
         bgen_geno_prob_BB,
         bgen_geno_prob_non_miss;

  uint* bgen_N;
  ushort* bgen_LS;
  ushort* bgen_LR;
  ushort* bgen_LC;
  uint* bgen_SNP_pos;
  uint* bgen_LA;
  string bgen_A_allele;
  uint* bgen_LB;
  string bgen_B_allele;
  uint* bgen_P;
  size_t unzipped_data_size;
  string id;
  string rs;
  string chr;
  writeln("Warning: WJA hard coded SNP missingness threshold of 10%");

  //start reading genotypes and analyze
  for (size_t t=0; t < indicator_snp.length; ++t)
  {

//    if (t>1) {break;}

    // read SNP header
    id = [];
    rs = [];
    chr = [];
    bgen_A_allele = [];
    bgen_B_allele = [];

    //infile.read(reinterpret_cast<char*>(&bgen_N),4);
    //infile.read(reinterpret_cast<char*>(&bgen_LS),2);
    bgen_N = cast(uint*)infile.rawRead(new char[4]);
    bgen_LS = cast(ushort*)infile.rawRead(new char[2]);

    //id.resize(bgen_LS);
    //infile.read(&id[0], bgen_LS);

    bgen_LR = cast(ushort*)infile.rawRead(new char[2]);
    //rs.resize(bgen_LR);
    rs = cast(string)infile.rawRead(new char[cast(size_t)bgen_LR]);

    bgen_LC = cast(ushort*)infile.rawRead(new char[2]);
    //chr.resize(bgen_LC);
    chr = cast(string)infile.rawRead(new char[cast(size_t)bgen_LC]);

    bgen_SNP_pos = cast(uint*)infile.rawRead(new char[4]);

    bgen_LA = cast(uint*)infile.rawRead(new char[4]);
    //bgen_A_allele.resize(bgen_LA);
    bgen_A_allele = cast(string)infile.rawRead(new char[cast(size_t)bgen_LA]);


    bgen_LB = cast(uint*)infile.rawRead(new char[4]);
    //bgen_B_allele.resize(bgen_LB);
    bgen_B_allele = cast(string)infile.rawRead(new char[cast(size_t)bgen_LB]);

    ushort* unzipped_data;// = new ushort[3 * cast(size_t)bgen_N];

    if (indicator_snp[t]==0) {
      if(CompressedSNPBlocks)
        bgen_P = cast(uint*)infile.rawRead(new char[4]);
      else
        //bgen_P = cast(uint*)(6 * bgen_N);

      infile.rawRead(new char[cast(size_t)bgen_P]);

      continue;
    }


    if(CompressedSNPBlocks)
    {
      //infile.read(reinterpret_cast<char*>(&bgen_P),4);
      bgen_P = cast(uint*)infile.rawRead(new char[4]);
      ushort* zipped_data; // = new ushort[cast(size_t)bgen_P];

      //unzipped_data_size=6*bgen_N;
      //infile.read(reinterpret_cast<char*>(zipped_data),bgen_P);
      unzipped_data_size= 6 * cast(size_t)bgen_N;
      zipped_data = cast(ushort*)infile.rawRead(new char[cast(size_t)bgen_P]);

      //int result = uncompress(reinterpret_cast<Bytef*>(unzipped_data), reinterpret_cast<uLongf*>(&unzipped_data_size), reinterpret_cast<Bytef*>(zipped_data), static_cast<uLong> (bgen_P));
      int result; // = uncompress(unzipped_data, unzipped_data_size, zipped_data, bgen_P);
      //assert(result == Z_OK);

    }
    else
    {
      //bgen_P = 6 * cast(uint)bgen_N; TODO
      unzipped_data = cast(ushort*)infile.rawRead(new char[cast(size_t)bgen_P]);
    }

    x_mean=0.0; c_phen=0; n_miss=0;
    x_miss = zeros_dmatrix(x_miss.shape[0], x_miss.shape[1]);

    for (size_t i=0; i < cast(size_t)bgen_N; ++i) {
      if (indicator_idv[i]==0) {continue;}


        bgen_geno_prob_AA = to!double(unzipped_data[i*3])/32768.0;
        bgen_geno_prob_AB = to!double(unzipped_data[i*3+1])/32768.0;
        bgen_geno_prob_BB = to!double(unzipped_data[i*3+2])/32768.0;
        // WJA
        bgen_geno_prob_non_miss=bgen_geno_prob_AA+bgen_geno_prob_AB+bgen_geno_prob_BB;
        if (bgen_geno_prob_non_miss<0.9) {x_miss.elements[c_phen] = 0.0; n_miss++;}
        else {

          bgen_geno_prob_AA /= bgen_geno_prob_non_miss;
          bgen_geno_prob_AB /= bgen_geno_prob_non_miss;
          bgen_geno_prob_BB /= bgen_geno_prob_non_miss;

          geno = 2.0 * bgen_geno_prob_BB + bgen_geno_prob_AB;

          x.elements[c_phen] = geno;
          x_miss.elements[c_phen] = 1.0;
          x_mean+=geno;
      }
      c_phen++;
    }

    x_mean /= to!double(ni_test-n_miss);

    for (size_t i=0; i < ni_test; ++i) {
      if(x_miss.elements[i]==0) {x.elements[i] = x_mean;}
      geno = x.elements[i];
      if (x_mean>1) {
        x.elements[i] = 2 - geno;
      }
    }


    //calculate statistics
    Wtx = matrix_mult(W.T, x.T); // check
    vPvResult res = CalcvPv(WtWi, Wty, Wtx, y, x); // chec xPwy and xPwx order
    Pvals p = LmCalcP(lm_mode, yPwy, res.y, res.x, df, W.shape[0]);
    //store summary data
    SUMSTAT SNPs = SUMSTAT(p.beta, p.se, 0.0, 0.0, p.p_wald, p.p_lrt, p.p_score, 0);
    sumStat ~= SNPs;
  }

  return;
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
  double xPwy, xPwx;
  double df = to!double(W.shape[0]) - to!double(W.shape[1]) - 1.0;

  DMatrix x; // = gsl_vector_alloc(W->size1);

  DMatrix WtW = matrix_mult(W.T, W);
  DMatrix WtWi = WtW.inverse;

  DMatrix Wty = matrix_mult(W.T, y);
  double yPwy = CalcvPv(WtWi, Wty, y);

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
    auto vPv = CalcvPv(WtWi, Wty, Wtx, y, x);
    auto p = LmCalcP(lm_mode, yPwy, vPv.y, vPv.x, df, W.shape[0]);

    // store summary data
    SUMSTAT SNPs = SUMSTAT(p.beta, p.se, 0.0, 0.0, p.p_wald, p.p_lrt, p.p_score, -0.0);
    sumStat ~= SNPs;
  }

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

void check_lm_results(SUMSTAT[] sumStat){
  enforce(modDiff(sumStat[0].beta,    0.174093)<0.001);
  enforce(modDiff(sumStat[0].p_wald,  0.136021)<0.001);
  enforce(modDiff(sumStat[0].p_score, 0.131061,)<0.001);
  enforce(modDiff(sumStat[1].beta,    0.174093)<0.001);
  enforce(modDiff(sumStat[1].p_wald,  0.136021)<0.001);
  enforce(modDiff(sumStat[1].p_score, 0.131061,)<0.001);
  enforce(modDiff(sumStat[2].beta,    0.174093)<0.001);
  enforce(modDiff(sumStat[2].p_wald,  0.136021)<0.001);
  enforce(modDiff(sumStat[2].p_score, 0.131061,)<0.001);

  enforce(modDiff(sumStat[$-3].beta,    0.0896522)<0.001);
  enforce(modDiff(sumStat[$-3].p_wald,  0.46775)<0.001);
  enforce(modDiff(sumStat[$-3].p_score, 0.455906)<0.001);
  enforce(modDiff(sumStat[$-2].beta,    0.145594)<0.001);
  enforce(modDiff(sumStat[$-2].p_wald,  0.22889)<0.001);
  enforce(modDiff(sumStat[$-2].p_score, 0.220097)<0.001);
  enforce(modDiff(sumStat[$-1].beta,    0.145594)<0.001);
  enforce(modDiff(sumStat[$-1].p_wald,  0.22889)<0.001);
  enforce(modDiff(sumStat[$-1].p_score, 0.220097)<0.001);

  writeln("Linear Model tests pass!");
}

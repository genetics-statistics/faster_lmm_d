/*
   This code is part of faster_lmm_d and published under the GPLv3
   License (see LICENSE.txt)

   Copyright © 2017-2018 Prasun Anand & Pjotr Prins
*/

module faster_lmm_d.gemma_io;

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
import std.typecons;
import std.experimental.logger;
import std.string;

import faster_lmm_d.dmatrix;
import faster_lmm_d.gemma_lmm;
import faster_lmm_d.gemma_param;
import faster_lmm_d.gemma_kinship;
import faster_lmm_d.helpers;
import faster_lmm_d.optmatrix;

import gsl.permutation;
import gsl.rng;
import gsl.randist;

alias Tuple!(int[], "indicator_snp", SNPINFO[], "snpInfo") Geno_result;

// Read files: obtain ns_total, ng_total, ns_test, ni_test.
void read_all_files() {
  string file_mcat, file_cat, file_wcat, file_wsnp, file_mk, file_epm, 
         file_bfile, file_snps, file_geno, file_pheno, file_ksnps, file_ebv, 
         file_gxe, file_weight, file_cvt, file_anno, file_log, file_mbfile,
         file_mgeno, file_gene, file_read;
  size_t[string] mapRS2cat, mapRS2wcat, mapRS2bp;
  size_t[size_t] mapID2num;
  double[string] mapRS2wsnp, mapRS2cM;
  string[string] mapRS2chr;
  string[] setSnps;
  string[] setKSnps;
  size_t n_vc, n_cvt, ns_total, ns_test, ni_max, ni_test;
  int[][] indicator_pheno;
  int[] indicator_cvt, indicator_idv, indicator_snp, indicator_gxe,
        mindicator_snp, indicator_read, indicator_bv, indicator_weight;
  size_t[] p_column;
  string file_str;
  double[][] cvt;
  double[][] pheno;
  double pheno_mean;
  double[] vec_bv, vec_read, gxe, weight;
  SNPINFO[] snpInfo, msnpInfo;
  double maf_level, miss_level, hwe_level, r2_level;

  // Read cat file.
  if (file_mcat != "") {
    if (ReadFile_mcat(file_mcat, mapRS2cat, n_vc) == false) {
      error = true;
    }
  } else if (file_cat != "") {
    if (ReadFile_cat(file_cat, mapRS2cat, n_vc) == false) {
      error = true;
    }
  }

  // Read snp weight files.
  if(file_wcat != "") {
    if (ReadFile_wsnp(file_wcat, n_vc, mapRS2wcat) == false) {
      error = true;
    }
  }
  if (file_wsnp != "") {
    if (ReadFile_wsnp(file_wsnp, mapRS2wsnp) == false) {
      error = true;
    }
  }

  // Count number of kinship files.
  if (file_mk != "") {
    if (CountFileLines(file_mk, n_vc) == false) {
      error = true;
    }
  }

  // Read SNP set.
  if (file_snps != "") {
    if (ReadFile_snps(file_snps, setSnps) == false) {
      error = true;
    }
  } else {
    setSnps = [];
  }

  // Read KSNP set.
  if (file_ksnps != "") {
    if (ReadFile_snps(file_ksnps, setKSnps) == false) {
      error = true;
    }
  } else {
    setKSnps = [];
  }

  // For prediction.
  if (file_epm != "") {
    //if (ReadFile_est(file_epm, est_column, mapRS2est) == false) {
    //  error = true;
    //}
    if (file_bfile != "") {
      file_str = file_bfile ~ ".bim";
      if (readfile_bim(file_str, snpInfo) == false) {
        error = true;
      }
      file_str = file_bfile ~ ".fam";
      //if (readfile_fam(file_str, indicator_pheno, pheno, mapID2num, p_column) ==
      //    false) {
      //  error = true;
      //}
    }

    if (file_geno != "") {
      //if (ReadFile_pheno(file_pheno, indicator_pheno, pheno, p_column) == false) {
      //  error = true;
      //}

      if (CountFileLines(file_geno, ns_total) == false) {
        error = true;
      }
    }

    if (file_ebv != "") {
      if (ReadFile_column(file_ebv, indicator_bv, vec_bv, 1) == false) {
        error = true;
      }
    }

    if (file_log != "") {
      if (ReadFile_log(file_log, pheno_mean) == false) {
        error = true;
      }
    }

    // Convert indicator_pheno to indicator_idv.
    int k = 1;
    for (size_t i = 0; i < indicator_pheno.length; i++) {
      k = 1;
      for (size_t j = 0; j < indicator_pheno[i].length; j++) {
        if (indicator_pheno[i][j] == 0) {
          k = 0;
        }
      }
      indicator_idv ~= k;
    }

    ns_test = 0;

    return;
  }

  // Read covariates before the genotype files.
  if (file_cvt != "") {
    if (readfile_cvt(file_cvt, indicator_cvt, cvt, n_cvt) == false) {
      error = true;
    }
    if (indicator_cvt.length == 0) {
      n_cvt = 1;
    }
  } else {
    n_cvt = 1;
  }
  trim_individuals(indicator_cvt, ni_max);

  if (file_gxe != "") {
    if (ReadFile_column(file_gxe, indicator_gxe, gxe, 1) == false) {
      error = true;
    }
  }
  if (file_weight != "") {
    if (ReadFile_column(file_weight, indicator_weight, weight, 1) == false) {
      error = true;
    }
  }

  trim_individuals(indicator_idv, ni_max);

  // Read genotype and phenotype file for PLINK format.
  if (!file_bfile.empty()) {
    file_str = file_bfile ~ ".bim";
    snpInfo = [];
    if (readfile_bim(file_str, snpInfo) == false) {
      error = true;
    }

    // If both fam file and pheno files are used, use
    // phenotypes inside the pheno file.
    if (file_pheno != "") {

      // Phenotype file before genotype file.
      //if (ReadFile_pheno(file_pheno, indicator_pheno, pheno, p_column) == false) {
      //  error = true;
      //}
    } else {
      file_str = file_bfile ~ ".fam";
      //if (readfile_fam(file_str, p_column) == false) {
      //  error = true;
      //}
    }

    // Post-process covariates and phenotypes, obtain
    // ni_test, save all useful covariates.
    //process_cvt_phen();

    // Obtain covariate matrix.
    DMatrix W1;
    //CopyCvt(W1);

    file_str = file_bfile ~ ".bed";
    //if (readfile_bed(file_str, setSnps, W1, indicator_idv, indicator_snp,
    //                 snpInfo, maf_level, miss_level, hwe_level, r2_level,
    //                 ns_test) == false) {
    //  error = true;
    //}
    ns_total = indicator_snp.length;
  }

  // Read genotype and phenotype file for BIMBAM format.
  if (file_geno != "") {

    // Annotation file before genotype file.
    if (file_anno != "") {
      if (ReadFile_anno(file_anno, mapRS2chr, mapRS2bp, mapRS2cM) == false) {
        error = true;
      }
    }

    // Phenotype file before genotype file.
    //if (ReadFile_pheno(file_pheno, indicator_pheno, pheno, p_column) == false) {
    //  error = true;
    //}

    // Post-process covariates and phenotypes, obtain
    // ni_test, save all useful covariates.
    //process_cvt_phen();

    // Obtain covariate matrix.
    DMatrix W2; // = gsl_matrix_safe_alloc(ni_test, n_cvt);
    //CopyCvt(W2);

    trim_individuals(indicator_idv, ni_max);
    trim_individuals(indicator_cvt, ni_max);
    //if (ReadFile_geno(file_geno, setSnps, W2, indicator_idv, indicator_snp,
    //                  maf_level, miss_level, hwe_level, r2_level, mapRS2chr,
    //                  mapRS2bp, mapRS2cM, snpInfo, ns_test) == false) {
    //  error = true;
    //}

    ns_total = indicator_snp.length;
  }

  // Read genotype file for multiple PLINK files.
  if (!file_mbfile.empty()) {
    File infile = File(file_mbfile);

    size_t t = 0, ns_test_tmp = 0;
    DMatrix W3;
    foreach(file_name; infile.byLine){
      file_str = to!string(file_name) ~ ".bim";

      if (readfile_bim(file_str, snpInfo) == false) {
        error = true;
      }

      if (t == 0) {

        // If both fam file and pheno files are used, use
        // phenotypes inside the pheno file.
        if (!file_pheno.empty()) {

          // Phenotype file before genotype file.
          //if (ReadFile_pheno(file_pheno, indicator_pheno, pheno, p_column) == false) {
          //  error = true;
          //}
        } else {
          file_str = to!string(file_name) ~ ".fam";
          //if (readfile_fam(file_str, indicator_pheno, pheno, mapID2num, p_column) == false) {
          //  error = true;
          //}
        }

        // Post-process covariates and phenotypes, obtain
        // ni_test, save all useful covariates.
        DMatrix mock;
        process_cvt_phen(mock);

        // Obtain covariate matrix.
        //W3 = gsl_matrix_safe_alloc(ni_test, n_cvt);
        //CopyCvt(W3);
      }

      file_str = to!string(file_name) ~ ".bed";
      //if (readfile_bed(file_str, setSnps, W3, indicator_idv, indicator_snp,
      //                 snpInfo, maf_level, miss_level, hwe_level, r2_level,
      //                 ns_test_tmp) == false) {
      //  error = true;
      //}
      mindicator_snp ~= indicator_snp;
      msnpInfo ~= snpInfo;
      ns_test += ns_test_tmp;
      ns_total += indicator_snp.length;

      t++;
    }

  }

  // Read genotype and phenotype file for multiple BIMBAM files.
  if (file_mgeno != "") {

    // Annotation file before genotype file.
    if (file_anno != "") {
      if (ReadFile_anno(file_anno, mapRS2chr, mapRS2bp, mapRS2cM) == false) {
        error = true;
      }
    }

    // Phenotype file before genotype file.
    //if (ReadFile_pheno(file_pheno, indicator_pheno, pheno, p_column) == false) {
    //  error = true;
    //}

    // Post-process covariates and phenotypes, obtain ni_test,
    // save all useful covariates.
    DMatrix mock;
    process_cvt_phen(mock);

    // Obtain covariate matrix.
    DMatrix W4;// = gsl_matrix_safe_alloc(ni_test, n_cvt);
    //CopyCvt(W4);

    File infile = File(file_mgeno);
    //if (!infile) {
    //  writeln("error! fail to open mgeno file: ", file_mgeno);
    //  error = true;
    //  return;
    //}

    size_t ns_test_tmp;
    foreach(file_name; infile.byLine) {
      //if (ReadFile_geno(file_name, setSnps, W4, indicator_idv, indicator_snp,
      //                  maf_level, miss_level, hwe_level, r2_level, mapRS2chr,
      //                  mapRS2bp, mapRS2cM, snpInfo, ns_test_tmp) == false) {
      //  error = true;
      //}

      mindicator_snp ~= indicator_snp;
      msnpInfo ~= snpInfo;
      ns_test += ns_test_tmp;
      ns_total += indicator_snp.length;
    }

  }

  if (file_gene != "") {
    //if (ReadFile_pheno(file_pheno, indicator_pheno, pheno, p_column) == false) {
    //  error = true;
    //}

    // Convert indicator_pheno to indicator_idv.
    int k = 1;
    for (size_t i = 0; i < indicator_pheno.length; i++) {
      k = 1;
      for (size_t j = 0; j < indicator_pheno[i].length; j++) {
        if (indicator_pheno[i][j] == 0) {
          k = 0;
        }
      }
      indicator_idv ~= k;
    }

    // Post-process covariates and phenotypes, obtain
    // ni_test, save all useful covariates.
    DMatrix mock;
    process_cvt_phen(mock);

    // Obtain covariate matrix.
    // gsl_matrix *W5 = gsl_matrix_alloc(ni_test, n_cvt);
    // CopyCvt(W5);

    //if (ReadFile_gene(file_gene, vec_read, snpInfo, ng_total) == false) {
    //  error = true;
    //}
  }

  // Read is after gene file.
  if (file_read != "") {
    if (ReadFile_column(file_read, indicator_read, vec_read, 1) == false) {
      error = true;
    }

    ni_test = 0;
    for (size_t i = 0; i < indicator_idv.length; ++i) {
      indicator_idv[i] *= indicator_read[i];
      ni_test += indicator_idv[i];
    }

    if (ni_test == 0) {
      writeln("error! number of analyzed individuals equals 0. ");
      error = true;
      return;
    }
  }

  // For ridge prediction, read phenotype only.
  if (file_geno == "" && file_gene == "" && file_pheno != "") {
    //if (ReadFile_pheno(file_pheno, indicator_pheno, pheno, p_column) == false) {
    //  error = true;
    //}

    // Post-process covariates and phenotypes, obtain
    // ni_test, save all useful covariates.
    DMatrix mock;
    process_cvt_phen(mock);
  }

  // Compute setKSnps when -loco is passed in
  //if (!loco.empty()) {
  //  LOCO_set_Snps(setKSnps, setGWASnps, mapRS2chr, loco);
  //}
  return;
}

// Read SNP file. A single column of SNP names.
bool ReadFile_snps(const string file_snps, string[] setSnps) {
  writeln("entered ReadFile_snps");
  setSnps = [];

  File infile = File(file_snps);

  foreach(line; infile.byLine) {
    //string[] ch_ptr = line.split("\t");
    //setSnps ~= ch_ptr;
  }

  return true;
}

// Trim #individuals to size which is used to write tests that run faster
//
// Note it actually trims the number of functional individuals
// (indicator_idv[x] == 1). This should match indicator_cvt etc. If
// this gives problems with certain sets we can simply trim to size.

void trim_individuals(int[] idvs, size_t ni_max) {
  if (ni_max > 0) {
    size_t count = 0;
    foreach(ind; idvs) {
      if (ind){
        count++;
      }
      if (count >= ni_max){
        break;
      }
    }
    if (count != idvs.length) {
      writeln("**** TEST MODE: trim individuals from ", idvs.length, " to ", count);
      //idvs.resize(count);
    }
  }
}

// Read var file, store mapRS2wsnp.
bool ReadFile_wsnp(const string file_wsnp, double[string] mapRS2weight) {
  writeln("entered ReadFile_wsnp");
  //mapRS2weight = [];

  File infile = File(file_wsnp);

  string rs;
  double weight;

  foreach(line; infile.byLine){
    auto ch_ptr = line.split("\t");
    rs = to!string(ch_ptr[0]);
    weight = to!double(ch_ptr[1]);
    mapRS2weight[rs] = weight;
  }

  return true;
}

bool ReadFile_wsnp(const string file_wcat, const size_t n_vc, size_t[string] mapRS2wvector) {
  writeln("entered ReadFile_wsnp");
  writeln("TODO");
  return true;
}

bool CountFileLines(const string file_input, size_t n_lines) {
  return true;
}


bool readfile_cvt(const string file_cvt, int[] indicator_cvt,
                  double[][] cvt, size_t n_cvt) {
  writeln("entered readfile_cvt");
  //indicator_cvt.clear();

  File infile = File(file_cvt);
  double d;

  int flag_na = 0;

  foreach(line; infile.byLine) {
    double[] v_d;
    flag_na = 0;
    auto chrs = line.split("\t");
    foreach(ch_ptr; chrs) {
      if (ch_ptr == "NA") {
        flag_na = 1;
        d = -9;
      } else {
        d = to!double(ch_ptr);
      }

      v_d ~= d;
    }
    if (flag_na == 0) {
      indicator_cvt ~= 1;
    } else {
      indicator_cvt ~= 0;
    }
    cvt ~= v_d;
  }

  if (indicator_cvt.length == 0) {
    n_cvt = 0;
  } else {
    flag_na = 0;
    foreach (i, ind; indicator_cvt) {
      if (indicator_cvt[i] == 0) {
        continue;
      }

      if (flag_na == 0) {
        flag_na = 1;
        n_cvt = cvt[i].length;
      }
      if (flag_na != 0 && n_cvt != cvt[i].length) {
        writeln("error! number of covariates in row ", i, " do not match other rows.");
        return false;
      }
    }
  }

  return true;
}

// Read bimbam mean genotype file, the first time, to obtain #SNPs for
// analysis (ns_test) and total #SNP (ns_total).
Geno_result ReadFile_geno1(const string geno_fn, const ulong ni_total, const DMatrix W, const int[] indicator_idv){

  writeln("ReadFile_geno", geno_fn);
  int[] indicator_snp;

  string[string] mapRS2chr;
  long[string] mapRS2bp, mapRS2cM;

  string[] setSnps;

  size_t ns_test;
  SNPINFO[] snpInfo;
  const double maf_level = 0.01;
  const double miss_level = 0.05;
  const double hwe_level = 0;
  const double r2_level = 0.9999;


  string filename = geno_fn;
  auto pipe = pipeShell("gunzip -c " ~ filename);
  File input = pipe.stdout;

  double[] genotype = new double[W.shape[0]];
  double[] genotype_miss = new double[W.shape[0]];

  // W refers to covariates
  double WtWi= 1/vector_ddot(W, W);

  int c_idv = 0;
  int n_0, n_1, n_2, flag_poly;
  long b_pos;
  double v_x, v_w, maf, geno, geno_old, cM;
  string rs, chr, major, minor;
  size_t file_pos, n_miss;

  int ni_test = 0;
  foreach (element; indicator_idv) {
    ni_test += element;
  }
  ns_test = 0;

  file_pos = 0;
  auto count_warnings = 0;
  foreach (line ; input.byLine) {
    auto ch_ptr = to!string(line).split(",");
    rs = ch_ptr[0];
    minor = ch_ptr[1];
    major = ch_ptr[2];

    auto chr_val = ch_ptr[3..$];

    if (setSnps.length != 0 && setSnps.count(rs) == 0) {
      // if SNP in geno but not in -snps we add an missing value
      SNPINFO sInfo = SNPINFO("-9", rs, -9, -9, minor, major,
                                0,  -9, -9, 0, 0, file_pos);
      snpInfo ~= sInfo;
      indicator_snp ~= 0;

      file_pos++;

      continue;
    }
    if (mapRS2bp.get(rs, 0) != -1) { // check
      if (count_warnings++ < 10) {
        writeln("Can't figure out position for ");
      }
      chr = "-9";
      b_pos = -9;
      cM = -9;
    } else {
      b_pos = mapRS2bp[rs];
      chr = mapRS2chr[rs];
      cM = mapRS2cM[rs];
    }

    maf = 0;
    n_miss = 0;
    flag_poly = 0;
    geno_old = -9;
    n_0 = 0;
    n_1 = 0;
    n_2 = 0;
    c_idv = 0;
    foreach(ref ele; genotype_miss){ele = 0;}
    foreach (i, idv; indicator_idv) {
      if (idv == 0)
        continue;
      auto digit = to!string(chr_val[i].strip());
      if (digit == "NA") {
        genotype_miss[c_idv] = 1;
        n_miss++;
        c_idv++;
        continue;
      }

      geno = to!double(digit);
      if (geno >= 0   && geno <= 0.5){ n_0++; }
      if (geno > 0.5  && geno <  1.5){ n_1++; }
      if (geno >= 1.5 && geno <= 2.0){ n_2++; }

      genotype[c_idv] = geno;

      if (flag_poly == 0) {
        geno_old = geno;
        flag_poly = 2;
      }

      if (flag_poly == 2 && geno != geno_old) { flag_poly = 1; }

      maf += geno;

      c_idv++;
    }

    maf /= 2.0 * to!double(ni_test - n_miss);

    snpInfo ~= SNPINFO(chr,    rs,
                       cM,     b_pos,
                       minor,  major,
                       n_miss, to!double(n_miss) / to!double(ni_test),
                       maf,    ni_test - n_miss,
                       0,      file_pos);
    file_pos++;

    if (to!double(n_miss) / to!double(ni_test) > miss_level) {
      indicator_snp ~= 0;
      continue;
    }

    if ((maf < maf_level || maf > (1.0 - maf_level)) && maf_level != -1) {
      indicator_snp ~= 0;
      continue;
    }

    if (flag_poly != 1) {
      indicator_snp ~= 0;
      continue;
    }

    if (hwe_level != 0 && maf_level != -1) {
      if (CalcHWE(n_0, n_2, n_1) < hwe_level) {
        indicator_snp ~=0;
        continue;
      }
    }

    // Filter SNP if it is correlated with W unless W has
    // only one column, of 1s.
    for (size_t i = 0; i < genotype.length; ++i) {
      if (genotype_miss[i] == 1) {
        geno = maf * 2.0;
        genotype[i] = geno;
      }
    }

    double Wtx = vector_ddot(W.elements, genotype);

    v_x = vector_ddot(genotype, genotype);
    v_w = Wtx * Wtx * WtWi;

    if (W.shape[1] != 1 && v_w / v_x >= r2_level) {
      indicator_snp ~= 0;
      continue;
    }

    indicator_snp ~= 1;
    ns_test++;
  }
  return Geno_result(indicator_snp, snpInfo);
}


// Read bimbam annotation file which consists of rows of SNP, POS and CHR
bool ReadFile_anno(const string file_anno, string[string] mapRS2chr,
                   size_t[string] mapRS2bp, double[string] mapRS2cM) {
  writeln("ReadFile_anno");
  //mapRS2chr = [];
  //mapRS2bp = [];

  File infile = File(file_anno);
  //if (!infile) {
  //  writeln("error opening annotation file: ", file_anno);
  //  return false;
  //}

  foreach(line; infile.byLine) {
    auto ch_ptr = line.split("\t");
    //enforce_str(ch_ptr, line + " Bad RS format");
    const string rs = to!string(ch_ptr[0]);
    //enforce_str(rs != "", line + " Bad RS format");

    //enforce_str(ch_ptr[1], line + " Bad format");
    long b_pos;
    if (ch_ptr[1] == "NA"){
      b_pos = -9;
    } else {
      b_pos = to!long(ch_ptr[1]);
    }
    //enforce_str(b_pos, line + " Bad pos format (is zero)");

    string chr;
    if (ch_ptr[2] == "NULL" || ch_ptr[2] == "NA") { // TODO
      chr = "-9";
    } else {
      chr = to!string(ch_ptr[2]);
      //enforce_str(chr != "", line + " Bad chr format");
    }

    double cM;
    if (ch_ptr[3] == "NULL" || ch_ptr[2] == "NA") { // TODO
      cM = -9;
    } else {
      cM = to!double(ch_ptr[3]);
      //enforce_str(b_pos, line + "Bad cM format (is zero)");
    }

    mapRS2chr[rs] = chr;
    mapRS2bp[rs] = b_pos;
    mapRS2cM[rs] = cM;
  }

  return true;
}


// Read .bim file.
bool readfile_bim(const string file_bim, SNPINFO[] snpInfo) {
  writeln("entered readfile_bim");
  //snpInfo.clear();

  File infile = File(file_bim);

  string rs;
  long   b_pos;
  string chr;
  double cM;
  string major;
  string minor;

  foreach(line; infile.byLine) {
    auto ch_ptr = line.split("\t");
    chr   = to!string(ch_ptr[0]);
    rs    = to!string(ch_ptr[1]);
    cM    = to!double(ch_ptr[2]);
    b_pos = to!long(ch_ptr[3]);
    minor = to!string(ch_ptr[4]);
    major = to!string(ch_ptr[5]);

    SNPINFO sInfo = SNPINFO(chr, rs, cM, b_pos, minor, major, 0, -9, -9, 0, 0, 0);
    snpInfo ~= sInfo;
  }

  return true;
}

// Read bed file, the first time.
int[] readfile_bed(const string file_bed, const string[] setSnps,
                  const DMatrix W, int[] indicator_idv, SNPINFO[] snpInfo,
                  const double maf_level, const double miss_level,
                  const double hwe_level, const double r2_level,
                  size_t ns_test) {
  writeln("entered readfile_bed");
  int[] indicator_snp = [];
  writeln("booyakasha");
  size_t ns_total = snpInfo.length;
  ns_total = 200;

  File infile = File(file_bed);

  DMatrix genotype = zeros_dmatrix(1, W.shape[0]);
  DMatrix genotype_miss;
  genotype_miss.shape = [1, W.shape[0]];

  DMatrix WtW = matrix_mult(W.T, W);
  int sig;
  DMatrix WtWi = WtW.inverse;
  double v_x, v_w, geno;
  size_t c_idv = 0;

  //int[] b;

  size_t ni_total = indicator_idv.length;
  writeln("ni_total", ni_total);
  size_t ni_test = 0;
  for (size_t i = 0; i < ni_total; ++i) {
    ni_test += indicator_idv[i];
  }
  writeln("ni_test", ni_test);
  ns_test = 0;

  // Calculate n_bit and c, the number of bit for each snp.
  size_t n_bit;
  if (ni_total % 4 == 0) {
    n_bit = ni_total / 4;
  } else {
    n_bit = ni_total / 4 + 1;
  }

  // Ignore the first three magic numbers.
  int num;
  for (int i = 0; i < 3; ++i) {
    // FIXME
    auto ch = infile.rawRead(new char[8]);
    writeln(to!int(ch));
  }

  double maf;
  size_t n_miss;
  size_t n_0, n_1, n_2, c;

  // Start reading snps and doing association test.
  for (size_t t = 0; t < ns_total; ++t) {

    // n_bit, and 3 is the number of magic numbers.
    infile.seek(t * n_bit + 3);

    if (setSnps.length != 0 && setSnps.count(snpInfo[t].rs_number) == 0) {
      snpInfo[t].n_miss = -9;
      snpInfo[t].missingness = -9;
      snpInfo[t].maf = -9;
      snpInfo[t].file_position = t;
      indicator_snp ~= 0;
      continue;
    }

    // Read genotypes.
    c = 0;
    maf = 0.0;
    n_miss = 0;
    n_0 = 0;
    n_1 = 0;
    n_2 = 0;
    c_idv = 0;
    genotype_miss = zeros_dmatrix(genotype_miss.shape[0], genotype_miss.shape[1]);
    for (size_t i = 0; i < n_bit; ++i) {
      // fixme
      auto b = infile.rawRead(new char[8]);

      // Minor allele homozygous: 2.0; major: 0.0;
      for (size_t j = 0; j < 4; ++j) {
        if ((i == (n_bit - 1)) && c == ni_total) {
          break;
        }
        if (indicator_idv[c] == 0) {
          c++;
          continue;
        }
        c++;

        if (b[2 * j] == 0) {
          if (b[2 * j + 1] == 0) {
            genotype.elements[c_idv] = 2.0;
            maf += 2.0;
            n_2++;
          } else {
            genotype.elements[c_idv] = 1.0;
            maf += 1.0;
            n_1++;
          }
        } else {
          if (b[2 * j + 1] == 1) {
            genotype.elements[c_idv] = 0;
            maf += 0.0;
            n_0++;
          } else {
            genotype_miss.elements[c_idv] = 1;
            n_miss++;
          }
        }
        c_idv++;
      }
    }
    maf /= 2.0 * to!double(ni_test - n_miss);

    //snpInfo[t].n_miss = n_miss;
    //snpInfo[t].missingness = to!double(n_miss) / to!double(ni_test);
    //snpInfo[t].maf = maf;
    //snpInfo[t].n_idv = ni_test - n_miss;
    //snpInfo[t].n_nb = 0;
    //snpInfo[t].file_position = t;

    if (to!double(n_miss) / to!double(ni_test) > miss_level) {
      indicator_snp ~= 0;
      continue;
    }

    if ((maf < maf_level || maf > (1.0 - maf_level)) && maf_level != -1) {
      indicator_snp ~= 0;
      continue;
    }

    if ((n_0 + n_1) == 0 || (n_1 + n_2) == 0 || (n_2 + n_0) == 0) {
      indicator_snp ~= 0;
      continue;
    }

    if (hwe_level != 0 && maf_level != -1) {
      if (CalcHWE(to!int(n_0), to!int(n_2), to!int(n_1)) < hwe_level) {
        indicator_snp ~= 0;
        continue;
      }
    }

    // Filter SNP if it is correlated with W unless W has
    // only one column, of 1s.
    for (size_t i = 0; i < genotype.size; ++i) {
      if (genotype_miss.elements[i] == 1) {
        geno = maf * 2.0;
        genotype.elements[i] = geno;
      }
    }

    DMatrix Wtx = matrix_mult(W.T, genotype);
    DMatrix WtWiWtx = matrix_mult(WtWi, Wtx);
    v_x = vector_ddot(genotype, genotype);
    v_w = vector_ddot(Wtx, WtWiWtx);

    if (W.shape[1] != 1 && v_w / v_x > r2_level) {
      indicator_snp ~= 0;
      continue;
    }

    indicator_snp ~= 1;
    ns_test++;
  }

  return indicator_snp;
}

// Read 1 column of phenotype.
bool ReadFile_column(const string file_pheno, int[] indicator_idv,
                     double[] pheno, const int p_column) {
  writeln("entered ReadFile_column");
  indicator_idv = [];
  pheno = [];

  File infile = File(file_pheno);

  string id;
  double p;
  foreach (line; infile.byLine) {
    auto ch_ptr = line.split("\t");
    if (ch_ptr[p_column -1] == "NA") {  // Note: It may be p_column
      indicator_idv ~= 0;
      pheno ~= -9;
    } else {
      // Pheno is different from pimass2.
      p = to!double(ch_ptr[p_column -1]);
      indicator_idv ~= 1;
      pheno ~= p;
    }
  }

  return true;
}

// Read .fam file (ignored with -p phenotypes switch)
Pheno_result readfile_fam(const string file_fam, size_t[] p_column) {
  writeln("in readfile_fam");

  File infile = File(file_fam ~ ".fam");

  size_t id;
  int c = 0;
  double p;

  double[] pheno_row;
  double[] ind_pheno_row;
  double[] pheno_elements;
  double[] indicator_pheno;
  size_t[size_t] mapID2num;

  p_column = [1]; // modify it later for multiple elements in p_column
  size_t p_max = p_column.reduce!(max);

  size_t[size_t] mapP2c; // size_t, size_t
  for (size_t i = 0; i < p_column.length; i++) {
    mapP2c[p_column[i]] = i;
    pheno_row ~= -9;
    ind_pheno_row ~= 0;
  }

  int rows = 0;

  foreach (line; infile.byLine) {
    auto ch_ptr = line.split(" ");
    id = to!size_t(ch_ptr[0]);

    // need to correct; make jumps in the order of 3, 5, 1
    size_t i = 0;
    while (i < p_max) {
      if((i+1) in mapP2c) {
        enforce(ch_ptr,"Problem reading FAM file (phenotypes out of range)");

        if (ch_ptr[5] == "NA") {
          ind_pheno_row[mapP2c[i + 1]] = 0;
          pheno_row[mapP2c[i + 1]] = -9;
        } else {
          p = to!double(ch_ptr[5]);

          if (p == -9) {
            ind_pheno_row[mapP2c[i + 1]] = 0;
            pheno_row[mapP2c[i + 1]] = -9;
          } else {
            ind_pheno_row[mapP2c[i + 1]] = 1;
            pheno_row[mapP2c[i + 1]] = p;
          }
        }
      }
      i++;
    }

    indicator_pheno ~= ind_pheno_row;
    rows++;
    pheno_elements ~= pheno_row;

    mapID2num[id] = c;
    c++;
  }

  writeln(pheno_elements);

  return Pheno_result(DMatrix([rows, pheno_elements.length/rows ], pheno_elements),
                      DMatrix([rows, indicator_pheno.length/rows ],indicator_pheno));
}


// Read category file, record mapRS2 in the category file does not
// contain a null category so if a snp has 0 for all categories, then
// it is not included in the analysis.
bool ReadFile_cat(const string file_cat, size_t[string]  mapRS2cat, size_t n_vc) {
  writeln("entered ReadFile_cat");
  //mapRS2cat = [];

  File infile = File(file_cat);

  string rs, chr, a1, a0, pos, cm;
  size_t i_cat;

  // Read header.
  HEADER header;
  string header_line = infile.readln();
  ReadHeader_io(header_line, header);

  // Use the header to count the number of categories.
  n_vc = header.coln;
  if (header.rs_col != 0) {
    n_vc--;
  }
  if (header.chr_col != 0) {
    n_vc--;
  }
  if (header.pos_col != 0) {
    n_vc--;
  }
  if (header.cm_col != 0) {
    n_vc--;
  }
  if (header.a1_col != 0) {
    n_vc--;
  }
  if (header.a0_col != 0) {
    n_vc--;
  }

  // Read the following lines to record mapRS2cat.
  foreach(line; infile.byLine) {
    i_cat = 0;
    auto ch_ptrs = line.split("\t");
    foreach (i, ch_ptr; ch_ptrs) {
      enforce(ch_ptr);
      if (header.rs_col != 0 && header.rs_col == i + 1) {
        rs = to!string(ch_ptr);
      } else if (header.chr_col != 0 && header.chr_col == i + 1) {
        chr = to!string(ch_ptr);
      } else if (header.pos_col != 0 && header.pos_col == i + 1) {
        pos = to!string(ch_ptr);
      } else if (header.cm_col != 0 && header.cm_col == i + 1) {
        cm = to!string(ch_ptr);
      } else if (header.a1_col != 0 && header.a1_col == i + 1) {
        a1 = to!string(ch_ptr);
      } else if (header.a0_col != 0 && header.a0_col == i + 1) {
        a0 = to!string(ch_ptr);
      } else if (to!int(ch_ptr) == 1 || to!int(ch_ptr) == 0) {
        if (i_cat == 0) {
          if (header.rs_col == 0) {
            rs = chr ~ ":" ~ pos;
          }
        }

        if (to!int(ch_ptr) == 1 && !(rs in mapRS2cat)) {
          mapRS2cat[rs] = i_cat;
        }
        i_cat++;
      } else {
      }
    }
  }

  return true;
}

void ReadHeader_io(string line, HEADER header){
  return ;
}

// Read category files.
// Read both continuous and discrete category file, record mapRS2catc.
// TODO : need DMatrix with int dtype
void ReadFile_BSLMM_cat(const string file_cat, const string[] vec_rs,
                  DMatrix Ac, DMatrix Ad, DMatrix dlevel,
                  size_t kc, size_t kd) {
  writeln("entered ReadFile_BSLMM_cat");
  File infile = File(file_cat);

  string rs, chr, a1, a0, pos, cm;

  // Read header.
  HEADER header;
  string header_line = infile.readln;
  ReadHeader_io(header_line, header);

  // Use the header to determine the number of categories.
  kc = header.catc_col.length;
  kd = header.catd_col.length;

  // set up storage and mapper
  //map<string, vector<double>> mapRS2catc;
  //map<string, vector<int>> mapRS2catd;
  double[] catc;
  int[] catd;

  // Read the following lines to record mapRS2cat.
  foreach(line; infile.byLine) {
    auto ch_ptr = line.split("\t");

    if (header.rs_col == 0) {
      rs = chr ~ ":" ~ pos;
    }

    catc = [];
    catd = [];

    for (size_t i = 0; i < header.coln; i++) {
      //enforce(ch_ptr);
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
      } else if (header.catc_col.length != 0 && header.catc_col.count(i + 1) != 0) {
        catc ~= to!double(ch_ptr[i]);
      } else if (header.catd_col.length != 0 && header.catd_col.count(i + 1) != 0) {
        catd ~= to!int(ch_ptr[i]);
      } else {
      }

      //ch_ptr = strtok(NULL, " , \t");
    }

    //if ((rs in mapRS2catc) && kc > 0) {
    //  mapRS2catc[rs] = catc;
    //}
    //if ((rs in mapRS2catd) && kd > 0) {
    //  mapRS2catd[rs] = catd;
    //}
  }

  // Load into Ad and Ac.
  if (kc > 0) {
    //Ac = gsl_matrix_alloc(vec_rs.size(), kc);
    for (size_t i = 0; i < vec_rs.length; i++) {
      //if (vec_rs[i] in mapRS2catc) {
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
    //Ad = gsl_matrix_int_alloc(vec_rs.size(), kd);

    for (size_t i = 0; i < vec_rs.length; i++) {
      //if (vec_rs[i] in mapRS2catd) {
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
      //rcd = [];
      for (size_t i = 0; i < Ad.shape[0]; i++) {
        val = to!int(Ad.accessor(i, j));
        rcd[val] = 1;
      }
      dlevel.elements[j] = rcd.length;
    }
  }

  return;
}

bool ReadFile_mcat(const string file_mcat, size_t[string] mapRS2cat, size_t n_vc) {
  writeln("entered ReadFile_mcat");
  //mapRS2cat = [];

  File infile = File(file_mcat);

  size_t[string] mapRS2cat_tmp;
  size_t n_vc_tmp, t = 0;

  foreach (file_name; infile.byLine) {
    //mapRS2cat_tmp = [];
    //ReadFile_cat(file_name, mapRS2cat_tmp, n_vc_tmp);
    //mapRS2cat.insert(mapRS2cat_tmp.begin(), mapRS2cat_tmp.end());
    if (t == 0) {
      n_vc = n_vc_tmp;
    } else {
      n_vc = max(n_vc, n_vc_tmp);
    }
    t++;
  }

  return true;
}

// Read log file.
bool ReadFile_log(const string file_log, double pheno_mean) {
  writeln("entered ReadFile_log");
  File infile = File(file_log);

  size_t flag = 0;

  foreach(line; infile.byLine){
    auto ch_ptr = line.split("\t");

    if (ch_ptr[1] != "NULL" && ch_ptr[1] == "estimated"){
      if (ch_ptr[2] != "NULL" && ch_ptr[2] == "mean"){
        if (ch_ptr[3] != "NULL" && ch_ptr[3] == "=") {
          pheno_mean = to!double(ch_ptr[4]);
          flag = 1;
        }
      }
    }

    if (flag == 1) {
      break;
    }
  }

  return true;
}

double CalcHWE(const int n_hom1, const int n_hom2, const int n_ab) {
  if ((n_hom1 + n_hom2 + n_ab) == 0) {
    return 1;
  }

  // "AA" is the rare allele.
  int n_aa = n_hom1 < n_hom2 ? n_hom1 : n_hom2;
  int n_bb = n_hom1 < n_hom2 ? n_hom2 : n_hom1;

  int rare_copies = 2 * n_aa + n_ab;
  int genotypes = n_ab + n_bb + n_aa;

  double[] het_probs = new double[rare_copies + 1];
  //if (het_probs == )
    //writeln("Internal error: SNP-HWE: Unable to allocate array");

  int i;
  for (i = 0; i <= rare_copies; i++)
    het_probs[i] = 0.0;

  // Start at midpoint.
  // XZ modified to add (long int)
  int mid = (to!int(rare_copies) * (2 * to!int(genotypes) - to!int(rare_copies))) / (2 * to!int(genotypes));

  // Check to ensure that midpoint and rare alleles have same
  // parity.
  if ((rare_copies & 1) ^ (mid & 1))
    mid++;

  int curr_hets = mid;
  int curr_homr = (rare_copies - mid) / 2;
  int curr_homc = genotypes - curr_hets - curr_homr;

  het_probs[mid] = 1.0;
  double sum = het_probs[mid];
  for (curr_hets = mid; curr_hets > 1; curr_hets -= 2) {
    het_probs[curr_hets - 2] = het_probs[curr_hets] * curr_hets *
                               (curr_hets - 1.0) /
                               (4.0 * (curr_homr + 1.0) * (curr_homc + 1.0));
    sum += het_probs[curr_hets - 2];

    // Two fewer heterozygotes for next iteration; add one
    // rare, one common homozygote.
    curr_homr++;
    curr_homc++;
  }

  curr_hets = mid;
  curr_homr = (rare_copies - mid) / 2;
  curr_homc = genotypes - curr_hets - curr_homr;
  for (curr_hets = mid; curr_hets <= rare_copies - 2; curr_hets += 2) {
    het_probs[curr_hets + 2] = het_probs[curr_hets] * 4.0 * curr_homr *
                               curr_homc /
                               ((curr_hets + 2.0) * (curr_hets + 1.0));
    sum += het_probs[curr_hets + 2];

    // Add 2 heterozygotes for next iteration; subtract
    // one rare, one common homozygote.
    curr_homr--;
    curr_homc--;
  }

  for (i = 0; i <= rare_copies; i++)
    het_probs[i] /= sum;

  double p_hwe = 0.0;

  // p-value calculation for p_hwe.
  for (i = 0; i <= rare_copies; i++) {
    if (het_probs[i] > het_probs[n_ab])
      continue;
    p_hwe += het_probs[i];
  }

  p_hwe = p_hwe > 1.0 ? 1.0 : p_hwe;

  return p_hwe;
}

/*
   This code is part of faster_lmm_d and published under the GPLv3
   License (see LICENSE.txt)

   Copyright © 2017-2018 Prasun Anand & Pjotr Prins
*/

module faster_lmm_d.gemma_kinship;

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
import faster_lmm_d.gemma_io;
import faster_lmm_d.gemma_lmm;
import faster_lmm_d.gemma_param;
import faster_lmm_d.helpers;
import faster_lmm_d.optmatrix;

import gsl.permutation;
import gsl.rng;
import gsl.randist;


alias Tuple!(DMatrix, "cvt", int[], "indicator_cvt", int[], "indicator_idv", int, "n_cvt", size_t, "ni_test") Indicators_result ;
alias Tuple!(DMatrix, "pheno", DMatrix, "indicator_pheno") Pheno_result;

DMatrix kinship_from_gemma(const string fn, const string test_name = ""){
  DMatrix X = read_matrix_from_file2(fn);

  DMatrix kinship = matrix_mult(X, X.T);
  DMatrix kinship_norm = divide_dmatrix_num(kinship, 10768);

  writeln(kinship_norm.shape);
  //if(test_name == "mouse_hs_1940"){
    check_kinship_from_gemma(test_name, kinship_norm.elements[0..3], kinship_norm.elements[$-3..$]);
  //}
  return kinship_norm;
}


// similar to batch_run mode 21||22
void generate_kinship(const string geno_fn, const string pheno_fn, const string test_name = "", const bool test_nind= false){
  writeln("in generate kinship");
  Read_files(geno_fn, pheno_fn, test_name);
  validate_kinship();
}

void generate_kinship_plink(const string file_bed, const string test_name = ""){
  writeln("in generate_kinship_plink");
  size_t[] p_column;
  // pheno
  auto pheno = readfile_fam(file_bed, p_column);

  auto indicators = process_cvt_phen(pheno.indicator_pheno);

  size_t ni_test = indicators.ni_test;
  DMatrix W = CopyCvt(indicators.cvt, indicators.indicator_cvt, indicators.indicator_idv, indicators.n_cvt, ni_test);
  string[] setSnps;
  SNPINFO[] snpInfo;
  double maf_level, miss_level, hwe_level, r2_level;
  size_t ns_test;
  //geno
  int[] indicator_snp = readfile_bed(file_bed ~ ".bed", setSnps, W, indicators.indicator_idv, snpInfo,
                  maf_level, miss_level, hwe_level, r2_level, ns_test);

  size_t ni_total = indicators.indicator_idv.length;

  DMatrix matrix_kin = plink_kin(file_bed, indicator_snp, 1, ni_total);
}

void Read_files(const string geno_fn, const string pheno_fn,  const string test_name = "", const string co_variate_fn = ""){
  double[] indicator_pheno;
  size_t[] p_column;

  auto pheno = ReadFile_pheno(pheno_fn, p_column);

  if(test_name == "hs1940_kinship"){
    check_pheno(pheno.pheno);
    check_indicator_pheno(pheno.indicator_pheno);
  }

  auto indicators = process_cvt_phen(pheno.indicator_pheno);

  size_t ni_test = indicators.ni_test;

  DMatrix W = CopyCvt(indicators.cvt, indicators.indicator_cvt, indicators.indicator_idv, indicators.n_cvt, ni_test);

  size_t ni_total = indicators.indicator_idv.length;

  if(test_name == "hs1940_kinship"){
    check_covariates_W(W);
  }

  DMatrix k = ReadFile_geno(geno_fn, ni_total, W, indicators.indicator_idv);

  //bimbam_kin(geno_fn, pheno_fn, W, indicator_snp);
}

Pheno_result ReadFile_pheno(const string file_pheno, size_t[] p_column){

  double[] pheno_elements;
  double[] indicator_pheno;

  File input = File(file_pheno);

  double[] pheno_row;
  double[] ind_pheno_row;

  p_column = [1]; // modify it later for multiple elements in p_column

  size_t p_max = p_column.reduce!(max);

  size_t[size_t] mapP2c;

  for (size_t i = 0; i < p_column.length; i++) {
    mapP2c[p_column[i]] = i;
    pheno_row ~= -9;
    ind_pheno_row ~= 0;
  }

  int rows = 0;

  foreach (line ; input.byLine) {
    auto ch_ptr = to!string(line).split("\t");
    size_t i = 0;
    while (i < p_max) {
      if((i+1) in mapP2c){
        if (ch_ptr[i] == "NA") {
          ind_pheno_row[mapP2c[i + 1]] = 0;
          pheno_row[mapP2c[i + 1]] = -9;
        } else {
          ind_pheno_row[mapP2c[i + 1]] = 1;
          pheno_row[mapP2c[i + 1]] = to!double(ch_ptr[i]);
        }
      }
      i++;
    }

    indicator_pheno ~= ind_pheno_row;
    rows++;
    pheno_elements ~= pheno_row;
  }
  return Pheno_result(DMatrix([rows, pheno_elements.length/rows ], pheno_elements),
                      DMatrix([rows, indicator_pheno.length/rows ],indicator_pheno));
}

bool validate_kinship(){
  return true;
}

DMatrix bimbam_kin(const string geno_fn, const string pheno_fn, const DMatrix W, const int[] indicator_snp,
                   const size_t ni_total = 1940, const bool test_nind= false){

  writeln("in bimbam_kin");
  string filename = geno_fn;
  auto pipe = pipeShell("gunzip -c " ~ filename);
  File input = pipe.stdout;

  int k_mode = 0;

  size_t n_miss;
  double d, geno_mean, geno_var;

  // setKSnp and/or LOCO support
  //bool process_ksnps = ksnps.size();

  DMatrix matrix_kin = zeros_dmatrix(ni_total, ni_total);

  double[] geno = new double[ni_total];
  double[] geno_miss = new double[ni_total];

  // Xlarge contains inds x markers
  size_t K_BATCH_SIZE = 20000;
  const size_t msize = K_BATCH_SIZE;
  DMatrix Xlarge = zeros_dmatrix(ni_total, msize);

  // For every SNP read the genotype per individual
  size_t ns_test = 0;
  size_t t = 0;

  foreach (line ; input.byLine) {
    if (indicator_snp[t] == 0){
      t++;
      continue;
    }

    auto chr = to!string(line).split(",")[3..$];

    if (test_nind) {
      if (chr.length != ni_total+3) {
        writeln("Columns in geno file do not match # individuals");
      }
    }

    // calc SNP stats
    geno_mean = 0.0;
    n_miss = 0;
    geno_var = 0.0;

    foreach(ref ele; geno_miss){ele = 0;}

    foreach(i; 0..ni_total) {
      auto digit = to!string(chr[i].strip());
      if (digit == "NA") {
        geno_miss[i] = 0;
        n_miss++;
      } else {
        d = to!double(digit);
        geno[i] = d;
        geno_miss[i] = 1;
        geno_mean += d;
        geno_var += d * d;
      }
    }

    geno_mean /= to!double(ni_total - n_miss);
    geno_var += (geno_mean * geno_mean * to!double(n_miss))/to!double(ni_total);
    geno_var -= geno_mean * geno_mean;

    foreach (i; 0..ni_total) {
      if (geno_miss[i] == 0) {geno[i] = geno_mean;}
    }

    foreach(ref ele; geno){ ele -= geno_mean;}

    if (k_mode == 2 && geno_var != 0) {
      foreach(ref ele; geno){
        ele /= sqrt(geno_var);
      }
    }

    // set the SNP column ns_test
    set_col2(Xlarge, ns_test % msize, DMatrix([geno.length, 1], geno));

    ns_test++;

    // compute kinship matrix and return in matrix_kin a SNP at a time
    if (ns_test % msize == 0) {
      matrix_kin = matrix_mult(Xlarge, Xlarge.T);
      Xlarge = zeros_dmatrix(ni_total, msize);
    }

    t++;
  }
  if (ns_test % msize != 0) {
    matrix_kin = matrix_mult(Xlarge, Xlarge.T);
  }

  matrix_kin = divide_dmatrix_num(matrix_kin, ns_test);
  matrix_kin = matrix_kin.T;


  check_kinship_from_gemma("test_name", matrix_kin.elements[0..3], matrix_kin.elements[$-3..$]);

  return matrix_kin;
}


// Read bimbam mean genotype file, the first time, to obtain #SNPs for
// analysis (ns_test) and total #SNP (ns_total).
DMatrix ReadFile_geno(const string geno_fn, const ulong ni_total, const DMatrix W,
                    const int[] indicator_idv, const bool test_nind= false){

  writeln("ReadFile_geno", geno_fn);

  int k_mode = 0;
  double d, geno_mean, geno_var;

  DMatrix[] kinship_collect;

  // setKSnp and/or LOCO support
  //bool process_ksnps = ksnps.size();

  DMatrix matrix_kin = zeros_dmatrix(ni_total, ni_total);

  double[] geno_v = new double[ni_total];
  double[] geno_miss = new double[ni_total];

  // Xlarge contains inds x markers
  size_t K_BATCH_SIZE = 1000;
  const size_t msize = K_BATCH_SIZE;
  DMatrix Xlarge = zeros_dmatrix(ni_total, msize);

  version(PARALLEL){
    auto task_pool = new TaskPool(totalCPUs);
  }

  // For every SNP read the genotype per individual
  size_t t = 0;

  string[string] mapRS2chr;
  long[string] mapRS2bp, mapRS2cM;

  string[] setSnps;

  size_t ns_test = 0;
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

    if (to!double(n_miss) / to!double(ni_test) > miss_level) {continue;}
    if ((maf < maf_level || maf > (1.0 - maf_level)) && maf_level != -1){continue;}
    if (flag_poly != 1) {continue;}
    if (hwe_level != 0 && maf_level != -1) {
      if (CalcHWE(n_0, n_2, n_1) < hwe_level) {continue;}
    }

    // Filter SNP if it is correlated with W unless W has
    // only one column, of 1s.
    foreach (i, miss; genotype_miss) {
      if (miss == 1) {
        geno = maf * 2.0;
        genotype[i] = geno;
      }
    }

    double Wtx = vector_ddot(W.elements, genotype);

    v_x = vector_ddot(genotype, genotype);
    v_w = Wtx * Wtx * WtWi;

    if (W.shape[1] != 1 && v_w / v_x >= r2_level) { continue;}

    ns_test++;


    if (test_nind) {
      if (chr_val.length != ni_total+3) {
        writeln("Columns in geno file do not match # individuals");
      }
    }

    // calc SNP stats
    geno_mean = 0.0;
    n_miss = 0;
    geno_var = 0.0;

    foreach(ref ele; geno_miss){ele = 0;}

    foreach(i; 0..ni_total) {
      auto digit = to!string(chr_val[i].strip());
      if (digit == "NA") {
        geno_miss[i] = 0;
        n_miss++;
      } else {
        d = to!double(digit);
        geno_v[i] = d;
        geno_miss[i] = 1;
        geno_mean += d;
        geno_var += d * d;
      }
    }

    geno_mean /= to!double(ni_total - n_miss);
    geno_var += (geno_mean * geno_mean * to!double(n_miss))/to!double(ni_total);
    geno_var -= geno_mean * geno_mean;

    foreach (i; 0..ni_total) {
      if (geno_miss[i] == 0) {geno_v[i] = geno_mean;}
    }

    foreach(ref ele; geno_v){ ele -= geno_mean;}

    if (k_mode == 2 && geno_var != 0) {
      foreach(ref ele; geno_v){
        ele /= sqrt(geno_var);
      }
    }

    // set the SNP column ns_test
    set_col2(Xlarge, ns_test % msize, DMatrix([geno_v.length, 1], geno_v));

    version(PARALLEL){
      void compute_mults(const DMatrix Xlarge){
        kinship_collect ~= cpu_mat_mult(Xlarge, 0, Xlarge, 1);
        writeln("batch processed");
      }

      // compute kinship matrix and return in matrix_kin a SNP at a time
      if (ns_test % msize == 0) {
        auto taskk = task(&compute_mults, Xlarge);
        task_pool.put(taskk);
        Xlarge = zeros_dmatrix(ni_total, msize);
      }
    }else{
      if (ns_test % msize == 0) {
        writeln("batch processed");
        matrix_kin = matrix_kin + cpu_mat_mult(Xlarge, 0, Xlarge, 1);
        Xlarge = zeros_dmatrix(ni_total, msize);
      }
    }


    t++;
  }

  version(PARALLEL){
    task_pool.finish(true);

    if(kinship_collect.length > 0 ){
      matrix_kin = taskPool.reduce!"a + b"(kinship_collect);
    }
  }

  if (ns_test % msize != 0) {
    matrix_kin = add_dmatrix(matrix_kin, cpu_mat_mult(Xlarge, 0, Xlarge, 1));
  }

  matrix_kin = divide_dmatrix_num(matrix_kin, ns_test);
  matrix_kin = matrix_kin.T;

  check_kinship_from_gemma("test_name", matrix_kin.elements[0..3], matrix_kin.elements[$-3..$]);

  return matrix_kin;
}

DMatrix CopyCvt(const DMatrix cvt, const int[] indicator_cvt, const int[] indicator_idv, const size_t n_cvt, const size_t ni_test) {
  DMatrix W = zeros_dmatrix(ni_test, n_cvt); // ni_test missing
  size_t ci_test = 0;

  foreach(i, idv; indicator_idv) {
    if (idv == 0 || indicator_cvt[i] == 0) {
      continue;
    }
    for (size_t j = 0; j < n_cvt; ++j) {
      W.elements[ci_test * W.cols + j] = cvt.accessor(i, j);
    }
    ci_test++;
  }

  return W;
}

// Post-process phenotypes and covariates.
Indicators_result process_cvt_phen(const DMatrix indicator_pheno, const string test_name = ""){

  writeln(indicator_pheno.shape);

  // Convert indicator_pheno to indicator_idv.
  int k = 1;
  int[] indicator_idv, indicator_cvt, indicator_weight;
  int[] indicator_gxe;
  size_t ni_test, ni_subsample;
  double[] cvt;
  int a_mode;

  foreach(i ; 0..indicator_pheno.elements.length) {
    k = 1;
    foreach(j; 0..indicator_pheno.cols) {
      if (indicator_pheno.accessor(i,j) == 0) {
        k = 0;
      }
    }
    indicator_idv ~= k;
  }

  ni_test = 0;
  foreach(i; 0..indicator_idv.length){
    // Remove individuals with missing covariates.
    if (indicator_cvt.length != 0) {
      indicator_idv[i] *= indicator_cvt[i];
    }
    // Remove individuals with missing gxe variables.
    if (indicator_gxe.length != 0) {
      indicator_idv[i] *= indicator_gxe[i];
    }
    // Remove individuals with missing residual weights.
    if (indicator_weight.length != 0) {
      indicator_idv[i] *= indicator_weight[i];
    }
    // Obtain ni_test.
    if (indicator_idv[i] == 0){ continue; }
    ni_test++;
  }

  // If subsample number is set, perform a random sub-sampling
  // to determine the subsampled ids.
  if (ni_subsample != 0) {
    if (ni_test < ni_subsample) {
      writeln("error! number of subsamples is less than number of analyzed individuals. ");
    }
    else {

      // Set up random environment.
      size_t randseed = -1;
      gsl_rng_env_setup();
      gsl_rng *gsl_r;
      const gsl_rng_type *gslType = gsl_rng_default;
      if (randseed < 0) {
        time_t rawtime;
        time(&rawtime);
        tm *ptm = gmtime(&rawtime);

        randseed = (ptm.tm_hour % 24 * 3600 + ptm.tm_min * 60 + ptm.tm_sec);
      }
      gsl_r = gsl_rng_alloc(gslType);
      gsl_rng_set(gsl_r, randseed);

      // From ni_test, sub-sample ni_subsample.
      size_t[] a, b;
      foreach(i; 0..ni_subsample) { a ~= 0; }
      foreach(i; 0..ni_test) { b ~= i; }

      gsl_ran_choose(gsl_r, cast(void *)(&a[0]), ni_subsample, cast(void *)(&b[0]), ni_test, size_t.sizeof);

      // Re-set indicator_idv and ni_test.
      size_t j = 0;
      foreach (ref indicator; indicator_idv) {
        if (indicator == 0) { continue; }
        if (a.canFind(j)){ indicator = 0; }
        j++;
      }
      ni_test = ni_subsample;
    }
  }

  // Check ni_test.
  if (ni_test == 0 && a_mode != 15) {
    error = true;
    writeln("error! number of analyzed individuals equals 0. ");
    exit(0);
  }

  // Check covariates to see if they are correlated with each
  // other, and to see if the intercept term is included.
  // After getting ni_test.
  // Add or remove covariates.


  if (indicator_cvt.length != 0) {
    //CheckCvt();
  } else {

    double[] cvt_row;
    cvt_row ~= 1;

    foreach (i; indicator_idv) {
      indicator_cvt ~= 1;
      cvt ~= cvt_row;
    }
  }

  DMatrix s_cvt = DMatrix([indicator_idv.length, cvt.length/indicator_idv.length] , cvt);

  writeln("done process_cvt_phen");
  writeln(ni_test);

  if(test_name == "hs1940_kinship"){ 
    check_indicator_cvt(cvt);
    check_cvt_matrix(s_cvt);
    check_indicator_idv(indicator_idv);
  }

  return Indicators_result(s_cvt, indicator_cvt, indicator_idv, 1, ni_test);
}


DMatrix plink_kin(const string file_bed, int[] indicator_snp, const int k_mode, size_t ni_total) {
  writeln("entered PlinkKin");

  File infile = File(file_bed ~ ".bed");

  size_t n_miss, ci_total;
  double d, geno_mean, geno_var;

  DMatrix matrix_kin = zeros_dmatrix(ni_total, ni_total);
  DMatrix geno = zeros_dmatrix(1, ni_total);

  size_t ns_test = 0;
  size_t n_bit;

  // Create a large matrix.
  size_t K_BATCH_SIZE = 20000;

  const size_t msize = K_BATCH_SIZE;
  DMatrix Xlarge = zeros_dmatrix(ni_total, msize);

  // Calculate n_bit and c, the number of bit for each snp.
  if (ni_total % 4 == 0) {
    n_bit = ni_total / 4;
  } else {
    n_bit = ni_total / 4 + 1;
  }

  // print the first three magic numbers
  for (int i = 0; i < 3; ++i) {
    auto b = infile.rawRead(new char[8]);
  }

  for (size_t t = 0; t < indicator_snp.length; ++t) {
    if (indicator_snp[t] == 0) {
      continue;
    }

    // n_bit, and 3 is the number of magic numbers.
    infile.seek(t * n_bit + 3);

    // Read genotypes.
    geno_mean = 0.0;
    n_miss = 0;
    ci_total = 0;
    geno_var = 0.0;
    for (int i = 0; i < n_bit; ++i) {
      auto b = infile.rawRead(new char[8]);

      // Minor allele homozygous: 2.0; major: 0.0.
      for (size_t j = 0; j < 4; ++j) {
        if ((i == (n_bit - 1)) && ci_total == ni_total) {
          break;
        }

        if (b[2 * j] == 0) {
          if (b[2 * j + 1] == 0) {
            geno.elements[ci_total] = 2.0;
            geno_mean += 2.0;
            geno_var += 4.0;
          } else {
            geno.elements[ci_total] = 1.0;
            geno_mean += 1.0;
            geno_var += 1.0;
          }
        } else {
          if (b[2 * j + 1] == 1) {
            geno.elements[ci_total] = 0.0;
          } else {
            geno.elements[ci_total] = -9.0;
            n_miss++;
          }
        }

        ci_total++;
      }
    }

    geno_mean /= to!double(ni_total - n_miss);
    geno_var += geno_mean * geno_mean * to!double(n_miss);
    geno_var /= to!double(ni_total);
    geno_var -= geno_mean * geno_mean;

    for (size_t i = 0; i < ni_total; ++i) {
      d = geno.elements[i];
      if (d == -9.0) {
        geno.elements[i] = geno_mean;
      }
    }

    geno = add_dmatrix_num(geno, -1.0 * geno_mean);

    if (k_mode == 2 && geno_var != 0) {
      geno = multiply_dmatrix_num(geno, 1.0 / sqrt(geno_var));
    }
    DMatrix Xlarge_col = get_col(Xlarge, ns_test % msize);
    //gsl_vector_memcpy(&Xlarge_col.vector, geno);

    ns_test++;

    if (ns_test % msize == 0) {
      matrix_kin = matrix_mult(Xlarge, Xlarge.T);
      Xlarge = zeros_dmatrix(ni_total, msize);
    }
  }

  if (ns_test % msize != 0) {
    matrix_kin = matrix_mult(Xlarge, Xlarge.T);
  }


  matrix_kin = divide_dmatrix_num(matrix_kin, ns_test);
  matrix_kin = matrix_kin.T;

  return matrix_kin;
}

bool plink_kin(const string file_bed,
              const int[] indicator_idv,
              const int[] indicator_snp,
              const double[string] mapRS2weight,
              const size_t[string] mapRS2cat,
              const SNPINFO[] snpInfo, const DMatrix W,
              DMatrix matrix_kin, DMatrix vector_ns) {
  writeln("entered PlinkKin");

  File infile = File(file_bed);

  //char ch[1];
  int[] b;

  size_t n_miss, ci_total, ci_test;
  double d, geno_mean, geno_var;

  size_t ni_test = matrix_kin.shape[0];
  size_t ni_total = indicator_idv.length;
  DMatrix geno; // = gsl_vector_safe_alloc(ni_test);

  DMatrix WtW = matrix_mult(W.T, W);

  DMatrix WtWi = WtW.inverse();

  size_t ns_test = 0;
  size_t n_bit;

  size_t n_vc = matrix_kin.shape[1] / ni_test, i_vc;
  string rs;
  size_t[] ns_vec;
  for (size_t i = 0; i < n_vc; i++) {
    ns_vec ~= 0;
  }

  // Create a large matrix.
  size_t K_BATCH_SIZE = 20_000;

  const size_t msize = K_BATCH_SIZE;
  DMatrix Xlarge = zeros_dmatrix(ni_test, msize * n_vc);

  // Calculate n_bit and c, the number of bit for each SNP.
  if (ni_total % 4 == 0) {
    n_bit = ni_total / 4;
  } else {
    n_bit = ni_total / 4 + 1;
  }

  // Print the first three magic numbers.
  for (int i = 0; i < 3; ++i) {
    //infile.read(ch, 1);
    //b = ch[0];
  }

  for (size_t t = 0; t < indicator_snp.length; ++t) {
    if (indicator_snp[t] == 0) {
      continue;
    }

    // n_bit, and 3 is the number of magic numbers
    infile.seek(t * n_bit + 3);

    rs = snpInfo[t].rs_number; // This line is new.

    // Read genotypes.
    geno_mean = 0.0;
    n_miss = 0;
    ci_total = 0;
    geno_var = 0.0;
    ci_test = 0;
    for (int i = 0; i < n_bit; ++i) {
      //infile.read(ch, 1);
      //b = ch[0];

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
            geno.elements[ci_test] = 2.0;
            geno_mean += 2.0;
            geno_var += 4.0;
          } else {
            geno.elements[ci_test] = 1.0;
            geno_mean += 1.0;
            geno_var += 1.0;
          }
        } else {
          if (b[2 * j + 1] == 1) {
            geno.elements[ci_test] = 0.0;
          } else {
            geno.elements[ci_test] = -9.0;
            n_miss++;
          }
        }

        ci_test++;
        ci_total++;
      }
    }

    geno_mean /= to!double(ni_test - n_miss);
    geno_var += geno_mean * geno_mean * to!double(n_miss);
    geno_var /= to!double(ni_test);
    geno_var -= geno_mean * geno_mean;

    for (size_t i = 0; i < ni_test; ++i) {
      d = geno.elements[i];
      if (d == -9.0) {
        geno.elements[i] = geno_mean;
      }
    }

    geno = add_dmatrix_num(geno, -1.0 * geno_mean);

    DMatrix Wtx = matrix_mult(W.T, geno);
    DMatrix WtWiWtx = matrix_mult(WtWi, Wtx);
    // TODO
    //gsl_blas_dgemv(CblasNoTrans, -1.0, W, WtWiWtx, 1.0, geno);
    geno_var = vector_ddot(geno, geno);
    geno_var /= to!double(ni_test);

    if (geno_var != 0 &&
        (mapRS2weight.length == 0 || (rs in mapRS2weight) )) {
      if (mapRS2weight.length == 0) {
        d = 1.0 / geno_var;
      } else {
        d = mapRS2weight[rs] / geno_var;
      }

      geno = multiply_dmatrix_num(geno, sqrt(d));
      if (n_vc == 1 || mapRS2cat.length == 0) {
        DMatrix Xlarge_col = get_col(Xlarge, ns_vec[0] % msize);
        //gsl_vector_memcpy(&Xlarge_col.vector, geno);
        ns_vec[0]++;

        if (ns_vec[0] % msize == 0) {
          matrix_kin = matrix_mult(Xlarge, Xlarge.T);
          Xlarge = zeros_dmatrix(ni_test, msize * n_vc);
        }
      } else if (rs in mapRS2weight) {
        i_vc = mapRS2cat[rs];

        //gsl_vector_view 
        DMatrix Xlarge_col = get_col(Xlarge, msize * i_vc + ns_vec[i_vc] % msize);
        //gsl_vector_memcpy(&Xlarge_col.vector, geno);

        ns_vec[i_vc]++;

        if (ns_vec[i_vc] % msize == 0) {
          //gsl_matrix_view 
          DMatrix X_sub = get_sub_dmatrix(Xlarge, 0, msize * i_vc, ni_test, msize);
          //gsl_matrix_view
          DMatrix kin_sub = get_sub_dmatrix( matrix_kin, 0, ni_test * i_vc, ni_test, ni_test);
          kin_sub = matrix_mult(X_sub, X_sub.T);

          X_sub = zeros_dmatrix(X_sub.shape[0], X_sub.shape[1]);
        }
      }
    }
    ns_test++;
  }

  for (size_t k = 0; k < n_vc; k++) {
    if (ns_vec[k] % msize != 0) {
      //gsl_matrix_view
      DMatrix X_sub = get_sub_dmatrix(Xlarge, 0, msize * k, ni_test, msize);
      //gsl_matrix_view
      DMatrix kin_sub = get_sub_dmatrix(matrix_kin, 0, ni_test * k, ni_test, ni_test);
      kin_sub = matrix_mult(X_sub, X_sub.T);
    }
  }

  for (size_t t = 0; t < n_vc; t++) {
    vector_ns.elements[t] = ns_vec[t];

    for (size_t i = 0; i < ni_test; ++i) {
      for (size_t j = 0; j <= i; ++j) {
        d = matrix_kin.accessor(j, i + ni_test * t);
        d /= to!double(ns_vec[t]);
        matrix_kin.set(i, j + ni_test * t, d);
        matrix_kin.set(j, i + ni_test * t, d);
      }
    }
  }

  return true;
}


void check_kinship_from_gemma(const string test_name, const double[] top, const double[] bottom){

  writeln(top);
  enforce(modDiff(top[0], 0.335059 ) < 0.001);
  enforce(modDiff(top[1], -0.0227226 ) < 0.001);
  enforce(modDiff(top[2], 0.0103535 ) < 0.001);

  writeln(bottom);
  enforce(modDiff(bottom[0], 0.0039059 ) < 0.001);
  enforce(modDiff(bottom[1], -0.0210802 ) < 0.001);
  enforce(modDiff(bottom[2], 0.3881094925 ) < 0.001);

  writeln("kinship tests pass successfully");
}

void check_indicator_snp(int[] indicator_idv){
  enforce(modDiff(to!double(indicator_idv[0]), 0 ) < 0.001);
  enforce(modDiff(to!double(indicator_idv[1]), 0 ) < 0.001);
  enforce(modDiff(to!double(indicator_idv[2]), 0 ) < 0.001);

  enforce(modDiff(to!double(indicator_idv[$-3]), 0) < 0.001);
  enforce(modDiff(to!double(indicator_idv[$-2]), 0) < 0.001);
  enforce(modDiff(to!double(indicator_idv[$-1]), 0) < 0.001);

  writeln("indicator snp tests pass");
}

void check_indicator_idv(int[] indicator_idv){
  enforce(modDiff(to!double(indicator_idv[0]), 1 ) < 0.001);
  enforce(modDiff(to!double(indicator_idv[1]), 1 ) < 0.001);
  enforce(modDiff(to!double(indicator_idv[2]), 1 ) < 0.001);

  enforce(modDiff(to!double(indicator_idv[$-3]), 1) < 0.001);
  enforce(modDiff(to!double(indicator_idv[$-2]), 0) < 0.001);
  enforce(modDiff(to!double(indicator_idv[$-1]), 0) < 0.001);

  writeln("indicator idv tests pass");
}

void check_indicator_cvt(double[] indicator_snp){
  enforce(modDiff(to!double(indicator_snp[0]), 1 ) < 0.001);
  enforce(modDiff(to!double(indicator_snp[1]), 1 ) < 0.001);
  enforce(modDiff(to!double(indicator_snp[2]), 1 ) < 0.001);

  enforce(modDiff(to!double(indicator_snp[$-3]), 1) < 0.001);
  enforce(modDiff(to!double(indicator_snp[$-2]), 1) < 0.001);
  enforce(modDiff(to!double(indicator_snp[$-1]), 1) < 0.001);

  writeln("indicator cvt tests pass");
}

void check_cvt_matrix(DMatrix cvt){
  enforce(cvt.rows == 1940);
  enforce(cvt.cols ==    1);

  enforce(modDiff(to!double(cvt.elements[0]), 1 ) < 0.001);
  enforce(modDiff(to!double(cvt.elements[1]), 1 ) < 0.001);
  enforce(modDiff(to!double(cvt.elements[2]), 1 ) < 0.001);

  enforce(modDiff(to!double(cvt.elements[$-3]), 1) < 0.001);
  enforce(modDiff(to!double(cvt.elements[$-2]), 1) < 0.001);
  enforce(modDiff(to!double(cvt.elements[$-1]), 1) < 0.001);

  writeln("cvt_matrix tests pass");
}

void check_pheno(DMatrix pheno){
  enforce(pheno.rows == 1940);
  enforce(pheno.cols ==    1);

  enforce(modDiff(to!double(pheno.elements[0]),  0.224992 ) < 0.001);
  enforce(modDiff(to!double(pheno.elements[1]), -0.974543 ) < 0.001);
  enforce(modDiff(to!double(pheno.elements[2]),  0.19591 ) < 0.001);

  enforce(modDiff(to!double(pheno.elements[$-3]),  0.69698) < 0.001);
  enforce(modDiff(to!double(pheno.elements[$-2]), -9) < 0.001);
  enforce(modDiff(to!double(pheno.elements[$-1]), -9) < 0.001);

  writeln("pheno tests pass");
}

void check_indicator_pheno(DMatrix indicator_pheno){
  enforce(indicator_pheno.rows == 1940);
  enforce(indicator_pheno.cols ==    1);

  enforce(indicator_pheno.elements[0] == 1);
  enforce(indicator_pheno.elements[1] == 1);
  enforce(indicator_pheno.elements[2] == 1);

  enforce(indicator_pheno.elements[$-3] == 1);
  enforce(indicator_pheno.elements[$-2] == 0);
  enforce(indicator_pheno.elements[$-1] == 0);

  writeln("indicator pheno pheno tests pass");
}

void check_covariates_W(DMatrix covariate_matrix){
  enforce(covariate_matrix.rows == 1410);
  enforce(covariate_matrix.cols ==    1);

  enforce(modDiff(to!double(covariate_matrix.elements[0]), 1 ) < 0.001);
  enforce(modDiff(to!double(covariate_matrix.elements[1]), 1 ) < 0.001);
  enforce(modDiff(to!double(covariate_matrix.elements[2]), 1 ) < 0.001);

  enforce(modDiff(to!double(covariate_matrix.elements[$-3]), 1) < 0.001);
  enforce(modDiff(to!double(covariate_matrix.elements[$-2]), 1) < 0.001);
  enforce(modDiff(to!double(covariate_matrix.elements[$-1]), 1) < 0.001);

  writeln("covariates tests pass");
}

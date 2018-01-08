/*
   This code is part of faster_lmm_d and published under the GPLv3
   License (see LICENSE.txt)

   Copyright © 2017-2018 Prasun Anand & Pjotr Prins
*/

module faster_lmm_d.gemma_kinship;

import core.stdc.stdlib : exit;

import std.conv;
import std.exception;
import std.file;
import std.math;
import std.parallelism;
import std.algorithm: min, max, reduce, countUntil;
alias mlog = std.math.log;
import std.process;
import std.range;
import std.stdio;
import std.typecons;
import std.experimental.logger;
import std.string;

import faster_lmm_d.dmatrix;
import faster_lmm_d.gemma_param;
import faster_lmm_d.helpers;
import faster_lmm_d.optmatrix;

import gsl.permutation;

void generate_kinship(string geno_fn, string pheno_fn, size_t ni_total = 1940, bool test_nind= false){

  string filename = geno_fn;
  auto pipe = pipeShell("gunzip -c " ~ filename);
  File input = pipe.stdout;

  int k_mode = 0;

  size_t n_miss;
  double d, geno_mean, geno_var;

  // setKSnp and/or LOCO support
  //bool process_ksnps = ksnps.size();

  DMatrix matrix_kin = zeros_dmatrix(ni_total, ni_total);

  int[] indicator_snp = ReadFile_geno(geno_fn, ni_total);
  foreach(ref ele; indicator_snp){ele = 1;}


  double[] geno = new double[ni_total];
  double[] geno_miss = new double[ni_total];

  // Xlarge contains inds x markers
  size_t K_BATCH_SIZE = 1000;
  const size_t msize = K_BATCH_SIZE;
  DMatrix Xlarge = zeros_dmatrix(ni_total, msize);

  // For every SNP read the genotype per individual
  size_t ns_test = 0;
  size_t t = 0;
  foreach (line ; input.byLine) {

    if (indicator_snp[t] == 0)
      continue;

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
    //gsl_vector_set_all(geno_miss, 0);
    for (size_t i = 0; i < ni_total; ++i) {
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
    geno_var += geno_mean * geno_mean * to!double(n_miss);
    geno_var /= to!double(ni_total);
    geno_var -= geno_mean * geno_mean;

    for (size_t i = 0; i < ni_total; ++i) {
      if (geno_miss[i] == 0) {
        geno[i] = geno_mean;
      }
    }

    foreach(ref ele; geno){
      ele -= geno_mean;
    }

    if (k_mode == 2 && geno_var != 0) {
      foreach(ref ele; geno){
        ele /= sqrt(geno_var);
      }
    }

    // set the SNP column ns_test
    DMatrix Xlarge_col = set_col(Xlarge, ns_test % msize, DMatrix([geno.length, 1], geno));

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
}

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

// Read bimbam mean genotype file, the first time, to obtain #SNPs for
// analysis (ns_test) and total #SNP (ns_total).
int[] ReadFile_geno(string geno_fn, ulong ni_total){
  //const string &file_geno, const set<string> &setSnps,
  //                 const gsl_matrix *W, vector<int> &indicator_idv,
  //                 vector<int> &indicator_snp, const double &maf_level,
  //                 const double &miss_level, const double &hwe_level,
  //                 const double &r2_level, map<string, string> &mapRS2chr,
  //                 map<string, long int> &mapRS2bp,
  //                 map<string, double> &mapRS2cM, vector<SNPINFO> &snpInfo,
  //                 size_t &ns_test) {
  writeln("ReadFile_geno");
  int[] indicator_snp;
  int[] indicator_idv;

  string[string] mapRS2chr;
  long[string] mapRS2bp, mapRS2cM;

  string[] setSnps;

  //snpInfo.clear();


  DMatrix W = ones_dmatrix(ni_total, ni_total);
  size_t ns_test;
  SNPINFO[] snpInfo;
  const double maf_level;
  const double miss_level;
  const double hwe_level;
  const double r2_level;

  string filename = geno_fn;
  auto pipe = pipeShell("gunzip -c " ~ filename);
  File input = pipe.stdout;

  double[] genotype = new double[ni_total];
  double[] genotype_miss = new double[ni_total];

  DMatrix WtW = matrix_mult(W.T, W);

  DMatrix WtWi = inverse(WtW);

  double v_x, v_w;
  int c_idv = 0;

  string rs;
  string chr;
  long b_pos;
  string major;
  string minor;
  double cM;
  size_t file_pos;

  double maf, geno, geno_old;
  size_t n_miss;
  int n_0, n_1, n_2;
  int flag_poly;

  int ni_test = 0;
  for (int i = 0; i < ni_total; ++i) {
    ni_test += indicator_idv[i];
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
                       0,    -9, -9, 0,  0,     file_pos);
      snpInfo ~= sInfo;
      indicator_snp ~= 0;

      file_pos++;
      continue;
    }
    //auto a = countUntil(mapRS2bp, rs);
    //if (a != -1) {
    //  if (count_warnings++ < 10) {
    //    writeln("Can't figure out position for ");
    //  }
    //  chr = "-9";
    //  b_pos = -9;
    //  cM = -9;
    //} else {
    //  b_pos = mapRS2bp[rs];
    //  chr = mapRS2chr[rs];
    //  cM = mapRS2cM[rs];
    //}

    maf = 0;
    n_miss = 0;
    flag_poly = 0;
    geno_old = -9;
    n_0 = 0;
    n_1 = 0;
    n_2 = 0;
    c_idv = 0;
    foreach(ref ele; genotype_miss){ele = 0;}
    for (int i = 0; i < ni_total; ++i) {
      if (indicator_idv[i] == 0)
        continue;
      auto digit = to!string(chr_val[i].strip());
      if (digit == "NA") {
        genotype_miss[c_idv] = 1;
        n_miss++;
        c_idv++;
        continue;
      }

      geno = to!double(digit);
      if (geno >= 0 && geno <= 0.5) {
        n_0++;
      }
      if (geno > 0.5 && geno < 1.5) {
        n_1++;
      }
      if (geno >= 1.5 && geno <= 2.0) {
        n_2++;
      }

      genotype[c_idv] = geno;

      if (flag_poly == 0) {
        geno_old = geno;
        flag_poly = 2;
      }
      if (flag_poly == 2 && geno != geno_old) {
        flag_poly = 1;
      }

      maf += geno;

      c_idv++;
    }
    maf /= 2.0 * to!double(ni_test - n_miss);

//(string chr, string rs_number, long cM, long base_position, string a_minor, string a_major, ulong n_miss, double missingness, double maf, ulong n_idv, ulong n_nb, ulong file_position) is not callable using argument types
//(string,     string,           double,  long,               string,         string,         ulong,        double,             double,     ulong,       int,        ulong)

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

    DMatrix Wtx = matrix_mult(W, DMatrix([genotype.length, 1], genotype));
    DMatrix WtWiWtx = matrix_mult(WtWi, Wtx);
    v_x = vector_ddot(genotype, genotype);
    v_w = vector_ddot(Wtx, WtWiWtx);

    if (W.shape[1] != 1 && v_w / v_x >= r2_level) {
      indicator_snp ~= 0;
      continue;
    }

    indicator_snp ~= 1;
    ns_test++;
  }

  return indicator_snp;
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


DMatrix CopyCvt(DMatrix cvt, int[] indicator_cvt, int[] indicator_idv, size_t n_cvt) {
  DMatrix W;
  size_t ci_test = 0;

  foreach(i, idv; indicator_idv) {
    if (idv == 0 || indicator_cvt[i] == 0) {
      continue;
    }
    for (size_t j = 0; j < n_cvt; ++j) {
      W.elements[ci_test * W.rows + j] = cvt.accessor(i, j);
    }
    ci_test++;
  }

  return W;
}


// Post-process phentoypes and covariates.
void process_cvt_phen() {

  // Convert indicator_pheno to indicator_idv.
  int k = 1;
  int[] indicator_idv, indicator_cvt, indicator_weight;
  int[][] indicator_pheno;
  int[] indicator_gxe;
  size_t ni_test, ni_subsample;
  double[] cvt;
  int a_mode;

  for (size_t i = 0; i < indicator_pheno.length; i++) {
    k = 1;
    for (size_t j = 0; j < indicator_pheno[i].length; j++) {
      if (indicator_pheno[i][j] == 0) {
        k = 0;
      }
    }
    indicator_idv ~= k;
  }

  // Remove individuals with missing covariates.
  if ((indicator_cvt).length != 0) {
    for (size_t i = 0; i < (indicator_idv).length; ++i) {
      indicator_idv[i] *= indicator_cvt[i];
    }
  }

  // Remove individuals with missing gxe variables.
  if ((indicator_gxe).length != 0) {
    for (size_t i = 0; i < (indicator_idv).length; ++i) {
      indicator_idv[i] *= indicator_gxe[i];
    }
  }

  // Remove individuals with missing residual weights.
  if ((indicator_weight).length != 0) {
    for (size_t  i = 0; i < (indicator_idv).length; ++i) {
      indicator_idv[i] *= indicator_weight[i];
    }
  }

  // Obtain ni_test.
  ni_test = 0;
  for (size_t  i = 0; i < (indicator_idv).length; ++i) {
    if (indicator_idv[i] == 0) {
      continue;
    }
    ni_test++;
  }

  // If subsample number is set, perform a random sub-sampling
  // to determine the subsampled ids.
  if (ni_subsample != 0) {
    if (ni_test < ni_subsample) {
      writeln("error! number of subsamples is less than number of analyzed individuals. ");
    } else {

      // Set up random environment.
      //gsl_rng_env_setup();
      //gsl_rng *gsl_r;
      //const gsl_rng_type *gslType;
      //gslType = gsl_rng_default;
      //if (randseed < 0) {
      //  time_t rawtime;
      //  time(&rawtime);
      //  tm *ptm = gmtime(&rawtime);

      //  randseed = (unsigned)(ptm->tm_hour % 24 * 3600 + ptm->tm_min * 60 +
      //                        ptm->tm_sec);
      //}
      //gsl_r = gsl_rng_alloc(gslType);
      //gsl_rng_set(gsl_r, randseed);

      // From ni_test, sub-sample ni_subsample.
      size_t[] a, b;
      for (size_t i = 0; i < ni_subsample; i++) {
        a ~= 0;
      }
      for (size_t i = 0; i < ni_test; i++) {
        b ~= i;
      }

      //gsl_ran_choose(gsl_r, static_cast<void *>(&a[0]), ni_subsample,
      //               static_cast<void *>(&b[0]), ni_test, sizeof(size_t));

      // Re-set indicator_idv and ni_test.
      int j = 0;
      for (size_t i = 0; i < (indicator_idv).length; ++i) {
        if (indicator_idv[i] == 0) {
          continue;
        }
        //////////////////////////////////
        //             FIXME            //
        //////////////////////////////////
        //if (find(a.begin(), a.end(), j) == a.end()) {
        //  indicator_idv[i] = 0;
        //}
        j++;
      }
      ni_test = ni_subsample;
    }
  }

  // Check ni_test.
  if (ni_test == 0 && a_mode != 15) {
    error = true;
    writeln("error! number of analyzed individuals equals 0. ");
    return;
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

    for (size_t i = 0; i < indicator_idv.length; ++i) {
      indicator_cvt ~= 1;
      cvt ~= cvt_row;
    }
  }

  return;
}

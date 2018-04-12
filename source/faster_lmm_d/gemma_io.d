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
import faster_lmm_d.helpers;
import faster_lmm_d.optmatrix;

import gsl.permutation;
import gsl.rng;
import gsl.randist;


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
void readfile_bed(const string file_bed, const string[] setSnps,
                  const DMatrix W, int[] indicator_idv,
                  int[] indicator_snp, SNPINFO[] snpInfo,
                  const double maf_level, const double miss_level,
                  const double hwe_level, const double r2_level,
                  size_t ns_test) {
  writeln("entered readfile_bed");
  indicator_snp = [];
  size_t ns_total = snpInfo.length;

  File infile = File(file_bed);

  DMatrix genotype; // = gsl_vector_safe_alloc(W->size1);
  DMatrix genotype_miss; // = gsl_vector_safe_alloc(W->size1);

  gsl_permutation *pmt = gsl_permutation_alloc(W.shape[1]);

  DMatrix WtW = matrix_mult(W.T, W);
  int sig;
  DMatrix WtWi = WtW.inverse;

  double v_x, v_w, geno;
  size_t c_idv = 0;

  char ch;
  int[] b;

  size_t ni_total = indicator_idv.length;
  size_t ni_test = 0;
  for (size_t i = 0; i < ni_total; ++i) {
    ni_test += indicator_idv[i];
  }
  ns_test = 0;

  // Calculate n_bit and c, the number of bit for each snp.
  size_t n_bit;
  if (ni_total % 4 == 0) {
    n_bit = ni_total / 4;
  } else {
    n_bit = ni_total / 4 + 1;
  }

  // Ignore the first three magic numbers.
  for (int i = 0; i < 3; ++i) {
    // FIXME
    //infile.read(ch, 1);
    //b = ch;
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
      //infile.read(ch, 1);
      //b = ch[0];

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

    snpInfo[t].n_miss = n_miss;
    snpInfo[t].missingness = to!double(n_miss) / to!double(ni_test);
    snpInfo[t].maf = maf;
    snpInfo[t].n_idv = ni_test - n_miss;
    snpInfo[t].n_nb = 0;
    snpInfo[t].file_position = t;

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

  gsl_permutation_free(pmt);

  //return true;
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

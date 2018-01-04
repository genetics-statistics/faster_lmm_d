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
import faster_lmm_d.helpers;
import faster_lmm_d.optmatrix;

void generate_kinship(string geno_fn, string pheno_fn, bool test_nind= false){

  string filename = geno_fn;
  auto pipe = pipeShell("gunzip -c " ~ filename);
  File input = pipe.stdout;

  int k_mode = 0;

  size_t n_miss;
  double d, geno_mean, geno_var;

  // setKSnp and/or LOCO support
  //bool process_ksnps = ksnps.size();

  DMatrix matrix_kin;
  size_t ni_total = matrix_kin.shape[0];

  double[] indicator_snp;


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
  //for (size_t t = 0; t < indicator_snp.size(); ++t) {

    if (indicator_snp[t] == 0)
      continue;

    auto chr = to!string(line).split(",")[3..$];

    if (test_nind) {
      // ascertain the number of genotype fields match

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
      if (chr[i] == "NA") {
        geno_miss[i] = 0;
        n_miss++;
      } else {
        d = to!double(chr[i]);
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

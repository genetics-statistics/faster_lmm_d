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

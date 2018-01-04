/*
   This code is part of faster_lmm_d and published under the GPLv3
   License (see LICENSE.txt)

   Copyright © 2017 - 2018 Prasun Anand & Pjotr Prins
*/

module test.covar_matrix;

import std.exception;
import std.experimental.logger;
import std.math : round, isNaN;
import std.stdio;

import faster_lmm_d.dmatrix;
import faster_lmm_d.helpers : modDiff, sum;

void check_covar_matrix(DMatrix  covar_matrix, string geno_fn){
  trace("covar_matrix.shape: ", covar_matrix.shape, "\n",
        covar_matrix.elements[0],  ",", covar_matrix.elements[1], ",", covar_matrix.elements[2], "...",
        covar_matrix.elements[$-3], ",", covar_matrix.elements[$-2], ",", covar_matrix.elements[$-1]);

  double c1 = covar_matrix.elements[0];
  double c2 = covar_matrix.elements[1];
  double cl = covar_matrix.elements[$-1];
  if(geno_fn == "data/rqtl/recla_geno.csv"){
    info("Validating results for ", geno_fn);
    enforce(covar_matrix.shape == [261,2]);
    enforce(modDiff(c1,1) == 0);
    enforce(modDiff(c2,1) == 0);
    enforce(modDiff(cl,0) == 0);
  }
  if(geno_fn == "data/rqtl/iron_geno.csv"){
    info("Validating results for ", geno_fn);
    enforce(covar_matrix.shape == [284, 2]);
    enforce(modDiff(c1,1) == 0);
    enforce(modDiff(c2,0) == 0);
    enforce(modDiff(cl,0) == 0);
  }
  info("Covariates Matrix test successful");
}

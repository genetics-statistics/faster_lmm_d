/*
   This code is part of faster_lmm_d and published under the GPLv3
   License (see LICENSE.txt)

   Copyright © 2017 - 2018 Prasun Anand & Pjotr Prins
*/

module test.pheno_vector;

import std.exception;
import std.experimental.logger;
import std.math : round, isNaN;
import std.stdio;

import faster_lmm_d.dmatrix;
import faster_lmm_d.helpers : modDiff;

void check_pheno_vector(double[] pheno_vector, string geno_fn){
  trace("pheno_vector.length: ", pheno_vector.length, "\n",
        pheno_vector[0], pheno_vector[1], pheno_vector[2], "...",
        pheno_vector[$-3], pheno_vector[$-2], pheno_vector[$-1]);

  double pv1 = pheno_vector[0];
  double pv2 = pheno_vector[1];
  double pvl = pheno_vector[$-1];
  if(geno_fn == "data/small.geno"){
    info("Validating results for ", geno_fn);
    enforce(modDiff(pv1, 0.5783) < 0.001);
    enforce(modDiff(pv2, 0.1957) < 0.001);
    enforce(modDiff(pvl,-1.9122) < 0.001);
  }
  if(geno_fn == "data/small_na.geno"){
    info("Validating results for ", geno_fn);
    enforce(modDiff(pv1, 0.578326) < 0.001);
    enforce(isNaN(pv2));
    enforce(modDiff(pvl,-1.9122)<0.001);
  }
  if(geno_fn == "data/genenetworpv/BXD.csv"){
    info("Validating results for ", geno_fn);
    enforce(modDiff(pv1, 61.92) < 0.001);
    enforce(modDiff(pv2, 88.33) < 0.001);
    enforce(modDiff(pvl, 73.84) < 0.001);
  }
  if(geno_fn == "data/rqtl/recla_geno.csv"){
    info("Validating results for ", geno_fn);
    enforce(modDiff(pv1, 1367.31) < 0.001);
    enforce(modDiff(pv2, 2598.86) < 0.001);
    enforce(modDiff(pvl,  901.09) < 0.001);
  }
  if(geno_fn == "data/rqtl/iron_geno.csv"){
    info("Validating results for ", geno_fn);
    enforce(modDiff(pv1, 61.92) < 0.001);
    enforce(modDiff(pv2, 88.33) < 0.001);
    enforce(modDiff(pvl, 73.84) < 0.001);
  }
  if(geno_fn == "data/test8000.geno"){
    info("Validating results for ", geno_fn);
    enforce(modDiff(pv1, 0.578326) < 0.001);
    enforce(modDiff(pv2, 0.195782) < 0.001);
    enforce(modDiff(pvl,-0.429985) < 0.001);
  }
  info("Phenotype Vector test successful");
}

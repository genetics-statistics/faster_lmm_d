module test.geno_matrix;

import std.exception;
import std.experimental.logger;
import std.math : round, isNaN;
import std.stdio;

import faster_lmm_d.dmatrix;
import faster_lmm_d.helpers : modDiff, sum;

void check_geno_matrix(DMatrix  geno_matrix, string geno_fn){
  trace("geno_matrix.shape: ", geno_matrix.shape, "\n",
        geno_matrix.elements[0],  ",", geno_matrix.elements[1], ",", geno_matrix.elements[2], "...",
        geno_matrix.elements[$-3], ",", geno_matrix.elements[$-2], ",", geno_matrix.elements[$-1]);

  double g1 = geno_matrix.elements[0];
  double g2 = geno_matrix.elements[1];
  double gl = geno_matrix.elements[$-1];
  if(geno_fn == "data/small.geno"){
    info("Validating results for ", geno_fn);
    enforce(modDiff(g1, 1) < 0.001);
    enforce(modDiff(g2, 1) < 0.001);
    enforce(modDiff(gl, 1) < 0.001);
  }
  if(geno_fn == "data/small_na.geno"){
    info("Validating results for ", geno_fn);
    enforce(modDiff(g1, 1) < 0.001);
    enforce(modDiff(g2, 1) < 0.001);
    enforce(modDiff(gl, 0) < 0.001);
  }
  if(geno_fn == "data/genenetworg/BXD.csv"){
    info("Validating results for ", geno_fn);
    enforce(modDiff(g1, 1) < 0.001);
    enforce(isNaN(g2));
    enforce(modDiff(gl, 0.5) < 0.001);
  }
  if(geno_fn == "data/rqtl/recla_geno.csv"){
    info("Validating results for ", geno_fn);
    enforce(modDiff(g1, 0) < 0.001);
    enforce(modDiff(g2, 0.5 ) < 0.001);
    enforce(modDiff(gl, 0) < 0.001);
  }
  if(geno_fn == "data/rqtl/iron_geno.csv"){
    info("Validating results for ", geno_fn);
    enforce(modDiff(g1, 1) < 0.001);
    enforce(isNaN(g2));
    enforce(modDiff(gl, 0.5) < 0.001);
  }
  if(geno_fn == "data/test8000.geno"){
    info("Validating results for ", geno_fn);
    enforce(modDiff(g1, 0) < 0.001);
    enforce(modDiff(g2, 0) < 0.001);
    enforce(modDiff(gl, 0) < 0.001);
  }
  info("Phenotype Vector test successful");
}

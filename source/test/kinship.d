module test.kinship;

import std.exception;
import std.experimental.logger;
import std.math : round;
import std.stdio;

import faster_lmm_d.dmatrix;
import faster_lmm_d.helpers : modDiff;

void check_kinship(DMatrix K, string geno_fn){
  trace(K.shape, "\n", K.elements[0], K.elements[1], K.elements[2], "...", K.elements[$-3], K.elements[$-2], K.elements[$-1]);
  double k1 = K.elements[0];
  double k2 = K.elements[1];
  double kl = K.elements[$-1];
  if(geno_fn == "data/small.geno"){
    info("Validating results for ", geno_fn);
    enforce(modDiff(k1,0.9375)<0.001);
    enforce(modDiff(k2,0.5202)<0.001);
    enforce(modDiff(kl,0.6597)<0.001);
  }
  if(geno_fn == "data/small_na.geno"){
    info("Validating results for ", geno_fn);
    enforce(modDiff(k1, 1.25) <0.001);
    enforce(modDiff(k2,-0.41666)<0.001);
    enforce(modDiff(kl, 1)<0.001);
  }
  if(geno_fn == "data/genenetwork/BXD.csv"){
    info("Validating results for ", geno_fn);
    enforce(modDiff(k1, 1.02346)<0.001);
    enforce(modDiff(k2,-0.10832)<0.001);
    enforce(modDiff(kl, 0.98783)<0.001);
  }
  if(geno_fn == "data/rqtl/recla_geno.csv"){
    info("Validating results for ", geno_fn);
    enforce(modDiff(k1, 0.97429)<0.001);
    enforce(modDiff(k2, 0.03363)<0.001);
    enforce(modDiff(kl, 0.98084)<0.001);
  }
  if(geno_fn == "data/rqtl/iron_geno.csv"){
    info("Validating results for ", geno_fn);
    enforce(modDiff(k1, 0.98876)<0.001);
    enforce(modDiff(k2, 0.07252)<0.001);
    enforce(modDiff(kl, 0.96619)<0.001);
  }
  if(geno_fn == "data/test8000.geno"){
    info("Validating results for ", geno_fn);
    enforce(modDiff(k1, 1.43484)<0.001);
    enforce(modDiff(k2, 1.43484)<0.001);
    enforce(modDiff(kl, 1.26798)<0.001);
  }
  info("Kinship test successful");
}
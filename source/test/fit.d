module test.fit;

import std.exception;
import std.experimental.logger;
import std.math : round, isNaN;
import std.stdio;

import faster_lmm_d.dmatrix;
import faster_lmm_d.helpers : modDiff, sum;
import faster_lmm_d.lmm2;

void check_lmm_fit(LMM lmm_object, string geno_fn){

  double heritability = lmm_object.opt_H;
  DMatrix beta= lmm_object.opt_beta;
  double sigma = lmm_object.opt_sigma;
  double sigmasq_g =  heritability * sigma;
  double sigmasq_e = (1- heritability ) * sigma;
  double loglik = lmm_object.opt_LL;

  if(geno_fn == "data/small.geno"){
    info("Validating results for ", geno_fn);
    enforce(modDiff(heritability, 0.465456) < 0.001);
    enforce(beta.shape == [1,1]);
    enforce(modDiff(beta.elements[0], -0.71432) < 0.001);
    enforce(modDiff(sigma, 1.19048) < 0.001);
    enforce(modDiff(sigmasq_g, 0.5541) < 0.001);
    enforce(modDiff(sigmasq_e, 0.6363) < 0.001);
    enforce(modDiff(loglik, -5.92186) < 0.001);
  }
  if(geno_fn == "data/small_na.geno"){
    info("Validating results for ", geno_fn);
    enforce(modDiff(heritability, 0.99) < 0.001);
    enforce(beta.shape == [1,1]);
    enforce(modDiff(beta.elements[0], -0.618393) < 0.001);
    enforce(modDiff(sigma, 0.853035) < 0.001);
    enforce(modDiff(sigmasq_g, 0.844505) < 0.001);
    enforce(modDiff(sigmasq_e, 0.008530) < 0.001);
    enforce(modDiff(loglik, -2.82463) < 0.001);
  }
  if(geno_fn == "data/genenetworg/BXD.csv"){
    info("Validating results for ", geno_fn);
    enforce(modDiff(heritability, 0.465456) < 0.001);
    enforce(beta.shape == [1,1]);
    enforce(modDiff(beta.elements[0], -0.71432) < 0.001);
    enforce(modDiff(sigma, 1.19048) < 0.001);
    enforce(modDiff(sigmasq_g, 0.5541) < 0.001);
    enforce(modDiff(sigmasq_e, 0.6363) < 0.001);
    enforce(modDiff(loglik, -5.92186) < 0.001);
  }
  if(geno_fn == "data/rqtl/recla_geno.csv"){
    info("Validating results for ", geno_fn);
    enforce(modDiff(heritability, 0.4076) < 0.001);
    enforce(beta.shape == [1,1]);
    enforce(modDiff(beta.elements[0], 1382.1153) < 0.001);
    enforce(modDiff(sigma, 266543.673) < 0.001);
    enforce(modDiff(sigmasq_g, 108643.161) < 0.001);
    enforce(modDiff(sigmasq_e, 157900.511) < 0.001);
    enforce(modDiff(loglik, -1982.034) < 0.001);
  }
  if(geno_fn == "data/rqtl/iron_geno.csv"){
    info("Validating results for ", geno_fn);
    enforce(modDiff(heritability, 0.428702) < 0.001);
    enforce(beta.shape == [1,1]);
    enforce(modDiff(beta.elements[0], 94.5198) < 0.001);
    enforce(modDiff(sigma, 1544.856) < 0.001);
    enforce(modDiff(sigmasq_g, 662.282) < 0.001);
    enforce(modDiff(sigmasq_e, 882.573) < 0.001);
    enforce(modDiff(loglik, -1391.851) < 0.001);
  }
  if(geno_fn == "data/test8000.geno"){
    info("Validating results for ", geno_fn);
    enforce(modDiff(heritability, 0.0045) < 0.001);
    enforce(beta.shape == [1,1]);
    enforce(modDiff(beta.elements[0], 0.0230974) < 0.001);
    enforce(modDiff(sigma, 0.987496) < 0.001);
    enforce(modDiff(sigmasq_g, 0.0045) < 0.001);
    enforce(modDiff(sigmasq_e, 0.9829) < 0.001);
    enforce(modDiff(loglik, -1717.274) < 0.001);
  }
  info("LMM fit test successful");
}
/*
   This code is part of faster_lmm_d and published under the GPLv3
   License (see LICENSE.txt)

   Copyright © 2017 Prasun Anand & Pjotr Prins
*/

module faster_lmm_d.gemma;

import core.stdc.stdlib : exit;

import std.conv;
import std.file;
import std.experimental.logger;
import std.exception;
import std.parallelism;
import std.range;
import std.stdio;
import std.typecons;
import std.process;
import std.string;

import faster_lmm_d.dmatrix;
import faster_lmm_d.kinship;
import faster_lmm_d.lmm2;
import faster_lmm_d.memory;
import faster_lmm_d.optmatrix;
import faster_lmm_d.output;
import faster_lmm_d.phenotype;
import faster_lmm_d.helpers : sum;

import test.covar_matrix;
import test.fit;
import test.geno_matrix;
import test.kinship;

DMatrix read_matrix_from_file(string filename){
  string input = to!string(std.file.read(filename));

  string[] lines = input.split("\n");
  size_t cols;
  size_t rows = lines.length - 1;
  double[] elements;
  foreach(line; lines[0..$-1]){
    string[] items = line.split("\t");
    foreach(item; items){
      elements ~= to!double(item) ;
    }
  }
  return DMatrix([rows, elements.length/rows], elements);
}

void run_gemma(string option_kinship, string option_pheno, string option_covar, string option_geno){

  writeln("reading kinship " , option_kinship);
  DMatrix K = read_matrix_from_file(option_kinship);
  writeln(K.shape);
  writeln(K.elements.length);

  writeln("reading pheno " , option_pheno);
  DMatrix Y = read_matrix_from_file(option_pheno);
  writeln(Y.shape);
  writeln(Y.elements.length);

  writeln("reading covar " , option_covar);
  DMatrix covar_matrix = read_matrix_from_file(option_covar);
  writeln(covar_matrix.shape);
  writeln(covar_matrix.elements.length);

  println("Compute GWAS");
  auto N = cast(N_Individuals)K.shape[0];
  auto kvakve = kvakve(K);

  LMM lmm1 = LMM(Y.elements, kvakve.kva, covar_matrix);
  auto lmm2 = lmm_transform(lmm1,N,Y.elements,kvakve.kve);

  trace("Computing fit for null model");

  auto lmm = lmm_fit(lmm2, N);

  writeln(
    "\nheritability = ", lmm.opt_H,
    "\nsum = ", sum(lmm.opt_beta.elements),
    "\nbeta = ", lmm.opt_beta,
    "\nsigma = ", lmm.opt_sigma,
    "\nsigmasq_g = ", lmm.opt_H * lmm.opt_sigma,
    "\nsigmasq_e = ", (1 - lmm.opt_H) * lmm.opt_sigma,
    "\nlog-likelihood = ", lmm.opt_LL,
  );

  DMatrix KveT = kvakve.kve.T; // compute out of loop
  trace("Call data offload");
  offload_cache(KveT);             // send this to the cache

  writeln(option_geno);

  //auto pipe = pipeShell(`gunzip -c ./example/mouse_hs1940.geno.txt.gz`, Redirect.all);
  ////auto pipe = pipeShell(`subl`);
  //File input = pipe.stdout;

  double[] output;
  auto file = File("./example/mouse_hs1940.geno.txt");

  int count = 0;
  foreach(line; file.byLine){
    if(line == "")  break;
    auto cells = line.split(",");
    foreach(cell; cells[3..$]){
      output ~= to!double((to!string(cell)).strip());
    }
    count+=1;
  }
  //writeln(output);

  DMatrix lol  = DMatrix([count, output.length/count ], output);

  DMatrix G = lol.T;

  writeln(G.shape);
  auto inds = G.cols();
  auto snps = G.rows();


//  version(PARALLEL) {
//    auto tsps = new TStat[snps];
//    auto items = iota(0,snps).array;

//    println("Parallel");
//    foreach (ref snp; taskPool.parallel(items,100)) {
//      tsps[snp] = lmm_association(snp, lmm, N, G, KveT);
//      if((snp+1) % 1000 == 0){
//        println(snp+1, " snps processed");
//      }
//    }
//  } else {
    TStat[] tsps;
    foreach(snp; 0..snps) {
      tsps ~= lmm_association(snp, lmm, N, G, KveT);
      if((snp+1) % 1000 == 0){
        println(snp+1, " snps processed");
      }
    }
//  }

  writeln(tsps);

}

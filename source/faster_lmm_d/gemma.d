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
import std.math;
import std.parallelism;
import std.range;
import std.stdio;
import std.typecons;
import std.process;
import std.string;

import faster_lmm_d.dmatrix;
import faster_lmm_d.gemma_helpers;
import faster_lmm_d.gemma_lmm;
import faster_lmm_d.gemma_param;
import faster_lmm_d.kinship;
import faster_lmm_d.lmm2;
import faster_lmm_d.memory;
import faster_lmm_d.optmatrix;
import faster_lmm_d.output;
import faster_lmm_d.phenotype;
import faster_lmm_d.helpers : sum;

import gsl.cdf;
import gsl.errno;
import gsl.math;
import gsl.min;
import gsl.roots;

import faster_lmm_d.helpers : modDiff, sum;
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
      elements ~= (item == "NA" ? 0 : to!double(item)) ;
    }
  }
  return DMatrix([rows, elements.length/rows], elements);
}

DMatrix read_matrix_from_file2(string filename){
  string input = to!string(std.file.read(filename));

  string[] lines = input.split("\n");
  size_t cols;
  size_t rows = lines.length - 1;
  double[] elements;
  foreach(line; lines[0..$-1]){
    string[] items = line.strip().split("\t");
    foreach(item; items){
      elements ~= (item == "NA" ? 0 : to!double(item)) ;
    }
  }
  return DMatrix([rows, elements.length/rows], elements);
}

// Kronecker product.
void Kronecker(const DMatrix K, const DMatrix V, DMatrix H) {
  for (size_t i = 0; i < K.shape[0]; i++) {
    for (size_t j = 0; j < K.shape[1]; j++) {
      DMatrix H_sub = get_sub_dmatrix(
          H, i * V.shape[0], j * V.shape[1], V.shape[0], V.shape[1]);
      H_sub.elements = V.elements.dup;
      H_sub = multiply_dmatrix_num(H_sub, accessor(K, i, j));
    }
  }
  return;
}

// Symmetric K matrix.
void KroneckerSym(const DMatrix K, const DMatrix V, DMatrix H) {
  for (size_t i = 0; i < K.shape[0]; i++) {
    for (size_t j = i; j < K.shape[1]; j++) {
      DMatrix H_sub = get_sub_dmatrix(
          H, i * V.shape[0], j * V.shape[1], V.shape[0], V.shape[1]);
      H_sub.elements = V.elements.dup;
      H_sub = multiply_dmatrix_num(H_sub, accessor(K, i, j));

      if (i != j) {
        DMatrix H_sub_sym = get_sub_dmatrix(
            H, j * V.shape[0], i * V.shape[1], V.shape[0], V.shape[1]);
        H_sub_sym.elements = H_sub.elements.dup;
      }
    }
  }
  return;
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


void CalcVCss(DMatrix a, DMatrix b, DMatrix c, DMatrix d, DMatrix e,
             ulong f, DMatrix g, DMatrix h, double i, double j, DMatrix k,
             DMatrix l, DMatrix m, DMatrix n){

}

void batch_run(string option_kinship, string option_pheno, string option_covar, 
              string option_geno, string indicator_idv, string indicator_snp, 
              string test_name){

  // Read Files.

  writeln("reading pheno " , option_pheno);
  DMatrix Y = read_matrix_from_file(option_pheno);
  writeln(Y.shape);
  //writeln(Y); also y

  writeln("reading covar " , option_covar);
  DMatrix covar_matrix = (option_covar != "" ? read_matrix_from_file(option_covar) : ones_dmatrix(Y.shape[0], Y.shape[1]));
  //DMatrix covar_matrix = ones_dmatrix(Y.shape[0], Y.shape[1]);
  writeln(covar_matrix.shape);

  writeln("reading kinship " , option_kinship);
  DMatrix G = read_matrix_from_file(option_kinship);
  writeln(G.shape);

  DMatrix U, eval;
  eval.shape = [1, Y.elements.length];
  U.shape = [Y.elements.length, Y.elements.length];

  auto k = kvakve(G);
  eval = k.kva;
  U = k.kve;

  DMatrix UtW = matrix_mult(U.T, covar_matrix);
  DMatrix Uty = matrix_mult(U.T, Y); 
  Param cPar;
  cPar.trace_G = sum(eval.elements)/eval.elements.length;
  cPar.a_mode = 1;
  cPar.indicator_idv_file = indicator_idv;
  cPar.indicator_snp_file = indicator_snp;
  cPar.file_geno = option_geno;
  writeln("Reading Files ... ");
  cPar.ReadFiles();
  if (cPar.error == true) {
    writeln("error! fail to read files. ");
    return;
  }
  cPar.CheckData();
  if (cPar.error == true) {
    writeln("error! fail to check data. ");
    return;
  }

  fit_model(cPar, U, eval, UtW, Uty, Y, covar_matrix, test_name);

  return;
}

void fit_model(Param cPar, DMatrix U, DMatrix eval, DMatrix  UtW, DMatrix UtY, DMatrix Y, DMatrix W, string test_name, size_t n_ph = 1){
  writeln("In LMM fit_model");

  if (n_ph == 1) { // one phenotype
    writeln ("Calculating REMLE/MLE");

    DMatrix beta = DMatrix([1,1] , [0]);
    DMatrix se_beta = DMatrix([1,1] , [0]);
    DMatrix UtY_col = get_col(UtY, 0);

    cPar.l_min = 0.000010;
    cPar.l_max = 100000;
    cPar.n_region = 10;

    CalcLambda('L', eval, UtW, UtY_col, cPar.l_min, cPar.l_max,
               cPar.n_region, cPar.l_mle_null, cPar.logl_mle_H0);

    writeln("==============cPar.l_mle_null=======================");
    writeln(cPar.l_mle_null);
    writeln(cPar.logl_mle_H0);

    CalcLmmVgVeBeta(eval, UtW, UtY_col, cPar.l_mle_null,
                    cPar.vg_mle_null, cPar.ve_mle_null, beta,
                    se_beta);


    CalcLambda('R', eval, UtW, UtY_col, cPar.l_min, cPar.l_max,
               cPar.n_region, cPar.l_remle_null, cPar.logl_remle_H0);

    writeln(cPar.l_remle_null);
    writeln(cPar.logl_remle_H0);

    CalcLmmVgVeBeta(eval, UtW, UtY_col, cPar.l_remle_null,
                    cPar.vg_remle_null, cPar.ve_remle_null, beta,
                    se_beta);

    CalcPve(eval, UtW, UtY_col, cPar.l_remle_null, cPar.trace_G,
            cPar.pve_null, cPar.pve_se_null);

    check_lambda(test_name, cPar);

    // calculate and output residuals
    if (cPar.a_mode == 5) {
      DMatrix Utu_hat;
      Utu_hat.shape = [1, Y.shape[0]];
      DMatrix Ute_hat;
      Ute_hat.shape = [1, Y.shape[0]];
      DMatrix u_hat;
      u_hat.shape = [1, Y.shape[0]];
      DMatrix e_hat;
      e_hat.shape = [1, Y.shape[0]];
      DMatrix y_hat;
      y_hat.shape = [1, Y.shape[0]];

      // obtain Utu and Ute
      y_hat.elements = UtY_col.elements.dup;
      //gsl_blas_dgemv(CblasNoTrans, -1.0, UtW, &beta.vector, 1.0, y_hat);

      double d, u, e;
      for (size_t i = 0; i < eval.elements.length; i++) {
        d = eval.elements[i];
        u = cPar.l_remle_null * d / (cPar.l_remle_null * d + 1.0) *
            y_hat.elements[i];
        e = 1.0 / (cPar.l_remle_null * d + 1.0) * y_hat.elements[i];
        Utu_hat.elements[i] = u;
        Ute_hat.elements[i] = e;
      }

      // output residuals
      cPar.WriteVector(u_hat, "residU");
      cPar.WriteVector(e_hat, "residE");
    }
  }

  DMatrix Y_col = get_col(Y, 0);
  DMatrix UtY_col = get_col(UtY, 0);

  GWAS_SNPs setGWASnps;


 
  AnalyzeBimbam(cPar, U, eval, UtW, UtY_col, W, Y_col, setGWASnps, 1);
}

void check_lambda(string test_name, Param cPar){
  writeln("Test name ->", test_name);
  if (test_name == "mouse_hs1940"){
    enforce(modDiff(cPar.l_mle_null, 4.34046) < 0.001);
    enforce(modDiff(cPar.l_remle_null, 4.32887) < 0.001);
    enforce(modDiff(cPar.logl_mle_H0, -1584.7216) < 0.001);
    enforce(modDiff(cPar.logl_remle_H0, -1584.3408) < 0.001);

    enforce(modDiff(cPar.vg_mle_null, 1.4794) < 0.001);
    enforce(modDiff(cPar.ve_mle_null, 0.34084) < 0.001);

    enforce(modDiff(cPar.vg_remle_null, 1.47657) < 0.001);
    enforce(modDiff(cPar.ve_remle_null, 0.341097) < 0.001);

    enforce(modDiff(cPar.trace_G, 0.359504) < 0.001);
    enforce(modDiff(cPar.pve_null, 0.608801) < 0.001);
    enforce(modDiff(cPar.pve_se_null, 0.032774) < 0.001);    

    //enforce(modDiff(cPar.beta, 0) < 0.001);
    //enforce(modDiff(cPar.se_beta, 0) < 0.001);
    //enforce(modDiff(cPar.pve_se_null, 0) < 0.001);
    writeln("===================Tests pass=====================");
  }else if(test_name == "BXD"){
    enforce(modDiff(cPar.l_mle_null, 1.07705) < 0.001);
    enforce(modDiff(cPar.l_remle_null, 1.03727) < 0.001);
    enforce(modDiff(cPar.logl_mle_H0, -571.815) < 0.001);
    enforce(modDiff(cPar.logl_remle_H0, -569.519) < 0.001);
    
    enforce(modDiff(cPar.vg_mle_null, 16.9179) < 0.001);
    enforce(modDiff(cPar.ve_mle_null, 15.7076) < 0.001);

    enforce(modDiff(cPar.vg_remle_null, 16.3864) < 0.001);
    enforce(modDiff(cPar.ve_remle_null, 15.7976) < 0.001);

    enforce(modDiff(cPar.trace_G, 0.226114) < 0.001);
    enforce(modDiff(cPar.pve_null, 0.189983) < 0.001);
    enforce(modDiff(cPar.pve_se_null, 0.109182) < 0.001); 
    writeln("===================BXD Tests pass=====================");   
  }    
}

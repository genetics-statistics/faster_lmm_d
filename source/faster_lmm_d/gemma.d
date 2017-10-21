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

// Calculate UtX.
void CalcUtX(const DMatrix U, DMatrix UtX) {
  DMatrix X;
  X.shape = [UtX.shape[0], UtX.shape[1]];
  X.elements =  UtX.elements.dup;
  //eigenlib_dgemm("T", "N", 1.0, U, X, 0.0, UtX);
  return;
}

void CalcUtX(const DMatrix U, const DMatrix X, DMatrix UtX) {
  //eigenlib_dgemm("T", "N", 1.0, U, X, 0.0, UtX);
  return;
}

//void CalcUtX(const DMatrix U, const DMatrix x, DMatrix Utx) {
//  //gsl_blas_dgemv(CblasTrans, 1.0, U, x, 0.0, Utx);
//  return;
//}

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

void batch_run(string option_kinship, string option_pheno, string option_covar, string option_geno, string indicator_idv, string indicator_snp){

  // Read Files.

  writeln("reading pheno " , option_pheno);
  DMatrix Y = read_matrix_from_file(option_pheno);
  writeln(Y.shape);
  //writeln(Y); also y

  writeln("reading covar " , option_covar);
  //DMatrix covar_matrix = read_matrix_from_file(option_covar);
  DMatrix covar_matrix = ones_dmatrix(Y.shape[0], Y.shape[1]);
  writeln(covar_matrix.shape);
  //writeln(covar_matrix); also w

  writeln("reading kinship " , option_kinship);
  DMatrix G = read_matrix_from_file(option_kinship);
  writeln(G.shape);

  DMatrix U, eval;
  eval.shape = [1, Y.elements.length];
  U.shape = [Y.elements.length, Y.elements.length];

  //double trace_G = EigenDecomp_Zeroed(G, U, eval, 0);
  auto k = kvakve(G);
  eval = k.kva;
  U = k.kve;

  DMatrix UtW = matrix_mult(U.T, covar_matrix);
  DMatrix Uty = matrix_mult(U.T, Y); 
  Param cPar;
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

  fit_model(cPar, U, eval, UtW, Uty, Y, covar_matrix);

  return;
}




//change file_name
void kinship_mode_43(Param cPar){
  // first, use individuals with full phenotypes to obtain estimates of Vg and
  // Ve
  DMatrix Y;
  Y.shape = [cPar.ni_test, cPar.n_ph];
  DMatrix W;
  W.shape = [Y.shape[0], cPar.n_cvt];
  DMatrix G;
  G.shape = [Y.shape[0], Y.shape[0]];
  DMatrix U;
  U.shape = [Y.shape[0], Y.shape[0]];
  DMatrix UtW;
  UtW.shape = [Y.shape[0], W.shape[1]];
  DMatrix UtY;
  UtY.shape = [Y.shape[0], Y.shape[1]];
  DMatrix eval;
  eval.shape = [1, Y.shape[0]];

  DMatrix Y_full;
  Y_full.shape = [cPar.ni_cvt, cPar.n_ph];
  DMatrix W_full;
  W_full.shape = [Y_full.shape[0], cPar.n_cvt];

  cPar.CopyCvtPhen(W, Y, 0);
  cPar.CopyCvtPhen(W_full, Y_full, 1);

  DMatrix Y_hat;
  Y_hat.shape = [Y_full.shape[0], cPar.n_ph];
  DMatrix G_full;
  G_full.shape = [Y_full.shape[0], Y_full.shape[0]];
  DMatrix H_full;
  H_full.shape = [Y_full.shape[0] * Y_hat.shape[1],
                                        Y_full.shape[0] * Y_hat.shape[1]];

  if (cPar.error == true) {
    writeln("error! fail to read kinship/relatedness file.");
    return;
  }
  // This is not so elegant. Reads twice to select on idv and then cvt
  ReadFile_kin(cPar.file_kin, cPar.indicator_cvt, cPar.mapID2num, cPar.k_mode, cPar.error, G_full);

  if (cPar.error == true) {
    writeln("error! fail to read kinship/relatedness file.");
    return;
  }

  // center matrix G
  CenterMatrix(G);
  CenterMatrix(G_full);
  validate_K(G,cPar.mode_check,cPar.mode_strict);

  // eigen-decomposition and calculate trace_G
  writeln("Start Eigen-Decomposition...");
  cPar.trace_G = EigenDecomp_Zeroed(G, U, eval, 0);

  // calculate UtW and Uty
  CalcUtX(U, W, UtW);
  CalcUtX(U, Y, UtY);

  // calculate variance component and beta estimates
  // and then obtain predicted values
  if (cPar.n_ph == 1) {
    DMatrix beta;
    beta.shape = [1, W.shape[1]];
    DMatrix se_beta;
    se_beta.shape = [1, W.shape[1]];

    double lambda, logl, vg, ve;
    DMatrix UtY_col = get_col(UtY, 0);

    // obtain estimates
    CalcLambda('R', eval, UtW, UtY_col, cPar.l_min, cPar.l_max,
               cPar.n_region, lambda, logl);
    CalcLmmVgVeBeta(eval, UtW, UtY_col, lambda, vg, ve, beta,
                    se_beta);

    writeln("REMLE estimate for vg in the null model = ", vg);
    writeln("REMLE estimate for ve in the null model = ", ve);
    cPar.vg_remle_null = vg;
    cPar.ve_remle_null = ve;

    // obtain Y_hat from fixed effects
    DMatrix Yhat_col = get_col(Y_hat, 0);
    //gsl_blas_dgemv(CblasNoTrans, 1.0, W_full, beta, 0.0, &Yhat_col.vector);

    // obtain H
    //gsl_matrix_set_identity(H_full);
    H_full = multiply_dmatrix_num(H_full, ve);
    G_full = multiply_dmatrix_num(G_full, vg);
    H_full = add_dmatrix(H_full, G_full);

    // free matrices
  } else {
    DMatrix Vg;
    Vg.shape = [cPar.n_ph, cPar.n_ph];
    DMatrix Ve;
    Ve.shape = [cPar.n_ph, cPar.n_ph];
    DMatrix B;
    B.shape = [cPar.n_ph, W.shape[1]];
    DMatrix se_B;
    se_B.shape = [cPar.n_ph, W.shape[1]];

    writeln("REMLE estimate for Vg in the null model: ");
    for (size_t i = 0; i < Vg.shape[0]; i++) {
      for (size_t j = 0; j <= i; j++) {
        writeln("\t",accessor(Vg, i, j));
      }
      writeln();
    }
    writeln("REMLE estimate for Ve in the null model: ");
    for (size_t i = 0; i < Ve.shape[0]; i++) {
      for (size_t j = 0; j <= i; j++) {
        writeln("\t",accessor(Ve, i, j));
      }
      writeln();
    }
    //cPar.Vg_remle_null.clear();
    //cPar.Ve_remle_null.clear();
    for (size_t i = 0; i < Vg.shape[0]; i++) {
      for (size_t j = i; j < Vg.shape[1]; j++) {
        //cPar.Vg_remle_null.push_back(accessor(Vg, i, j));
        //cPar.Ve_remle_null.push_back(accessor(Ve, i, j));
      }
    }

    // obtain Y_hat from fixed effects
    //gsl_blas_dgemm(CblasNoTrans, CblasTrans, 1.0, W_full, B, 0.0, Y_hat);

    // obtain H
    KroneckerSym(G_full, Vg, H_full);
    for (size_t i = 0; i < G_full.shape[0]; i++) {
      DMatrix H_sub = get_sub_dmatrix(
          H_full, i * Ve.shape[0], i * Ve.shape[1], Ve.shape[0], Ve.shape[1]);
      H_sub = add_dmatrix(H_sub, Ve);
    }

    // free matrices
  }

  //TODO
  //PRDT cPRDT;

  //cPRDT.CopyFromParam(cPar);

  //writeln("Predicting Missing Phentypes ... ");
  //cPRDT.MvnormPrdt(Y_hat, H_full, Y_full);

  //cPRDT.WriteFiles(Y_full);
}




void fit_linear_model(Param cPar){
  DMatrix Y;
  Y.shape = [cPar.ni_test, cPar.n_ph];
  DMatrix W;
  W.shape = [Y.shape[0], cPar.n_cvt];

  // set covariates matrix W and phenotype matrix Y
  // an intercept should be included in W,
  cPar.CopyCvtPhen(W, Y, 0);

  // Fit LM or mvLM
  if (cPar.n_ph == 1) {
    LM cLm;
    cLm.CopyFromParam(cPar);

    DMatrix Y_col = get_col(Y, 0);

    if (!cPar.file_gene.empty()) {
      cLm.AnalyzeGene(W, Y_col); // y is the predictor, not the phenotype
    } else if (!cPar.file_bfile.empty()) {
      cLm.AnalyzePlink(W, Y_col);
    } else if (!cPar.file_oxford.empty()) {
      cLm.Analyzebgen(W, Y_col);
    } else {
      cLm.AnalyzeBimbam(W, Y_col);
    }

    cLm.WriteFiles();
    cLm.CopyToParam(cPar);
  }
  // release all matrices and vectors
}

void fit_model(Param cPar, DMatrix U, DMatrix eval, DMatrix  UtW, DMatrix UtY, DMatrix Y, DMatrix W, size_t n_ph = 1){
  writeln("In LMM fit_model");


  if (2 == 5) { // cPar.a_mode == 31
    //cPar.WriteMatrix(U, "eigenU");
    //cPar.WriteVector(eval, "eigenD");
  } else if (2 == 5) { //!cPar.file_gene.empty()
    // calculate UtW and Ut

    //assert_issue(cPar.issue == 26, ROUND(UtY.data[0]) == -16.6143);

    LMM cLmm;
    //cLmm.CopyFromParam(cPar);

    DMatrix Y_col = get_col(Y, 0);
    DMatrix UtY_col = get_col(UtY, 0);

    //cLmm.AnalyzeGene(U, eval, UtW, UtY_col, W,
    //                 Y_col); // y is the predictor, not the phenotype

    //cLmm.WriteFiles();
    //cLmm.CopyToParam(cPar);
  } else {

    // calculate REMLE/MLE estimate and pve for univariate model
    if (n_ph == 1) { // one phenotype
      writeln ("Calculating REMLE/MLE");
      //DMatrix beta = get_row(B, 0);
      //DMatrix se_beta = get_row(se_B, 0);
      DMatrix beta = DMatrix([1,1] , [0]);
      DMatrix se_beta = DMatrix([1,1] , [0]);
      DMatrix UtY_col = get_col(UtY, 0);

      //assert_issue(cPar.issue == 26, ROUND(UtY.data[0]) == -16.6143);
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

      writeln("================CalcLmmVgVeBeta==========================");
      writeln("vg_mle_null => ",cPar.vg_mle_null);
      writeln("ve_mle_null => ",cPar.ve_mle_null);
      writeln("beta => ", beta);
      writeln("se_beta => ", se_beta);

      //cPar.beta_mle_null.clear();
      //cPar.se_beta_mle_null.clear();
      //DMatrix B;
      //for (size_t i = 0; i < B.shape[1]; i++) {
        //cPar.beta_mle_null.push_back(accessor(B, 0, i));
        //cPar.se_beta_mle_null.push_back(accessor(se_B, 0, i));
      //}

      CalcLambda('R', eval, UtW, UtY_col, cPar.l_min, cPar.l_max,
                 cPar.n_region, cPar.l_remle_null, cPar.logl_remle_H0);


     

      writeln("==============cPar.l_remle_null=======================");
      writeln(cPar.l_remle_null);
      writeln(cPar.logl_remle_H0);

      CalcLmmVgVeBeta(eval, UtW, UtY_col, cPar.l_remle_null,
                      cPar.vg_remle_null, cPar.ve_remle_null, beta,
                      se_beta);

      writeln("================CalcLmmVgVeBeta==========================");
      writeln("vg_mle_null => ",cPar.vg_remle_null);
      writeln("ve_mle_null => ",cPar.ve_remle_null);
      writeln("beta => ", beta);
      writeln("se_beta => ", se_beta);

      //cPar.beta_remle_null.clear();
      //cPar.se_beta_remle_null.clear();
      //for (size_t i = 0; i < B.shape[1]; i++) {
      //  //cPar.beta_remle_null.push_back(accessor(B, 0, i));
      //  //cPar.se_beta_remle_null.push_back(accessor(se_B, 0, i));
      //}

      CalcPve(eval, UtW, UtY_col, cPar.l_remle_null, cPar.trace_G,
              cPar.pve_null, cPar.pve_se_null);
      //cPar.PrintSummary();
      check_lambda("", cPar);

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

        // obtain u and e //TODO
        //gsl_blas_dgemv(CblasNoTrans, 1.0, U, Utu_hat, 0.0, u_hat);
        //gsl_blas_dgemv(CblasNoTrans, 1.0, U, Ute_hat, 0.0, e_hat);

        // output residuals
        cPar.WriteVector(u_hat, "residU");
        cPar.WriteVector(e_hat, "residE");
      }
    }

    DMatrix Y_col = get_col(Y, 0);
    DMatrix UtY_col = get_col(UtY, 0);

    GWAS_SNPs setGWASnps;


   
    AnalyzeBimbam(cPar, U, eval, UtW, UtY_col, W, Y_col, setGWASnps, 1);

    // Fit LMM or mvLMM (w. LOCO)
    if (cPar.a_mode == 1 || cPar.a_mode == 2 || cPar.a_mode == 3 ||
        cPar.a_mode == 4) {
      if (cPar.n_ph == 1) {
        LMM cLmm;
        //cLmm.CopyFromParam(cPar);

         Y_col = get_col(Y, 0);
         UtY_col = get_col(UtY, 0);

        if (!cPar.file_bfile.empty()) {
          if (cPar.file_gxe.empty()) {
            //cLmm.AnalyzePlink(U, eval, UtW, UtY_col, W, Y_col);
          } else {
            //cLmm.AnalyzePlinkGXE(U, eval, UtW, UtY_col, W, Y_col, env);
          }
        }
        // WJA added
        else if (!cPar.file_oxford.empty()) {
          //cLmm.Analyzebgen(U, eval, UtW, UtY_col, W, Y_col);
        } else {
          if (cPar.file_gxe.empty()) {
            AnalyzeBimbam(cPar, U, eval, UtW, UtY_col, W, Y_col, setGWASnps, 1);
          } else {
            //cLmm.AnalyzeBimbamGXE(U, eval, UtW, UtY_col, W, Y_col, env);
          }
        }
      }
    }
  }
}

void check_lambda(string test_name, Param cPar){
  if (test_name == ""){
    enforce(modDiff(cPar.l_mle_null, 4.34046) < 0.001);
    enforce(modDiff(cPar.l_remle_null, 4.32887) < 0.001);
    enforce(modDiff(cPar.logl_mle_H0, -1584.7216) < 0.001);
    enforce(modDiff(cPar.logl_remle_H0, -1584.3408) < 0.001);
    //enforce(modDiff(cPar.beta, 0) < 0.001);
    //enforce(modDiff(cPar.se_beta, 0) < 0.001);
    //enforce(modDiff(cPar.pve_se_null, 0) < 0.001);
  }
}
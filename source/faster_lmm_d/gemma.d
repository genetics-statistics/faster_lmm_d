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
import faster_lmm_d.gemma_lmm;
import faster_lmm_d.gemma_param;
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


void batch_run(Param cPar){

  // Read Files.
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

  // Prediction for bslmm
  if (cPar.a_mode == 41 || cPar.a_mode == 42) {
    DMatrix y_prdt;

    y_prdt.shape = [1, cPar.ni_total - cPar.ni_test];

    // set to zero TODO
    //gsl_vector_set_zero(y_prdt);

    PRDT cPRDT;
    cPRDT.CopyFromParam(cPar);

    // add breeding value if needed
    if (!cPar.file_kin.empty() && !cPar.file_ebv.empty()) {
      writeln("Adding Breeding Values ... ");

      DMatrix G;
      G.shape = [cPar.ni_total, cPar.ni_total];
      DMatrix u_hat;
      u_hat.shape = [1, cPar.ni_test];

      // read kinship matrix and set u_hat
      int[] indicator_all;
      size_t c_bv = 0;
      for (size_t i = 0; i < cPar.indicator_idv.size(); i++) {
        //indicator_all.push_back(1);
        if (cPar.indicator_bv[i] == 1) {
          //gsl_vector_set(u_hat, c_bv, cPar.vec_bv[i]);
          c_bv++;
        }
      }

      ReadFile_kin(cPar.file_kin, indicator_all, cPar.mapID2num, cPar.k_mode,
                   cPar.error, G);
      if (cPar.error == true) {
        writeln("error! fail to read kinship/relatedness file.");
        return;
      }

      // read u
      cPRDT.AddBV(G, u_hat, y_prdt);

    }

    // add beta
    if (!cPar.file_bfile.empty()) {
      cPRDT.AnalyzePlink(y_prdt);
    } else {
      cPRDT.AnalyzeBimbam(y_prdt);
    }

    // add mu
    gsl_vector_add_constant(y_prdt, cPar.pheno_mean);

    // convert y to probability if needed
    if (cPar.a_mode == 42) {
      double d;
      for (size_t i = 0; i < y_prdt.elements.length; i++) {
        d = y_prdt.elements[i];
        d = gsl_cdf_gaussian_P(d, 1.0);
        gsl_vector_set(y_prdt, i, d);
      }
    }

    cPRDT.CopyToParam(cPar);

    cPRDT.WriteFiles(y_prdt);

    gsl_vector_free(y_prdt);
  }

  // Prediction with kinship matrix only; for one or more phenotypes
  if (cPar.a_mode == 43) {
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

    // set covariates matrix W and phenotype matrix Y
    // an intercept should be included in W,
    cPar.CopyCvtPhen(W, Y, 0);
    cPar.CopyCvtPhen(W_full, Y_full, 1);

    DMatrix Y_hat;
    Y_hat.shape = [Y_full.shape[0], cPar.n_ph];
    DMatrix G_full;
    G_full.shape = [Y_full.shape[0], Y_full.shape[0]];
    DMatrix H_full;
    H_full.shape = [Y_full.shape[0] * Y_hat.shape[1],
                                          Y_full.shape[0] * Y_hat.shape[1]];

    // read relatedness matrix G, and matrix G_full
    ReadFile_kin(cPar.file_kin, cPar.indicator_idv, cPar.mapID2num, cPar.k_mode,
                 cPar.error, G);
    if (cPar.error == true) {
      writeln("error! fail to read kinship/relatedness file.");
      return;
    }
    // This is not so elegant. Reads twice to select on idv and then cvt
    ReadFile_kin(cPar.file_kin, cPar.indicator_cvt, cPar.mapID2num, cPar.k_mode,
                 cPar.error, G_full);
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

      // obtain estimates
      CalcMvLmmVgVeBeta(eval, UtW, UtY, cPar.em_iter, cPar.nr_iter,
                        cPar.em_prec, cPar.nr_prec, cPar.l_min, cPar.l_max,
                        cPar.n_region, Vg, Ve, B, se_B);

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
          //cPar.Vg_remle_null.push_back(gsl_matrix_get(Vg, i, j));
          //cPar.Ve_remle_null.push_back(gsl_matrix_get(Ve, i, j));
        }
      }

      // obtain Y_hat from fixed effects
      gsl_blas_dgemm(CblasNoTrans, CblasTrans, 1.0, W_full, B, 0.0, Y_hat);

      // obtain H
      KroneckerSym(G_full, Vg, H_full);
      for (size_t i = 0; i < G_full.shape[0]; i++) {
        gsl_matrix_view H_sub = gsl_matrix_submatrix(
            H_full, i * Ve.shape[0], i * Ve.shape[1], Ve.shape[0], Ve.shape[1]);
        gsl_matrix_add(&H_sub.matrix, Ve);
      }

      // free matrices
    }

    PRDT cPRDT;

    cPRDT.CopyFromParam(cPar);

    writeln("Predicting Missing Phentypes ... ");
    cPRDT.MvnormPrdt(Y_hat, H_full, Y_full);

    cPRDT.WriteFiles(Y_full);

  }

  // Generate Kinship matrix (optionally using LOCO)
  if (cPar.a_mode == 21 || cPar.a_mode == 22) {
    cout << "Calculating Relatedness Matrix ... " << endl;

    DMatrix G;
    G.shape = [cPar.ni_total, cPar.ni_total];
    log(G, "allocate G"); // just to be sure

    cPar.CalcKin(G);

    if (cPar.error == true) {
      cout << "error! fail to calculate relatedness matrix. " << endl;
      return;
    }

    // Now we have the Kinship matrix test it
    validate_K(G,cPar.mode_check,cPar.mode_strict);

    if (cPar.a_mode == 21) {
      cPar.WriteMatrix(G, "cXX");
    } else {
      cPar.WriteMatrix(G, "sXX");
    }
  }

  // Compute the LDSC weights (not implemented yet)
  if (cPar.a_mode == 72) {
    cout << "Calculating Weights ... " << endl;

    VARCOV cVarcov;
    cVarcov.CopyFromParam(cPar);

    if (!cPar.file_bfile.empty()) {
      cVarcov.AnalyzePlink();
    } else {
      cVarcov.AnalyzeBimbam();
    }

    cVarcov.CopyToParam(cPar);
  }

  // Compute the S matrix (and its variance), that is used for
  // variance component estimation using summary statistics.
  if (cPar.a_mode == 25 || cPar.a_mode == 26) {
    cout << "Calculating the S Matrix ... " << endl;

    DMatrix S;
    S.shape = [cPar.n_vc * 2, cPar.n_vc];
    DMatrix ns;
    ns.shape = [cPar.n_vc + 1];
    //gsl_matrix_set_zero(S);
    //gsl_vector_set_zero(ns);

    DMatrix S_mat;// = gsl_matrix_submatrix(S, 0, 0, cPar.n_vc, cPar.n_vc);
    DMatrix Svar_mat;// =
        //gsl_matrix_submatrix(S, cPar.n_vc, 0, cPar.n_vc, cPar.n_vc);
    DMatrix ns_vec; //= gsl_vector_subvector(ns, 0, cPar.n_vc);

    DMatrix K;
    K.shape = [cPar.ni_test, cPar.n_vc * cPar.ni_test];
    DMatrix A;
    A.shape = [cPar.ni_test, cPar.n_vc * cPar.ni_test];
    gsl_matrix_set_zero(K);
    gsl_matrix_set_zero(A);

    DMatrix y;
    y.shape = [1, cPar.ni_test];
    DMatrix W;
    W.shape = [cPar.ni_test, cPar.n_cvt];

    cPar.CopyCvtPhen(W, y, 0);

    //set<string> setSnps_beta;
    //map<string, double> mapRS2wA, mapRS2wK;

    cPar.ObtainWeight(setSnps_beta, mapRS2wK);

    cPar.CalcS(mapRS2wA, mapRS2wK, W, A, K, &S_mat.matrix, &Svar_mat.matrix,
               &ns_vec.vector);
    if (cPar.error == true) {
      writeln("error! fail to calculate the S matrix. ");
      return;
    }

    gsl_vector_set(ns, cPar.n_vc, cPar.ni_test);

    cPar.WriteMatrix(S, "S");
    cPar.WriteVector(ns, "size");
    cPar.WriteVar("snps");

  }

  // Compute the q vector, that is used for variance component estimation using
  // summary statistics
  if (cPar.a_mode == 27 || cPar.a_mode == 28) {
    DMatrix Vq;
    Vq.shape = [cPar.n_vc, cPar.n_vc];
    DMatrix q;
    q.shape = [1, cPar.n_vc];
    DMatrix s;
    s.shape = [1, cPar.n_vc + 1];
    //gsl_vector_set_zero(q);
    //gsl_vector_set_zero(s);

    gsl_vector_view s_vec = gsl_vector_subvector(s, 0, cPar.n_vc);

    //vector<size_t> vec_cat, vec_ni;
    //vector<double> vec_weight, vec_z2;
    //map<string, double> mapRS2weight;
    mapRS2weight.clear();

    ReadFile_beta(cPar.file_beta, cPar.mapRS2cat, mapRS2weight, vec_cat, vec_ni,
                  vec_weight, vec_z2, cPar.ni_total, cPar.ns_total,
                  cPar.ns_test);
    writeln("## number of total individuals = ", cPar.ni_total);
    writeln("## number of total SNPs = ", cPar.ns_total);
    writeln("## number of analyzed SNPs = ", cPar.ns_test);
    writeln("## number of variance components = ", cPar.n_vc);
    writeln("Calculating the q vector ... ");
    Calcq(cPar.n_block, vec_cat, vec_ni, vec_weight, vec_z2, Vq, q,
          &s_vec.vector);

    if (cPar.error == true) {
      cout << "error! fail to calculate the q vector. " << endl;
      return;
    }

    gsl_vector_set(s, cPar.n_vc, cPar.ni_total);

    cPar.WriteMatrix(Vq, "Vq");
    cPar.WriteVector(q, "q");
    cPar.WriteVector(s, "size");
  }

  // Calculate SNP covariance.
  if (cPar.a_mode == 71) {
    VARCOV cVarcov;
    cVarcov.CopyFromParam(cPar);

    if (!cPar.file_bfile.empty()) {
      cVarcov.AnalyzePlink();
    } else {
      cVarcov.AnalyzeBimbam();
    }

    cVarcov.CopyToParam(cPar);
  }

  // LM.
  if (cPar.a_mode == 51 || cPar.a_mode == 52 || cPar.a_mode == 53 ||
      cPar.a_mode == 54) { // Fit LM
    DMatrix Y;
    Y = [cPar.ni_test, cPar.n_ph];
    DMatrix W;
    W = [Y.shape[0], cPar.n_cvt];

    // set covariates matrix W and phenotype matrix Y
    // an intercept should be included in W,
    cPar.CopyCvtPhen(W, Y, 0);

    // Fit LM or mvLM
    if (cPar.n_ph == 1) {
      LM cLm;
      cLm.CopyFromParam(cPar);

      gsl_vector_view Y_col = gsl_matrix_column(Y, 0);

      if (!cPar.file_gene.empty()) {
        cLm.AnalyzeGene(W,
                        &Y_col.vector); // y is the predictor, not the phenotype
      } else if (!cPar.file_bfile.empty()) {
        cLm.AnalyzePlink(W, &Y_col.vector);
      } else if (!cPar.file_oxford.empty()) {
        cLm.Analyzebgen(W, &Y_col.vector);
      } else {
        cLm.AnalyzeBimbam(W, &Y_col.vector);
      }

      cLm.WriteFiles();
      cLm.CopyToParam(cPar);
    }
    // release all matrices and vectors
  }

  // VC estimation with one or multiple kinship matrices
  // REML approach only
  // if file_kin or file_ku/kd is provided, then a_mode is changed to 5 already,
  // in param.cpp
  // for one phenotype only;
  if (cPar.a_mode == 61 || cPar.a_mode == 62 || cPar.a_mode == 63) {
    if (!cPar.file_beta.empty()) {
      // need to obtain a common set of SNPs between beta file and the genotype
      // file; these are saved in mapRS2wA and mapRS2wK
      // normalize the weight in mapRS2wK to have an average of one; each
      // element of mapRS2wA is 1
      // update indicator_snps, so that the numbers are in accordance with
      // mapRS2wK

      //set<string> setSnps_beta;
      ReadFile_snps_header(cPar.file_beta, setSnps_beta);

      //map<string, double> mapRS2wA, mapRS2wK;
      cPar.ObtainWeight(setSnps_beta, mapRS2wK);

      cPar.UpdateSNP(mapRS2wK);

      // Setup matrices and vectors.
      DMatrix S;
      S.shape = [cPar.n_vc * 2, cPar.n_vc];
      DMatrix Vq;
      Vq.shape = [cPar.n_vc, cPar.n_vc];
      DMatrix q;
      q.shape = [1, cPar.n_vc];
      DMatrix s;
      s.shape = [1, cPar.n_vc + 1];

      DMatrix K;
      K.shape = [cPar.ni_test, cPar.n_vc * cPar.ni_test];
      DMatrix A;
      A.shape = [cPar.ni_test, cPar.n_vc * cPar.ni_test];

      DMatrix y;
      y.shape = [1, cPar.ni_test];
      DMatrix W;
      W.shape = [cPar.ni_test, cPar.n_cvt];

      gsl_matrix_set_zero(K);
      gsl_matrix_set_zero(A);

      gsl_matrix_set_zero(S);
      gsl_matrix_set_zero(Vq);
      gsl_vector_set_zero(q);
      gsl_vector_set_zero(s);

      cPar.CopyCvtPhen(W, y, 0);

      gsl_matrix_view S_mat =
          gsl_matrix_submatrix(S, 0, 0, cPar.n_vc, cPar.n_vc);
      gsl_matrix_view Svar_mat =
          gsl_matrix_submatrix(S, cPar.n_vc, 0, cPar.n_vc, cPar.n_vc);
      gsl_vector_view s_vec = gsl_vector_subvector(s, 0, cPar.n_vc);

      size_t[] vec_cat, vec_ni;
      double[] vec_weight, vec_z2;

      // read beta, based on the mapRS2wK
      ReadFile_beta(cPar.file_beta, cPar.mapRS2cat, mapRS2wK, vec_cat, vec_ni,
                    vec_weight, vec_z2, cPar.ni_study, cPar.ns_study,
                    cPar.ns_test);

      writeln("Study Panel: ");
      writeln("## number of total individuals = ", cPar.ni_study);
      writeln("## number of total SNPs = ", cPar.ns_study);
      writeln("## number of analyzed SNPs = ", cPar.ns_test);
      writeln("## number of variance components = ", cPar.n_vc);

      // compute q
      Calcq(cPar.n_block, vec_cat, vec_ni, vec_weight, vec_z2, Vq, q,
            &s_vec.vector);

      // compute S
      cPar.CalcS(mapRS2wA, mapRS2wK, W, A, K, &S_mat.matrix, &Svar_mat.matrix,
                 &s_vec.vector);
      cPar.time_G += (clock() - time_start) / (double(CLOCKS_PER_SEC) * 60.0);
      if (cPar.error == true) {
        cout << "error! fail to calculate the S matrix. " << endl;
        return;
      }

      // compute vc estimates
      CalcVCss(Vq, &S_mat.matrix, &Svar_mat.matrix, q, &s_vec.vector,
               cPar.ni_study, cPar.v_pve, cPar.v_se_pve, cPar.pve_total,
               cPar.se_pve_total, cPar.v_sigma2, cPar.v_se_sigma2,
               cPar.v_enrich, cPar.v_se_enrich);

      assert(!has_nan(cPar.v_se_pve));

      // if LDSC weights, then compute the weights and run the above steps again
      if (cPar.a_mode == 62) {
        // compute the weights and normalize the weights for A
        cPar.UpdateWeight(1, mapRS2wK, cPar.ni_study, &s_vec.vector, mapRS2wA);

        // read beta file again, and update weigths vector
        ReadFile_beta(cPar.file_beta, cPar.mapRS2cat, mapRS2wA, vec_cat, vec_ni,
                      vec_weight, vec_z2, cPar.ni_study, cPar.ns_total,
                      cPar.ns_test);

        // compute q
        Calcq(cPar.n_block, vec_cat, vec_ni, vec_weight, vec_z2, Vq, q,
              &s_vec.vector);

        // compute S
        cPar.CalcS(mapRS2wA, mapRS2wK, W, A, K, &S_mat.matrix, &Svar_mat.matrix,
                   &s_vec.vector);
        cPar.time_G += (clock() - time_start) / (double(CLOCKS_PER_SEC) * 60.0);
        if (cPar.error == true) {
          writeln("error! fail to calculate the S matrix.");
          return;
        }

        // compute vc estimates
        CalcVCss(Vq, &S_mat.matrix, &Svar_mat.matrix, q, &s_vec.vector,
                 cPar.ni_study, cPar.v_pve, cPar.v_se_pve, cPar.pve_total,
                 cPar.se_pve_total, cPar.v_sigma2, cPar.v_se_sigma2,
                 cPar.v_enrich, cPar.v_se_enrich);
        assert(!has_nan(cPar.v_se_pve));
      }


      gsl_vector_set(s, cPar.n_vc, cPar.ni_test);

      cPar.WriteMatrix(S, "S");
      cPar.WriteMatrix(Vq, "Vq");
      cPar.WriteVector(q, "q");
      cPar.WriteVector(s, "size");

    } else if (!cPar.file_study.empty() || !cPar.file_mstudy.empty()) {
      if (!cPar.file_study.empty()) {
        string sfile = cPar.file_study + ".size.txt";
        CountFileLines(sfile, cPar.n_vc);
      } else {
        string file_name;
        //igzstream infile(cPar.file_mstudy.c_str(), igzstream::in);
        if (!infile) {
          writeln("error! fail to open mstudy file:", cPar.file_study);
          return;
        }

        safeGetline(infile, file_name);

        infile.clear();
        infile.close();

        string sfile = file_name + ".size.txt";
        CountFileLines(sfile, cPar.n_vc);
      }

      cPar.n_vc = cPar.n_vc - 1;

      gsl_matrix *S = gsl_matrix_alloc(2 * cPar.n_vc, cPar.n_vc);
      gsl_matrix *Vq = gsl_matrix_alloc(cPar.n_vc, cPar.n_vc);
      // gsl_matrix *V=gsl_matrix_alloc (cPar.n_vc+1,
      // (cPar.n_vc*(cPar.n_vc+1))/2*(cPar.n_vc+1) );
      // gsl_matrix *Vslope=gsl_matrix_alloc (n_lines+1,
      // (n_lines*(n_lines+1))/2*(n_lines+1) );
      gsl_vector *q = gsl_vector_alloc(cPar.n_vc);
      gsl_vector *s_study = gsl_vector_alloc(cPar.n_vc);
      gsl_vector *s_ref = gsl_vector_alloc(cPar.n_vc);
      gsl_vector *s = gsl_vector_alloc(cPar.n_vc + 1);

      gsl_matrix_set_zero(S);
      gsl_matrix_view S_mat =
          gsl_matrix_submatrix(S, 0, 0, cPar.n_vc, cPar.n_vc);
      gsl_matrix_view Svar_mat =
          gsl_matrix_submatrix(S, cPar.n_vc, 0, cPar.n_vc, cPar.n_vc);

      gsl_matrix_set_zero(Vq);
      // gsl_matrix_set_zero(V);
      // gsl_matrix_set_zero(Vslope);
      gsl_vector_set_zero(q);
      gsl_vector_set_zero(s_study);
      gsl_vector_set_zero(s_ref);

      if (!cPar.file_study.empty()) {
        ReadFile_study(cPar.file_study, Vq, q, s_study, cPar.ni_study);
      } else {
        ReadFile_mstudy(cPar.file_mstudy, Vq, q, s_study, cPar.ni_study);
      }

      if (!cPar.file_ref.empty()) {
        ReadFile_ref(cPar.file_ref, &S_mat.matrix, &Svar_mat.matrix, s_ref,
                     cPar.ni_ref);
      } else {
        ReadFile_mref(cPar.file_mref, &S_mat.matrix, &Svar_mat.matrix, s_ref,
                      cPar.ni_ref);
      }

      writeln("## number of variance components = ", cPar.n_vc);
      writeln("## number of individuals in the sample = ", cPar.ni_study);
      writeln("## number of individuals in the reference = ", cPar.ni_ref);

      CalcVCss(Vq, &S_mat.matrix, &Svar_mat.matrix, q, s_study, cPar.ni_study,
               cPar.v_pve, cPar.v_se_pve, cPar.pve_total, cPar.se_pve_total,
               cPar.v_sigma2, cPar.v_se_sigma2, cPar.v_enrich,
               cPar.v_se_enrich);
      assert(!has_nan(cPar.v_se_pve));

      gsl_vector_view s_sub = gsl_vector_subvector(s, 0, cPar.n_vc);
      gsl_vector_memcpy(&s_sub.vector, s_ref);
      gsl_vector_set(s, cPar.n_vc, cPar.ni_ref);

      cPar.WriteMatrix(S, "S");
      cPar.WriteMatrix(Vq, "Vq");
      cPar.WriteVector(q, "q");
      cPar.WriteVector(s, "size");

    } else {
      DMatrix Y;
      Y.shape = [cPar.ni_test, cPar.n_ph];
      DMatrix W;
      W.shape = [Y.shape[0], cPar.n_cvt];
      DMatrix G;
      G.shape = [Y.shape[0], Y.shape[0] * cPar.n_vc];

      // set covariates matrix W and phenotype matrix Y
      // an intercept should be included in W,
      cPar.CopyCvtPhen(W, Y, 0);

      // read kinship matrices
      if (!(cPar.file_mk).empty()) {
        ReadFile_mk(cPar.file_mk, cPar.indicator_idv, cPar.mapID2num,
                    cPar.k_mode, cPar.error, G);
        if (cPar.error == true) {
          cout << "error! fail to read kinship/relatedness file. " << endl;
          return;
        }

        // center matrix G, and obtain v_traceG
        double d = 0;
        (cPar.v_traceG).clear();
        for (size_t i = 0; i < cPar.n_vc; i++) {
          gsl_matrix_view G_sub =
              gsl_matrix_submatrix(G, 0, i * G.shape[0], G.shape[0], G.shape[0]);
          CenterMatrix(&G_sub.matrix);
          d = 0;
          for (size_t j = 0; j < G.shape[0]; j++) {
            d += accessor(G_sub.matrix, j, j);
          }
          d /= to!double(G.shape[0]);
          (cPar.v_traceG).push_back(d);
        }
      } else if (!(cPar.file_kin).empty()) {
        ReadFile_kin(cPar.file_kin, cPar.indicator_idv, cPar.mapID2num,
                     cPar.k_mode, cPar.error, G);
        if (cPar.error == true) {
          cout << "error! fail to read kinship/relatedness file. " << endl;
          return;
        }

        // center matrix G
        CenterMatrix(G);
        validate_K(G,cPar.mode_check,cPar.mode_strict);

        (cPar.v_traceG).clear();
        double d = 0;
        for (size_t j = 0; j < G.shape[0]; j++) {
          d += gsl_matrix_get(G, j, j);
        }
        d /= to!double(G.shape[0]);
        (cPar.v_traceG).push_back(d);
      }
      // fit multiple variance components
      if (cPar.n_ph == 1) {
        //      if (cPar.n_vc==1) {
        //      } else {
        gsl_vector_view Y_col = gsl_matrix_column(Y, 0);
        VC cVc;
        cVc.CopyFromParam(cPar);
        if (cPar.a_mode == 61) {
          cVc.CalcVChe(G, W, &Y_col.vector);
        } else if (cPar.a_mode == 62) {
          cVc.CalcVCreml(cPar.noconstrain, G, W, &Y_col.vector);
        } else {
          cVc.CalcVCacl(G, W, &Y_col.vector);
        }
        cVc.CopyToParam(cPar);
        // obtain pve from sigma2
        // obtain se_pve from se_sigma2

        //}
      }
    }
  }

  // compute confidence intervals with additional summary statistics
  // we do not check the sign of z-scores here, but they have to be matched with
  // the genotypes
  if (cPar.a_mode == 66 || cPar.a_mode == 67) {
    // read reference file first
    gsl_matrix *S = gsl_matrix_alloc(cPar.n_vc, cPar.n_vc);
    gsl_matrix *Svar = gsl_matrix_alloc(cPar.n_vc, cPar.n_vc);
    gsl_vector *s_ref = gsl_vector_alloc(cPar.n_vc);

    gsl_matrix_set_zero(S);
    gsl_matrix_set_zero(Svar);
    gsl_vector_set_zero(s_ref);

    if (!cPar.file_ref.empty()) {
      ReadFile_ref(cPar.file_ref, S, Svar, s_ref, cPar.ni_ref);
    } else {
      ReadFile_mref(cPar.file_mref, S, Svar, s_ref, cPar.ni_ref);
    }

    // need to obtain a common set of SNPs between beta file and the genotype
    // file; these are saved in mapRS2wA and mapRS2wK
    // normalize the weight in mapRS2wK to have an average of one; each element
    // of mapRS2wA is 1


    //set<string> setSnps_beta;
    ReadFile_snps_header(cPar.file_beta, setSnps_beta);

    // obtain the weights for wA, which contains the SNP weights for SNPs used
    // in the model

    //map<string, double> mapRS2wK;
    cPar.ObtainWeight(setSnps_beta, mapRS2wK);

    // set up matrices and vector
    gsl_matrix *Xz = gsl_matrix_alloc(cPar.ni_test, cPar.n_vc);
    gsl_matrix *XWz = gsl_matrix_alloc(cPar.ni_test, cPar.n_vc);
    gsl_matrix *XtXWz =
        gsl_matrix_alloc(mapRS2wK.size(), cPar.n_vc * cPar.n_vc);
    gsl_vector *w = gsl_vector_alloc(mapRS2wK.size());
    gsl_vector *w1 = gsl_vector_alloc(mapRS2wK.size());
    gsl_vector *z = gsl_vector_alloc(mapRS2wK.size());
    gsl_vector *s_vec = gsl_vector_alloc(cPar.n_vc);

    size_t[] vec_cat, vec_size;
    double[] vec_z;

    //map<string, double> mapRS2z, mapRS2wA;
    //map<string, string> mapRS2A1;
    string file_str;

    // update s_vec, the number of snps in each category
    for (size_t i = 0; i < cPar.n_vc; i++) {
      vec_size.push_back(0);
    }

    //TODO
    //for (map<string, double>::const_iterator it = mapRS2wK.begin();
    //     it != mapRS2wK.end(); ++it) {
    //  vec_size[cPar.mapRS2cat[it->first]]++;
    //}

    for (size_t i = 0; i < cPar.n_vc; i++) {
      //gsl_vector_set(s_vec, i, vec_size[i]);
      s_vec[i] = vec_size[i];
    }

    // update mapRS2wA using v_pve and s_vec
    if (cPar.a_mode == 66) {
      //todo
      //for (map<string, double>::const_iterator it = mapRS2wK.begin();
      //     it != mapRS2wK.end(); ++it) {
      //  mapRS2wA[it->first] = 1;
      //}
    } else {
      cPar.UpdateWeight(0, mapRS2wK, cPar.ni_test, s_vec, mapRS2wA);
    }

    // read in z-scores based on allele 0, and save that into a vector
    ReadFile_beta(cPar.file_beta, mapRS2wA, mapRS2A1, mapRS2z);

    // update snp indicator, save weights to w, save z-scores to vec_z, save
    // category label to vec_cat
    // sign of z is determined by matching alleles
    cPar.UpdateSNPnZ(mapRS2wA, mapRS2A1, mapRS2z, w, z, vec_cat);

    // compute an n by k matrix of X_iWz
    cout << "Calculating Xz ... " << endl;

    gsl_matrix_set_zero(Xz);
    gsl_vector_set_all(w1, 1);

    if (!cPar.file_bfile.empty()) {
      file_str = cPar.file_bfile + ".bed";
      PlinkXwz(file_str, cPar.d_pace, cPar.indicator_idv, cPar.indicator_snp,
               vec_cat, w1, z, 0, Xz);
    } else if (!cPar.file_geno.empty()) {
      BimbamXwz(cPar.file_geno, cPar.d_pace, cPar.indicator_idv,
                cPar.indicator_snp, vec_cat, w1, z, 0, Xz);
    } else if (!cPar.file_mbfile.empty()) {
      MFILEXwz(1, cPar.file_mbfile, cPar.d_pace, cPar.indicator_idv,
               cPar.mindicator_snp, vec_cat, w1, z, Xz);
    } else if (!cPar.file_mgeno.empty()) {
      MFILEXwz(0, cPar.file_mgeno, cPar.d_pace, cPar.indicator_idv,
               cPar.mindicator_snp, vec_cat, w1, z, Xz);
    }
    if (cPar.a_mode == 66) {
      gsl_matrix_memcpy(XWz, Xz);
    } else if (cPar.a_mode == 67) {
      cout << "Calculating XWz ... " << endl;

      gsl_matrix_set_zero(XWz);

      if (!cPar.file_bfile.empty()) {
        file_str = cPar.file_bfile + ".bed";
        PlinkXwz(file_str, cPar.d_pace, cPar.indicator_idv, cPar.indicator_snp,
                 vec_cat, w, z, 0, XWz);
      } else if (!cPar.file_geno.empty()) {
        BimbamXwz(cPar.file_geno, cPar.d_pace, cPar.indicator_idv,
                  cPar.indicator_snp, vec_cat, w, z, 0, XWz);
      } else if (!cPar.file_mbfile.empty()) {
        MFILEXwz(1, cPar.file_mbfile, cPar.d_pace, cPar.indicator_idv,
                 cPar.mindicator_snp, vec_cat, w, z, XWz);
      } else if (!cPar.file_mgeno.empty()) {
        MFILEXwz(0, cPar.file_mgeno, cPar.d_pace, cPar.indicator_idv,
                 cPar.mindicator_snp, vec_cat, w, z, XWz);
      }
    }
    // compute an p by k matrix of X_j^TWX_iWz
    cout << "Calculating XtXWz ... " << endl;
    gsl_matrix_set_zero(XtXWz);

    if (!cPar.file_bfile.empty()) {
      file_str = cPar.file_bfile + ".bed";
      PlinkXtXwz(file_str, cPar.d_pace, cPar.indicator_idv, cPar.indicator_snp,
                 XWz, 0, XtXWz);
    } else if (!cPar.file_geno.empty()) {
      BimbamXtXwz(cPar.file_geno, cPar.d_pace, cPar.indicator_idv,
                  cPar.indicator_snp, XWz, 0, XtXWz);
    } else if (!cPar.file_mbfile.empty()) {
      MFILEXtXwz(1, cPar.file_mbfile, cPar.d_pace, cPar.indicator_idv,
                 cPar.mindicator_snp, XWz, XtXWz);
    } else if (!cPar.file_mgeno.empty()) {
      MFILEXtXwz(0, cPar.file_mgeno, cPar.d_pace, cPar.indicator_idv,
                 cPar.mindicator_snp, XWz, XtXWz);
    }
    // compute confidence intervals
    CalcCIss(Xz, XWz, XtXWz, S, Svar, w, z, s_vec, vec_cat, cPar.v_pve,
             cPar.v_se_pve, cPar.pve_total, cPar.se_pve_total, cPar.v_sigma2,
             cPar.v_se_sigma2, cPar.v_enrich, cPar.v_se_enrich);
    assert(!has_nan(cPar.v_se_pve));

  }

  // LMM or mvLMM or Eigen-Decomposition
  if (cPar.a_mode == 1 || cPar.a_mode == 2 || cPar.a_mode == 3 ||
      cPar.a_mode == 4 || cPar.a_mode == 5 ||
      cPar.a_mode == 31) { // Fit LMM or mvLMM or eigen
    DMatrix Y;
    Y.shape = [cPar.ni_test, cPar.n_ph];
    enforce_msg(Y, "allocate Y"); // just to be sure
    DMatrix W;
    W.shape = [Y.shape[0], cPar.n_cvt];
    DMatrix B;
    B.shape = [Y.shape[1], W.shape[1]]; // B is a d by c
                                                          // matrix
    DMatrix se_B;
    se_B.shape = [Y.shape[1], W.shape[1]];
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
    DMatrix env;
    env.shape = [1, Y.shape[0]];
    DMatrix weight;
    weight.shape = [1, Y.shape[0]];
    assert_issue(cPar.issue == 26, UtY.data[0] == 0.0);

    // set covariates matrix W and phenotype matrix Y
    // an intercept should be included in W,
    cPar.CopyCvtPhen(W, Y, 0);
    if (!cPar.file_gxe.empty()) {
      cPar.CopyGxe(env);
    }

    // read relatedness matrix G
    if (!(cPar.file_kin).empty()) {
      ReadFile_kin(cPar.file_kin, cPar.indicator_idv, cPar.mapID2num,
                   cPar.k_mode, cPar.error, G);
      if (cPar.error == true) {
        cout << "error! fail to read kinship/relatedness file. " << endl;
        return;
      }

      // center matrix G
      CenterMatrix(G);
      validate_K(G,cPar.mode_check,cPar.mode_strict);

      // is residual weights are provided, then
      if (!cPar.file_weight.empty()) {
        cPar.CopyWeight(weight);
        double d, wi, wj;
        for (size_t i = 0; i < G.shape[0]; i++) {
          wi = gsl_vector_get(weight, i);
          for (size_t j = i; j < G.shape[1]; j++) {
            wj = gsl_vector_get(weight, j);
            d = gsl_matrix_get(G, i, j);
            if (wi <= 0 || wj <= 0) {
              d = 0;
            } else {
              d /= sqrt(wi * wj);
            }
            gsl_matrix_set(G, i, j, d);
            if (j != i) {
              gsl_matrix_set(G, j, i, d);
            }
          }
        }
      }

      // eigen-decomposition and calculate trace_G
      writeln("Start Eigen-Decomposition...");

      if (cPar.a_mode == 31) {
        cPar.trace_G = EigenDecomp_Zeroed(G, U, eval, 1);
      } else {
        cPar.trace_G = EigenDecomp_Zeroed(G, U, eval, 0);
      }

      if (!cPar.file_weight.empty()) {
        double wi;
        for (size_t i = 0; i < U.shape[0]; i++) {
          wi = gsl_vector_get(weight, i);
          if (wi <= 0) {
            wi = 0;
          } else {
            wi = sqrt(wi);
          }
          gsl_vector_view Urow = gsl_matrix_row(U, i);
          gsl_vector_scale(&Urow.vector, wi);
        }
      }

    } else {
      ReadFile_eigenU(cPar.file_ku, cPar.error, U);
      if (cPar.error == true) {
        writeln("error! fail to read the U file. ");
        return;
      }

      ReadFile_eigenD(cPar.file_kd, cPar.error, eval);
      if (cPar.error == true) {
        writeln("error! fail to read the D file. ");
        return;
      }

      cPar.trace_G = 0.0;
      for (size_t i = 0; i < eval.elements.length; i++) {
        if (gsl_vector_get(eval, i) < 1e-10) {
          gsl_vector_set(eval, i, 0);
        }
        cPar.trace_G += gsl_vector_get(eval, i);
      }
      cPar.trace_G /= to!double(eval.elements.length);
    }

    if (cPar.a_mode == 31) {
      cPar.WriteMatrix(U, "eigenU");
      cPar.WriteVector(eval, "eigenD");
    } else if (!cPar.file_gene.empty()) {
      // calculate UtW and Uty
      CalcUtX(U, W, UtW);
      CalcUtX(U, Y, UtY);

      assert_issue(cPar.issue == 26, ROUND(UtY.data[0]) == -16.6143);

      LMM cLmm;
      cLmm.CopyFromParam(cPar);

      gsl_vector_view Y_col = gsl_matrix_column(Y, 0);
      gsl_vector_view UtY_col = gsl_matrix_column(UtY, 0);

      cLmm.AnalyzeGene(U, eval, UtW, &UtY_col.vector, W,
                       &Y_col.vector); // y is the predictor, not the phenotype

      cLmm.WriteFiles();
      cLmm.CopyToParam(cPar);
    } else {
      // calculate UtW and Uty
      CalcUtX(U, W, UtW);
      CalcUtX(U, Y, UtY);
      assert_issue(cPar.issue == 26, ROUND(UtY.data[0]) == -16.6143);

      // calculate REMLE/MLE estimate and pve for univariate model
      if (cPar.n_ph == 1) { // one phenotype
        gsl_vector_view beta = gsl_matrix_row(B, 0);
        gsl_vector_view se_beta = gsl_matrix_row(se_B, 0);
        gsl_vector_view UtY_col = gsl_matrix_column(UtY, 0);

        assert_issue(cPar.issue == 26, ROUND(UtY.data[0]) == -16.6143);

        CalcLambda('L', eval, UtW, &UtY_col.vector, cPar.l_min, cPar.l_max,
                   cPar.n_region, cPar.l_mle_null, cPar.logl_mle_H0);
        //assert(!std::isnan(UtY.data[0]));
        //assert(!std::isnan(B.data[0]));
        //assert(!std::isnan(se_B.data[0]));

        CalcLmmVgVeBeta(eval, UtW, &UtY_col.vector, cPar.l_mle_null,
                        cPar.vg_mle_null, cPar.ve_mle_null, &beta.vector,
                        &se_beta.vector);

        //assert(!std::isnan(UtY.data[0]));
        //assert(!std::isnan(B.data[0]));
        //assert(!std::isnan(se_B.data[0]));

        cPar.beta_mle_null.clear();
        cPar.se_beta_mle_null.clear();
        for (size_t i = 0; i < B.shape[1]; i++) {
          cPar.beta_mle_null.push_back(gsl_matrix_get(B, 0, i));
          cPar.se_beta_mle_null.push_back(gsl_matrix_get(se_B, 0, i));
        }
        //assert(!std::isnan(UtY.data[0]));
        //assert(!std::isnan(B.data[0]));
        //assert(!std::isnan(se_B.data[0]));
        //assert(!std::isnan(cPar.beta_mle_null.front()));
        //assert(!std::isnan(cPar.se_beta_mle_null.front()));

        CalcLambda('R', eval, UtW, &UtY_col.vector, cPar.l_min, cPar.l_max,
                   cPar.n_region, cPar.l_remle_null, cPar.logl_remle_H0);
        CalcLmmVgVeBeta(eval, UtW, &UtY_col.vector, cPar.l_remle_null,
                        cPar.vg_remle_null, cPar.ve_remle_null, &beta.vector,
                        &se_beta.vector);

        cPar.beta_remle_null.clear();
        cPar.se_beta_remle_null.clear();
        for (size_t i = 0; i < B.shape[1]; i++) {
          cPar.beta_remle_null.push_back(gsl_matrix_get(B, 0, i));
          cPar.se_beta_remle_null.push_back(gsl_matrix_get(se_B, 0, i));
        }

        CalcPve(eval, UtW, &UtY_col.vector, cPar.l_remle_null, cPar.trace_G,
                cPar.pve_null, cPar.pve_se_null);
        cPar.PrintSummary();

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
          gsl_vector_memcpy(y_hat, &UtY_col.vector);
          gsl_blas_dgemv(CblasNoTrans, -1.0, UtW, &beta.vector, 1.0, y_hat);

          double d, u, e;
          for (size_t i = 0; i < eval.elements.length; i++) {
            d = eval.elements[i];
            u = cPar.l_remle_null * d / (cPar.l_remle_null * d + 1.0) *
                gsl_vector_get(y_hat, i);
            e = 1.0 / (cPar.l_remle_null * d + 1.0) * y_hat.elements[i];
            gsl_vector_set(Utu_hat, i, u);
            gsl_vector_set(Ute_hat, i, e);
          }

          // obtain u and e
          gsl_blas_dgemv(CblasNoTrans, 1.0, U, Utu_hat, 0.0, u_hat);
          gsl_blas_dgemv(CblasNoTrans, 1.0, U, Ute_hat, 0.0, e_hat);

          // output residuals
          cPar.WriteVector(u_hat, "residU");
          cPar.WriteVector(e_hat, "residE");
        }
      }

      // Fit LMM or mvLMM (w. LOCO)
      if (cPar.a_mode == 1 || cPar.a_mode == 2 || cPar.a_mode == 3 ||
          cPar.a_mode == 4) {
        if (cPar.n_ph == 1) {
          LMM cLmm;
          cLmm.CopyFromParam(cPar);

          gsl_vector_view Y_col = gsl_matrix_column(Y, 0);
          gsl_vector_view UtY_col = gsl_matrix_column(UtY, 0);

          if (!cPar.file_bfile.empty()) {
            if (cPar.file_gxe.empty()) {
              cLmm.AnalyzePlink(U, eval, UtW, &UtY_col.vector, W,
                                &Y_col.vector);
            } else {
              cLmm.AnalyzePlinkGXE(U, eval, UtW, &UtY_col.vector, W,
                                   &Y_col.vector, env);
            }
          }
          // WJA added
          else if (!cPar.file_oxford.empty()) {
            cLmm.Analyzebgen(U, eval, UtW, &UtY_col.vector, W, &Y_col.vector);
          } else {
            if (cPar.file_gxe.empty()) {
              cLmm.AnalyzeBimbam(U, eval, UtW, &UtY_col.vector, W,
                                 &Y_col.vector, cPar.setGWASnps);
            } else {
              cLmm.AnalyzeBimbamGXE(U, eval, UtW, &UtY_col.vector, W,
                                    &Y_col.vector, env);
            }
          }

          cLmm.WriteFiles();
          cLmm.CopyToParam(cPar);
        } else {
          MVLMM cMvlmm;
          cMvlmm.CopyFromParam(cPar);

          if (!cPar.file_bfile.empty()) {
            if (cPar.file_gxe.empty()) {
              cMvlmm.AnalyzePlink(U, eval, UtW, UtY);
            } else {
              cMvlmm.AnalyzePlinkGXE(U, eval, UtW, UtY, env);
            }
          } else if (!cPar.file_oxford.empty()) {
            cMvlmm.Analyzebgen(U, eval, UtW, UtY);
          } else {
            if (cPar.file_gxe.empty()) {
              cMvlmm.AnalyzeBimbam(U, eval, UtW, UtY);
            } else {
              cMvlmm.AnalyzeBimbamGXE(U, eval, UtW, UtY, env);
            }
          }

          cMvlmm.WriteFiles();
          cMvlmm.CopyToParam(cPar);
        }
      }
    }

    // release all matrices and vectors
  }

  // BSLMM
  if (cPar.a_mode == 11 || cPar.a_mode == 12 || cPar.a_mode == 13) {
    DMatrix y;
    y.shape = [1, cPar.ni_test];
    DMatrix W;
    W.shape = [y.elements.length, cPar.n_cvt];
    DMatrix G;
    G.shape = [y.elements.length, y.elements.length];
    DMatrix UtX;
    UtX.shape = [y.elements.length, cPar.ns_test];

    // set covariates matrix W and phenotype vector y
    // an intercept should be included in W,
    cPar.CopyCvtPhen(W, y, 0);

    // center y, even for case/control data
    cPar.pheno_mean = CenterVector(y);

    // run bvsr if rho==1
    if (cPar.rho_min == 1 && cPar.rho_max == 1) {
      // read genotypes X (not UtX)
      cPar.ReadGenotypes(UtX, G, false);

      // perform BSLMM analysis
      BSLMM cBslmm;
      cBslmm.CopyFromParam(cPar);
      cBslmm.MCMC(UtX, y);
      cBslmm.CopyToParam(cPar);
      // else, if rho!=1
    } else {
      DMatrix U;
      U.shape = [y.elements.length, y.elements.length];
      DMatrix eval;
      eval.shape = [1, y.elements.length];
      DMatrix UtW;
      UtW.shape = [y.elements.length, W.shape[1]];
      DMatrix Uty;
      Uty.shape = [1, y.elements.length];

      // read relatedness matrix G
      if (!(cPar.file_kin).empty()) {
        cPar.ReadGenotypes(UtX, G, false);

        // read relatedness matrix G
        ReadFile_kin(cPar.file_kin, cPar.indicator_idv, cPar.mapID2num,
                     cPar.k_mode, cPar.error, G);
        if (cPar.error == true) {
          cout << "error! fail to read kinship/relatedness file. " << endl;
          return;
        }

        // center matrix G
        CenterMatrix(G);
        validate_K(G,cPar.mode_check,cPar.mode_strict);
      } else {
        cPar.ReadGenotypes(UtX, G, true);
      }

      // eigen-decomposition and calculate trace_G
      cout << "Start Eigen-Decomposition..." << endl;

      cPar.trace_G = EigenDecomp_Zeroed(G, U, eval, 0);

      // calculate UtW and Uty
      CalcUtX(U, W, UtW);
      CalcUtX(U, y, Uty);

      // calculate REMLE/MLE estimate and pve
      CalcLambda('L', eval, UtW, Uty, cPar.l_min, cPar.l_max, cPar.n_region,
                 cPar.l_mle_null, cPar.logl_mle_H0);
      CalcLambda('R', eval, UtW, Uty, cPar.l_min, cPar.l_max, cPar.n_region,
                 cPar.l_remle_null, cPar.logl_remle_H0);
      CalcPve(eval, UtW, Uty, cPar.l_remle_null, cPar.trace_G, cPar.pve_null,
              cPar.pve_se_null);

      cPar.PrintSummary();

      // Creat and calcualte UtX, use a large memory
      writeln("Calculating UtX...");
      CalcUtX(U, UtX);

      // perform BSLMM or BSLMMDAP analysis
      if (cPar.a_mode == 11 || cPar.a_mode == 12 || cPar.a_mode == 13) {
        BSLMM cBslmm;
        cBslmm.CopyFromParam(cPar);
        if (cPar.a_mode == 12) { // ridge regression
          cBslmm.RidgeR(U, UtX, Uty, eval, cPar.l_remle_null);
        } else { // Run MCMC
          cBslmm.MCMC(U, UtX, Uty, eval, y);
        }
        cPar.time_opt =
            (clock() - time_start) / (double(CLOCKS_PER_SEC) * 60.0);
        cBslmm.CopyToParam(cPar);
      } else {
      }
    }
  }

  // BSLMM-DAP
  if (cPar.a_mode == 14 || cPar.a_mode == 15 || cPar.a_mode == 16) {
    if (cPar.a_mode == 14) {
      DMatrix y;
      y.shape = [1, cPar.ni_test];
      DMatrix W;
      W.shape = [y.elements.length, cPar.n_cvt];
      DMatrix G;
      G.shape = [y.elements.length, y.elements.length];
      DMatrix UtX;
      UtX.shape = [y.elements.length, cPar.ns_test];

      // set covariates matrix W and phenotype vector y
      // an intercept should be included in W,
      cPar.CopyCvtPhen(W, y, 0);

      // center y, even for case/control data
      cPar.pheno_mean = CenterVector(y);

      // run bvsr if rho==1
      if (cPar.rho_min == 1 && cPar.rho_max == 1) {
        // read genotypes X (not UtX)
        cPar.ReadGenotypes(UtX, G, false);

        // perform BSLMM analysis
        BSLMM cBslmm;
        cBslmm.CopyFromParam(cPar);
        cBslmm.MCMC(UtX, y);
        cBslmm.CopyToParam(cPar);
        // else, if rho!=1
      } else {
        DMatrix U;
        U.shape = [y.elements.length, y.elements.length];
        DMatrix eval;
        eval.shape = [1, y.elements.length];
        DMatrix UtW;
        UtW.shape = [y.elements.length, W.shape[1]];
        DMatrix Uty;
        Uty.shape = [1, y.elements.length];

        // read relatedness matrix G
        if (!(cPar.file_kin).empty()) {
          cPar.ReadGenotypes(UtX, G, false);

          // read relatedness matrix G
          ReadFile_kin(cPar.file_kin, cPar.indicator_idv, cPar.mapID2num,
                       cPar.k_mode, cPar.error, G);
          if (cPar.error == true) {
            writeln("error! fail to read kinship/relatedness file. ");
            return;
          }

          // center matrix G
          CenterMatrix(G);
          validate_K(G,cPar.mode_check,cPar.mode_strict);

        } else {
          cPar.ReadGenotypes(UtX, G, true);
        }

        // eigen-decomposition and calculate trace_G
        writeln("Start Eigen-Decomposition...");
        time_start = clock();
        cPar.trace_G = EigenDecomp_Zeroed(G, U, eval, 0);

        // calculate UtW and Uty
        CalcUtX(U, W, UtW);
        CalcUtX(U, y, Uty);

        // calculate REMLE/MLE estimate and pve
        CalcLambda('L', eval, UtW, Uty, cPar.l_min, cPar.l_max, cPar.n_region,
                   cPar.l_mle_null, cPar.logl_mle_H0);
        CalcLambda('R', eval, UtW, Uty, cPar.l_min, cPar.l_max, cPar.n_region,
                   cPar.l_remle_null, cPar.logl_remle_H0);
        CalcPve(eval, UtW, Uty, cPar.l_remle_null, cPar.trace_G, cPar.pve_null,
                cPar.pve_se_null);

        cPar.PrintSummary();

        // Creat and calcualte UtX, use a large memory
        writeln("Calculating UtX...");
        CalcUtX(U, UtX);

        // perform analysis; assume X and y are already centered
        BSLMMDAP cBslmmDap;
        cBslmmDap.CopyFromParam(cPar);
        cBslmmDap.DAP_CalcBF(U, UtX, Uty, eval, y);

        cBslmmDap.CopyToParam(cPar);

      }

    } else if (cPar.a_mode == 15) {
      // perform EM algorithm and estimate parameters
      string[] vec_rs;
      double[] vec_sa2, vec_sb2, wab;
      //vector<vector<vector<double>>> BF;

      // read hyp and bf files (functions defined in BSLMMDAP)
      ReadFile_hyb(cPar.file_hyp, vec_sa2, vec_sb2, wab);
      ReadFile_bf(cPar.file_bf, vec_rs, BF);

      cPar.ns_test = vec_rs.size();
      if (wab.size() != BF[0][0].size()) {
        writeln("error! hyp and bf files dimension do not match");
      }

      // load annotations
      DMatrix Ac;
      DMatrix Ad;
      DMatrix dlevel;
      size_t kc, kd;
      if (!cPar.file_cat.empty()) {
        ReadFile_cat(cPar.file_cat, vec_rs, Ac, Ad, dlevel, kc, kd);
      } else {
        kc = 0;
        kd = 0;
      }

      writeln("## number of blocks = ", BF.size());
      writeln("## number of analyzed SNPs = ", vec_rs.size());
      writeln("## grid size for hyperparameters = ", wab.size());
      writeln("## number of continuous annotations = ", kc);
      writeln("## number of discrete annotations = ", kd);

      // DAP_EstimateHyper (const size_t kc, const size_t kd, const
      // vector<string> &vec_rs, const vector<double> &vec_sa2, const
      // vector<double> &vec_sb2, const vector<double> &wab, const
      // vector<vector<vector<double> > > &BF, gsl_matrix *Ac, gsl_matrix_int
      // *Ad, gsl_vector_int *dlevel);

      // perform analysis
      BSLMMDAP cBslmmDap;
      cBslmmDap.CopyFromParam(cPar);
      cBslmmDap.DAP_EstimateHyper(kc, kd, vec_rs, vec_sa2, vec_sb2, wab, BF, Ac,
                                  Ad, dlevel);
      cBslmmDap.CopyToParam(cPar);

    } else {
      //
    }
  }

  return;
}
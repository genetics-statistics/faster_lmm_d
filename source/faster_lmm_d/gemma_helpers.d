module faster_lmm_d.gemma_helpers;

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
import faster_lmm_d.gemma_lmm;
import faster_lmm_d.gemma_param;
import faster_lmm_d.kinship;
import faster_lmm_d.lmm2;
import faster_lmm_d.memory;
import faster_lmm_d.optmatrix;
import faster_lmm_d.output;
import faster_lmm_d.phenotype;
import faster_lmm_d.helpers : sum;
import faster_lmm_d.gemma;

import gsl.cdf;
import gsl.errno;
import gsl.math;
import gsl.min;
import gsl.roots;


void kinship_with_loco(Param cPar){
  writeln("Calculating Relatedness Matrix ... ");

  DMatrix G;
  G.shape = [cPar.ni_total, cPar.ni_total];
  log(G, "allocate G"); // just to be sure

  cPar.CalcKin(G);

  if (cPar.error == true) {
    writeln("error! fail to calculate relatedness matrix. ");
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

void calc_weights(Param cPar){
  writeln("Calculating Weights ... ");

  //VARCOV cVarcov;
  //cVarcov.CopyFromParam(cPar);

  //if (!cPar.file_bfile.empty()) {
  //  cVarcov.AnalyzePlink();
  //} else {
  //  cVarcov.AnalyzeBimbam();
  //}

  //cVarcov.CopyToParam(cPar);
}

void calc_S(Param cPar){
  writeln("Calculating the S Matrix ... ");

  DMatrix S;
  S.shape = [cPar.n_vc * 2, cPar.n_vc];
  DMatrix ns;
  ns.shape = [cPar.n_vc + 1];
  S = set_zeros_dmatrix(S);
  ns = set_zeros_dmatrix(ns);

  DMatrix S_mat;// = get_sub_dmatrix(S, 0, 0, cPar.n_vc, cPar.n_vc);
  DMatrix Svar_mat;// =
      //get_sub_dmatrix(S, cPar.n_vc, 0, cPar.n_vc, cPar.n_vc);
  DMatrix ns_vec; //= gsl_vector_subvector(ns, 0, cPar.n_vc);

  DMatrix K;
  K.shape = [cPar.ni_test, cPar.n_vc * cPar.ni_test];
  DMatrix A;
  A.shape = [cPar.ni_test, cPar.n_vc * cPar.ni_test];
  K = set_zeros_dmatrix(K);
  A = set_zeros_dmatrix(A);

  DMatrix y;
  y.shape = [1, cPar.ni_test];
  DMatrix W;
  W.shape = [cPar.ni_test, cPar.n_cvt];

  cPar.CopyCvtPhen(W, y, 0);

  string[] setSnps_beta;
  mapRS mapRS2wA, mapRS2wK;

  cPar.ObtainWeight(setSnps_beta, mapRS2wK);

  cPar.CalcS(mapRS2wA, mapRS2wK, W, A, K, S_mat, Svar_mat, ns_vec);
  if (cPar.error == true) {
    writeln("error! fail to calculate the S matrix. ");
    return;
  }

  //gsl_vector_set(ns, cPar.n_vc, cPar.ni_test);

  cPar.WriteMatrix(S, "S");
  cPar.WriteVector(ns, "size");
  cPar.WriteVar("snps");
}

void calc_Vq(Param cPar){
  DMatrix Vq;
  Vq.shape = [cPar.n_vc, cPar.n_vc];
  DMatrix q;
  q.shape = [1, cPar.n_vc];
  DMatrix s;
  s.shape = [1, cPar.n_vc + 1];
  q = set_zeros_dmatrix(q);
  s = set_zeros_dmatrix(s);

  DMatrix s_vec;// = gsl_vector_subvector(s, 0, cPar.n_vc);

  size_t[] vec_cat, vec_ni;
  double[] vec_weight, vec_z2;
  mapRS mapRS2weight;
  mapRS2weight.clear();

  ReadFile_beta(cPar.file_beta, cPar.mapRS2cat, mapRS2weight, vec_cat, vec_ni,
                vec_weight, vec_z2, cPar.ni_total, cPar.ns_total,
                cPar.ns_test);
  writeln("## number of total individuals = ", cPar.ni_total);
  writeln("## number of total SNPs = ", cPar.ns_total);
  writeln("## number of analyzed SNPs = ", cPar.ns_test);
  writeln("## number of variance components = ", cPar.n_vc);
  writeln("Calculating the q vector ... ");
  Calcq(cPar.n_block, vec_cat, vec_ni, vec_weight, vec_z2, Vq, q, s_vec);

  if (cPar.error == true) {
    writeln("error! fail to calculate the q vector.");
    return;
  }

  s.elements[cPar.n_vc] = cPar.ni_total;

  cPar.WriteMatrix(Vq, "Vq");
  cPar.WriteVector(q, "q");
  cPar.WriteVector(s, "size");
}

void vc_estimation(Param cPar){
  if (!cPar.file_beta.empty()) {
    // need to obtain a common set of SNPs between beta file and the genotype
    // file; these are saved in mapRS2wA and mapRS2wK
    // normalize the weight in mapRS2wK to have an average of one; each
    // element of mapRS2wA is 1
    // update indicator_snps, so that the numbers are in accordance with
    // mapRS2wK

    string[] setSnps_beta;
    ReadFile_snps_header(cPar.file_beta, setSnps_beta);

    mapRS mapRS2wA, mapRS2wK;
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

    //gsl_matrix_set_zero(K);
    //gsl_matrix_set_zero(A);

    //gsl_matrix_set_zero(S);
    //gsl_matrix_set_zero(Vq);
    //gsl_vector_set_zero(q);
    //gsl_vector_set_zero(s);

    cPar.CopyCvtPhen(W, y, 0);

    DMatrix S_mat =
        get_sub_dmatrix(S, 0, 0, cPar.n_vc, cPar.n_vc);
    DMatrix Svar_mat =
        get_sub_dmatrix(S, cPar.n_vc, 0, cPar.n_vc, cPar.n_vc);
    DMatrix s_vec;// = gsl_vector_subvector(s, 0, cPar.n_vc);

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
    Calcq(cPar.n_block, vec_cat, vec_ni, vec_weight, vec_z2, Vq, q, s_vec);

    // compute S
    cPar.CalcS(mapRS2wA, mapRS2wK, W, A, K, S_mat, Svar_mat, s_vec);
    if (cPar.error == true) {
      writeln("error! fail to calculate the S matrix.");
      return;
    }

    // compute vc estimates
    CalcVCss(Vq, S_mat, Svar_mat, q, s_vec, cPar.ni_study, cPar.v_pve,
             cPar.v_se_pve, cPar.pve_total, cPar.se_pve_total, cPar.v_sigma2,
             cPar.v_se_sigma2, cPar.v_enrich, cPar.v_se_enrich);

    //assert(!has_nan(cPar.v_se_pve));

    // if LDSC weights, then compute the weights and run the above steps again
    if (cPar.a_mode == 62) {
      // compute the weights and normalize the weights for A
      cPar.UpdateWeight(1, mapRS2wK, cPar.ni_study, s_vec, mapRS2wA);

      // read beta file again, and update weigths vector
      ReadFile_beta(cPar.file_beta, cPar.mapRS2cat, mapRS2wA, vec_cat, vec_ni,
                    vec_weight, vec_z2, cPar.ni_study, cPar.ns_total, cPar.ns_test);

      // compute q
      Calcq(cPar.n_block, vec_cat, vec_ni, vec_weight, vec_z2, Vq, q, s_vec);

      // compute S
      cPar.CalcS(mapRS2wA, mapRS2wK, W, A, K, S_mat, Svar_mat, s_vec);

      if (cPar.error == true) {
        writeln("error! fail to calculate the S matrix.");
        return;
      }

      // compute vc estimates
      CalcVCss(Vq, S_mat, Svar_mat, q, s_vec, cPar.ni_study, cPar.v_pve,
               cPar.v_se_pve, cPar.pve_total, cPar.se_pve_total, cPar.v_sigma2,
               cPar.v_se_sigma2, cPar.v_enrich, cPar.v_se_enrich);
      //assert(!has_nan(cPar.v_se_pve));
    }


  s.elements[cPar.n_vc] = cPar.ni_test;

    cPar.WriteMatrix(S, "S");
    cPar.WriteMatrix(Vq, "Vq");
    cPar.WriteVector(q, "q");
    cPar.WriteVector(s, "size");

  } else if (!cPar.file_study.empty() || !cPar.file_mstudy.empty()) {
    if (!cPar.file_study.empty()) {
      string sfile = cPar.file_study ~ ".size.txt";
      //CountFileLines(sfile, cPar.n_vc);
    } else {
      string file_name;
      //igzstream infile(cPar.file_mstudy.c_str(), igzstream::in);
      //if (!infile) {
      //  writeln("error! fail to open mstudy file:", cPar.file_study);
      //  return;
      //}

      //safeGetline(infile, file_name);

      //infile.clear();
      //infile.close();

      string sfile = file_name ~ ".size.txt";
      //CountFileLines(sfile, cPar.n_vc);
    }

    cPar.n_vc = cPar.n_vc - 1;

    DMatrix S;
    S.shape = [2 * cPar.n_vc, cPar.n_vc];
    DMatrix Vq;
    Vq.shape = [cPar.n_vc, cPar.n_vc];
    // gsl_matrix *V=gsl_matrix_alloc (cPar.n_vc+1,
    // (cPar.n_vc*(cPar.n_vc+1))/2*(cPar.n_vc+1) );
    // gsl_matrix *Vslope=gsl_matrix_alloc (n_lines+1,
    // (n_lines*(n_lines+1))/2*(n_lines+1) );
    DMatrix q;
    q.shape = [1, cPar.n_vc];
    DMatrix s_study;
    s_study.shape = [1, cPar.n_vc];
    DMatrix s_ref;
    s_ref.shape = [1, cPar.n_vc];
    DMatrix s;
    s.shape = [1, cPar.n_vc + 1];

    //gsl_matrix_set_zero(S);
    DMatrix S_mat =
        get_sub_dmatrix(S, 0, 0, cPar.n_vc, cPar.n_vc);
    DMatrix Svar_mat =
        get_sub_dmatrix(S, cPar.n_vc, 0, cPar.n_vc, cPar.n_vc);

    //gsl_matrix_set_zero(Vq);
    // gsl_matrix_set_zero(V);
    // gsl_matrix_set_zero(Vslope);
    //gsl_vector_set_zero(q);
    //gsl_vector_set_zero(s_study);
    //gsl_vector_set_zero(s_ref);

    //if (!cPar.file_study.empty()) {
    //  ReadFile_study(cPar.file_study, Vq, q, s_study, cPar.ni_study);
    //} else {
    //  ReadFile_mstudy(cPar.file_mstudy, Vq, q, s_study, cPar.ni_study);
    //}

    //if (!cPar.file_ref.empty()) {
    //  ReadFile_ref(cPar.file_ref, &S_mat.matrix, &Svar_mat.matrix, s_ref,
    //               cPar.ni_ref);
    //} else {
    //  ReadFile_mref(cPar.file_mref, &S_mat.matrix, &Svar_mat.matrix, s_ref,
    //                cPar.ni_ref);
    //}

    writeln("## number of variance components = ", cPar.n_vc);
    writeln("## number of individuals in the sample = ", cPar.ni_study);
    writeln("## number of individuals in the reference = ", cPar.ni_ref);

    CalcVCss(Vq, S_mat, Svar_mat, q, s_study, cPar.ni_study,
             cPar.v_pve, cPar.v_se_pve, cPar.pve_total, cPar.se_pve_total,
             cPar.v_sigma2, cPar.v_se_sigma2, cPar.v_enrich,
             cPar.v_se_enrich);
    //assert(!has_nan(cPar.v_se_pve));

    DMatrix s_sub;// = gsl_vector_subvector(s, 0, cPar.n_vc);
    s_sub.elements =  s_ref.elements.dup;
    s.elements[cPar.n_vc] = cPar.ni_ref;

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
      //ReadFile_mk(cPar.file_mk, cPar.indicator_idv, cPar.mapID2num,
      //            cPar.k_mode, cPar.error, G);
      if (cPar.error == true) {
        writeln("error! fail to read kinship/relatedness file.");
        return;
      }

      // center matrix G, and obtain v_traceG
      double d = 0;
      //(cPar.v_traceG).clear();
      for (size_t i = 0; i < cPar.n_vc; i++) {
        DMatrix G_sub =
            get_sub_dmatrix(G, 0, i * G.shape[0], G.shape[0], G.shape[0]);
        CenterMatrix(G_sub);
        d = 0;
        for (size_t j = 0; j < G.shape[0]; j++) {
          d += accessor(G_sub, j, j);
        }
        d /= to!double(G.shape[0]);
        //(cPar.v_traceG).push_back(d);
      }
    } else if (!(cPar.file_kin).empty()) {
      ReadFile_kin(cPar.file_kin, cPar.indicator_idv, cPar.mapID2num,
                   cPar.k_mode, cPar.error, G);
      if (cPar.error == true) {
        writeln("error! fail to read kinship/relatedness file.");
        return;
      }

      // center matrix G
      CenterMatrix(G);
      validate_K(G,cPar.mode_check,cPar.mode_strict);

      //(cPar.v_traceG).clear();
      double d = 0;
      for (size_t j = 0; j < G.shape[0]; j++) {
        d += accessor(G, j, j);
      }
      d /= to!double(G.shape[0]);
      //(cPar.v_traceG).push_back(d);
    }
    // fit multiple variance components
    if (cPar.n_ph == 1) {
      //      if (cPar.n_vc==1) {
      //      } else {
      DMatrix Y_col = get_col(Y, 0);
      //VC cVc;
      //cVc.CopyFromParam(cPar);
      //if (cPar.a_mode == 61) {
      //  cVc.CalcVChe(G, W, &Y_col.vector);
      //} else if (cPar.a_mode == 62) {
      //  cVc.CalcVCreml(cPar.noconstrain, G, W, &Y_col.vector);
      //} else {
      //  cVc.CalcVCacl(G, W, &Y_col.vector);
      //}
      //cVc.CopyToParam(cPar);
      // obtain pve from sigma2
      // obtain se_pve from se_sigma2

      //}
    }
  }
}

void calc_SNP_covariance(Param cPar){
  //VARCOV cVarcov;
  //cVarcov.CopyFromParam(cPar);

  //if (!cPar.file_bfile.empty()) {
  //  cVarcov.AnalyzePlink();
  //} else {
  //  cVarcov.AnalyzeBimbam();
  //}

  //cVarcov.CopyToParam(cPar);
}

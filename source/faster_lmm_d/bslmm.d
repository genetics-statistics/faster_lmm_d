module faster_lmm_d.bslmm;

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
import faster_lmm_d.gemma;
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

void bslmm_predictor(Param cPar){
  DMatrix y_prdt;

  y_prdt.shape = [1, cPar.ni_total - cPar.ni_test];

  // set to zero TODO
  y_prdt = set_zeros_dmatrix(y_prdt);

  PRDT cPRDT;
  //cPRDT.CopyFromParam(cPar);

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
    for (size_t i = 0; i < cPar.indicator_idv.length; i++) {
      //indicator_all.push_back(1);
      if (cPar.indicator_bv.elements[i] == 1) {
        //gsl_vector_set(u_hat, c_bv, cPar.vec_bv[i]);
        c_bv++;
      }
    }

    ReadFile_kin(cPar.file_kin, indicator_all, cPar.mapID2num, cPar.k_mode, cPar.error, G);
      writeln("error! fail to read kinship/relatedness file.");
      return;
    }

    if (cPar.error == true) {
    // read u
    //cPRDT.AddBV(G, u_hat, y_prdt);

  }

  // add beta
  if (!cPar.file_bfile.empty()) {
    //cPRDT.AnalyzePlink(y_prdt);
  } else {
    //cPRDT.AnalyzeBimbam(y_prdt);
  }

  // add mu
  add_dmatrix_num(y_prdt, cPar.pheno_mean);

  // convert y to probability if needed
  if (cPar.a_mode == 42) {
    double d;
    for (size_t i = 0; i < y_prdt.elements.length; i++) {
      d = y_prdt.elements[i];
      d = gsl_cdf_gaussian_P(d, 1.0);
      y_prdt.elements[i] = d;
    }
  }

  //cPRDT.CopyToParam(cPar);

  //cPRDT.WriteFiles(y_prdt);
}

void fit_bslmm_DAP(Param cPar){
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
    //cPar.pheno_mean = CenterVector(y);

    // run bvsr if rho==1
    if (cPar.rho_min == 1 && cPar.rho_max == 1) {
      // read genotypes X (not UtX)
      //cPar.ReadGenotypes(UtX, G, false);

      // perform BSLMM analysis
      //BSLMM cBslmm;
      //cBslmm.CopyFromParam(cPar);
      //cBslmm.MCMC(UtX, y);
      //cBslmm.CopyToParam(cPar);
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
        //cPar.ReadGenotypes(UtX, G, false);

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
        //cPar.ReadGenotypes(UtX, G, true);
      }

      // eigen-decomposition and calculate trace_G
      writeln("Start Eigen-Decomposition...");

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

      //cPar.PrintSummary();

      // Creat and calcualte UtX, use a large memory
      writeln("Calculating UtX...");
      CalcUtX(U, UtX);

      // perform analysis; assume X and y are already centered
      //BSLMMDAP cBslmmDap;
      //cBslmmDap.CopyFromParam(cPar);
      //cBslmmDap.DAP_CalcBF(U, UtX, Uty, eval, y);

      //cBslmmDap.CopyToParam(cPar);

    }

  } else if (cPar.a_mode == 15) {
    // perform EM algorithm and estimate parameters
    string[] vec_rs;
    double[] vec_sa2, vec_sb2, wab;
    DMatrix BF; //its 3 dimensional

    // read hyp and bf files (functions defined in BSLMMDAP)
    //ReadFile_hyb(cPar.file_hyp, vec_sa2, vec_sb2, wab);
    //ReadFile_bf(cPar.file_bf, vec_rs, BF);

    cPar.ns_test = vec_rs.length;
    if (wab.length != BF.elements.length) { //check BF[0][0]
      writeln("error! hyp and bf files dimension do not match");
    }

    // load annotations
    DMatrix Ac;
    DMatrix Ad;
    DMatrix dlevel;
    size_t kc, kd;
    if (!cPar.file_cat.empty()) {
      //ReadFile_cat(cPar.file_cat, vec_rs, Ac, Ad, dlevel, kc, kd);
    } else {
      kc = 0;
      kd = 0;
    }

    writeln("## number of blocks = ", BF.elements.length);
    writeln("## number of analyzed SNPs = ", vec_rs.length);
    writeln("## grid size for hyperparameters = ", wab.length);
    writeln("## number of continuous annotations = ", kc);
    writeln("## number of discrete annotations = ", kd);

    // DAP_EstimateHyper (const size_t kc, const size_t kd, const
    // vector<string> &vec_rs, const vector<double> &vec_sa2, const
    // vector<double> &vec_sb2, const vector<double> &wab, const
    // vector<vector<vector<double> > > &BF, gsl_matrix *Ac, gsl_matrix_int
    // *Ad, gsl_vector_int *dlevel);

    // perform analysis
    //BSLMMDAP cBslmmDap;
    //cBslmmDap.CopyFromParam(cPar);
    //cBslmmDap.DAP_EstimateHyper(kc, kd, vec_rs, vec_sa2, vec_sb2, wab, BF, Ac,
    //                            Ad, dlevel);
    //cBslmmDap.CopyToParam(cPar);

  } else {
    //
  }
}

void calc_cofidence_interval(Param cPar){
  // read reference file first
  DMatrix S;
  S.shape = [cPar.n_vc, cPar.n_vc];
  DMatrix Svar;
  Svar.shape = [cPar.n_vc, cPar.n_vc];
  DMatrix s_ref;
  s_ref.shape = [1, cPar.n_vc];

  //gsl_matrix_set_zero(S);
  //gsl_matrix_set_zero(Svar);
  //gsl_vector_set_zero(s_ref);

  if (!cPar.file_ref.empty()) {
    //ReadFile_ref(cPar.file_ref, S, Svar, s_ref, cPar.ni_ref);
  } else {
    //ReadFile_mref(cPar.file_mref, S, Svar, s_ref, cPar.ni_ref);
  }

  // need to obtain a common set of SNPs between beta file and the genotype
  // file; these are saved in mapRS2wA and mapRS2wK
  // normalize the weight in mapRS2wK to have an average of one; each element
  // of mapRS2wA is 1


  string[] setSnps_beta;
  ReadFile_snps_header(cPar.file_beta, setSnps_beta);

  // obtain the weights for wA, which contains the SNP weights for SNPs used
  // in the model

  mapRS mapRS2wK;
  cPar.ObtainWeight(setSnps_beta, mapRS2wK);

  // set up matrices and vector
  DMatrix Xz;
  Xz.shape = [cPar.ni_test, cPar.n_vc];
  DMatrix XWz;
  XWz.shape = [cPar.ni_test, cPar.n_vc];
  DMatrix XtXWz;
  //XtXWz.shape = [mapRS2wK.size(), cPar.n_vc * cPar.n_vc];
  DMatrix w;
  //w.shape = [1, mapRS2wK.size()];
  DMatrix w1;
  //w1.shape = [1, mapRS2wK.size()];
  DMatrix z;
  //z.shape = [1, mapRS2wK.size()];
  DMatrix s_vec;
  s_vec.shape = [1, cPar.n_vc];

  size_t[] vec_cat, vec_size;
  double[] vec_z;

  mapRS mapRS2z, mapRS2wA;
  mapRS mapRS2A1;
  string file_str;

  // update s_vec, the number of snps in each category
  for (size_t i = 0; i < cPar.n_vc; i++) {
    //vec_size.push_back(0);
  }

  //TODO
  //for (map<string, double>::const_iterator it = mapRS2wK.begin();
  //     it != mapRS2wK.end(); ++it) {
  //  vec_size[cPar.mapRS2cat[it.first]]++;
  //}

  for (size_t i = 0; i < cPar.n_vc; i++) {
    //gsl_vector_set(s_vec, i, vec_size[i]);
    s_vec.elements[i] = vec_size[i];
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
  //ReadFile_beta(cPar.file_beta, mapRS2wA, mapRS2A1, mapRS2z);

  // update snp indicator, save weights to w, save z-scores to vec_z, save
  // category label to vec_cat
  // sign of z is determined by matching alleles
  //cPar.UpdateSNPnZ(mapRS2wA, mapRS2A1, mapRS2z, w, z, vec_cat);

  // compute an n by k matrix of X_iWz
  writeln("Calculating Xz ... ");

  //gsl_matrix_set_zero(Xz);
  //gsl_vector_set_all(w1, 1);

  //if (!cPar.file_bfile.empty()) {
  //  file_str = cPar.file_bfile + ".bed";
  //  PlinkXwz(file_str, cPar.d_pace, cPar.indicator_idv, cPar.indicator_snp,
  //           vec_cat, w1, z, 0, Xz);
  //} else if (!cPar.file_geno.empty()) {
  //  BimbamXwz(cPar.file_geno, cPar.d_pace, cPar.indicator_idv,
  //            cPar.indicator_snp, vec_cat, w1, z, 0, Xz);
  //} else if (!cPar.file_mbfile.empty()) {
  //  MFILEXwz(1, cPar.file_mbfile, cPar.d_pace, cPar.indicator_idv,
  //           cPar.mindicator_snp, vec_cat, w1, z, Xz);
  //} else if (!cPar.file_mgeno.empty()) {
  //  MFILEXwz(0, cPar.file_mgeno, cPar.d_pace, cPar.indicator_idv,
  //           cPar.mindicator_snp, vec_cat, w1, z, Xz);
  //}
  //if (cPar.a_mode == 66) {
  //  gsl_matrix_memcpy(XWz, Xz);
  //} else if (cPar.a_mode == 67) {
  //  writeln("Calculating XWz ... ");

  //  gsl_matrix_set_zero(XWz);

  //  if (!cPar.file_bfile.empty()) {
  //    file_str = cPar.file_bfile + ".bed";
  //    PlinkXwz(file_str, cPar.d_pace, cPar.indicator_idv, cPar.indicator_snp,
  //             vec_cat, w, z, 0, XWz);
  //  } else if (!cPar.file_geno.empty()) {
  //    BimbamXwz(cPar.file_geno, cPar.d_pace, cPar.indicator_idv,
  //              cPar.indicator_snp, vec_cat, w, z, 0, XWz);
  //  } else if (!cPar.file_mbfile.empty()) {
  //    MFILEXwz(1, cPar.file_mbfile, cPar.d_pace, cPar.indicator_idv,
  //             cPar.mindicator_snp, vec_cat, w, z, XWz);
  //  } else if (!cPar.file_mgeno.empty()) {
  //    MFILEXwz(0, cPar.file_mgeno, cPar.d_pace, cPar.indicator_idv,
  //             cPar.mindicator_snp, vec_cat, w, z, XWz);
  //  }
  //}
  // compute an p by k matrix of X_j^TWX_iWz
  writeln("Calculating XtXWz ... ");
  //gsl_matrix_set_zero(XtXWz);

  //if (!cPar.file_bfile.empty()) {
  //  file_str = cPar.file_bfile + ".bed";
  //  PlinkXtXwz(file_str, cPar.d_pace, cPar.indicator_idv, cPar.indicator_snp,
  //             XWz, 0, XtXWz);
  //} else if (!cPar.file_geno.empty()) {
  //  BimbamXtXwz(cPar.file_geno, cPar.d_pace, cPar.indicator_idv,
  //              cPar.indicator_snp, XWz, 0, XtXWz);
  //} else if (!cPar.file_mbfile.empty()) {
  //  MFILEXtXwz(1, cPar.file_mbfile, cPar.d_pace, cPar.indicator_idv,
  //             cPar.mindicator_snp, XWz, XtXWz);
  //} else if (!cPar.file_mgeno.empty()) {
  //  MFILEXtXwz(0, cPar.file_mgeno, cPar.d_pace, cPar.indicator_idv,
  //             cPar.mindicator_snp, XWz, XtXWz);
  //}
  //// compute confidence intervals
  //CalcCIss(Xz, XWz, XtXWz, S, Svar, w, z, s_vec, vec_cat, cPar.v_pve,
  //         cPar.v_se_pve, cPar.pve_total, cPar.se_pve_total, cPar.v_sigma2,
  //         cPar.v_se_sigma2, cPar.v_enrich, cPar.v_se_enrich);
  //assert(!has_nan(cPar.v_se_pve));

}

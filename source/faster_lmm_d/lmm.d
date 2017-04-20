module faster_lmm_d.lmm;
import faster_lmm_d.dmatrix;
import faster_lmm_d.gwas;
import faster_lmm_d.helpers;
import faster_lmm_d.optmatrix;
import faster_lmm_d.kinship;
import faster_lmm_d.phenotype;

import std.stdio;
import std.typecons;

struct KGstruct{
  dmatrix K, G;

  this(dmatrix K, dmatrix G){
    this.K = K;
    this.G = G;
  }
}

auto run_gwas(string species, int n, int m, dmatrix k, double[] y, dmatrix geno){
    //int cov, bool reml = true,bool refit = false, string inputfn = "", bool new_code = true){
//  //"""
//  //Invoke pylmm using genotype as a matrix or as a (SNP) iterator.
//  //"""
  int cov;
  bool reml = true;
  bool refit = false;
  string inputfn = "";
  bool new_code = true;

  writeln("run_gwas");
  writeln("pheno ", y.length," ", y[0..5]);
  writeln(geno.shape);
  //assert(geno.shape[0] == y.size, [np.size(geno[0]), y.size]);
  assert(y.length == n);
  //if(k != null){
  //writeln(k.shape[0]);
  //}
  //else{
  writeln("No kinship matrix passed in!");
  //}

  writeln(m,geno.shape);
  assert(geno.shape[1] == m);
//  //sys.exit(1)

  if(species == "human"){
    writeln("kinship", k );
    //return run_human(y, cov, inputfn, k, refit);
    return Tuple!(double[], double[])([-1],[-1]);
  }
  else{
    writeln("geno");
    double ps, ts;
    if(new_code){
      return run_other_new(n, m, y, geno, reml, refit);
    }else{
      return run_other_new(n, m, y, geno, reml, refit);
    }
  }
}

auto run_other_new(int n, int m, double[] pheno_vector, dmatrix geno, bool restricted_max_likelihood= true, bool refit = false){

  //"""Takes the phenotype vector and genotype matrix and returns a set of p-values and t-statistics

  //restricted_max_likelihood -- whether to use restricted max likelihood; True or False
  //refit -- whether to refit the variance component for each marker

  //"""

  writeln("Running the new LMM2 engine in run_other_new");
  writeln("REML=",restricted_max_likelihood," REFIT=",refit);

  //# Adjust phenotypes

  phenoStruct pheno = remove_missing(n,pheno_vector);
  n = pheno.n;
  double[] Y = pheno.Y;
  bool[] keep = pheno.keep;

  geno = removeCols(geno,keep);
  writeln("Calculate Kinship");
  KGstruct KG = calculate_kinship_new(geno);
  writeln("kinship_matrix.shape: ", KG.K.shape);
  writeln("run_other_new genotype_matrix: ", KG.G.shape);

  return gwas(Y, KG.G, KG.K, true, false, true);
}

KGstruct calculate_kinship_new(dmatrix genotype_matrix){
  //"""
  //Call the new kinship calculation where genotype_matrix contains
  //inds (columns) by snps (rows).
  //"""
  writeln("call calculate_kinship_new");
  writeln(genotype_matrix.shape);
  dmatrix G = normalize_along_row(genotype_matrix);
  dmatrix K = kinship_full(G);
  return KGstruct(K, G);
}

module faster_lmm_d.phenotype;
import faster_lmm_d.dmatrix;
import faster_lmm_d.helpers;
import std.stdio;

struct phenoStruct{
  double[] Y;
  bool[] keep;
  int n;

  this(double[] Y, bool[] keep, int n){
    this.Y = Y;
    this.keep = keep;
    this.n = n;
  }
}

int remove_missing(ref dmatrix n, ref dmatrix y, ref dmatrix g){
  //"""
  //Remove missing data from matrices, make sure the genotype data has
  //individuals as rows
  //"""
  //assert(y!=null);
  assert(y.shape[0] == g.shape[0]);//,"y (n) %d, g (n,m) %s" % (y.shape[0],g.shape)

  auto y1 = y;
  auto g1 = g;
  //v = np.isnan(y)
  //keep = True - v
  //if v.sum():
  //    info("runlmm.py: Cleaning the phenotype vector and genotype matrix by removing %d individuals...\n" % (v.sum()))
  //    y1 = y[keep]
  //    g1 = g[keep,:]
  //    n = y1.shape[0]
  //return n,y1,g1,keep
  return 1;
}

phenoStruct remove_missing_new( int n, double[] y){
  //"""
  //Remove missing data. Returns new n,y,keep
  //"""
  writeln("In remove missing new");
  bool[] v = isnan(y);
  bool[] keep = negateBool(v);
  //if v.sum():
  //    info("runlmm.py: Cleaning the phenotype vector by removing %d individuals" % (v.sum()))
  double[] Y = getNumArray(y,keep);
  n = cast(int)Y.length;
  return phenoStruct(Y, keep, n);
}
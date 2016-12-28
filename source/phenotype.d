module simplelmm.phenotype;
import simplelmm.dmatrix;
import simplelmm.helpers;
import std.stdio;

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

void remove_missing_new(ref double[] Y, ref bool[] keep, int n, ref double[] y){
  //"""
  //Remove missing data. Returns new n,y,keep
  //"""
  //assert(y!=null);
  writeln("In remove missing new");
  Y = y;
  bool[] v;
  v = isnan(y);
  writeln(v);
  keep = negateBool(v);
  writeln(keep);
  //if v.sum():
  //    info("runlmm.py: Cleaning the phenotype vector by removing %d individuals" % (v.sum()))
  //    Y = y[keep]
    n = cast(int)Y.length;
}
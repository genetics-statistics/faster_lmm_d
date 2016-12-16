module simplelmm.phenotype;

int remove_missing(string n, string y, string g){
  //"""
  //Remove missing data from matrices, make sure the genotype data has
  //individuals as rows
  //"""
  //assert(y is not None)
  //assert y.shape[0] == g.shape[0],"y (n) %d, g (n,m) %s" % (y.shape[0],g.shape)

  //y1 = y
  //g1 = g
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

int remove_missing_new(string n, string y){
  //"""
  //Remove missing data. Returns new n,y,keep
  //"""
  //assert(y is not None)
  //y1 = y
  //v = np.isnan(y)
  //keep = True - v
  //if v.sum():
  //    info("runlmm.py: Cleaning the phenotype vector by removing %d individuals" % (v.sum()))
  //    y1 = y[keep]
  //    n = y1.shape[0]
  //return n,y1,keep
  return 1;
}
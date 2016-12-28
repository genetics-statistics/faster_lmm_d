module simplelmm.genotype;
import simplelmm.helpers;

int replace_missing_with_MAF(string snp_g){
  //"""
  //Replace the missing genotype with the minor allele frequency (MAF)
  //in the snp row. It is rather slow!
  //"""
  //cnt = Counter(snp_g)
  //tuples = sorted(cnt.items(), key=operator.itemgetter(1))
  //l2 = [t for t in tuples if not np.isnan(t[0])]
  //maf = l2[0][0]
  //res = np.array([maf if np.isnan(snp) else snp for snp in snp_g])
  //return res
  return 1;
}


double normalize(double[] ind_g){
  //"""
  //Run for every SNP list (for one individual) and return
  //normalized SNP genotype values with missing data filled in
  //"""
  auto g = ind_g;                        // copy to avoid side effects
  bool[] missing = isnan(g);
  bool[] along = negateBool(missing);
  dmatrix values;
   //= g[along];
  double mean = globalMean(values);      // Global mean value
  double stddev = sqrt(values.var());    // Global stddev
  g[missing] = mean;                     // Plug-in mean values for missing data
  if(stddev == 0){
    g = g - mean;                        // Subtract the mean
  }else{
    g = (g - mean) / stddev;             // Normalize the deviation
  }
  return g;
}
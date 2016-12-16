module simplelmm.genotype;

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


int normalize(string ind_g){
  //"""
  //Run for every SNP list (for one individual) and return
  //normalized SNP genotype values with missing data filled in
  //"""
  //g = np.copy(ind_g)              # copy to avoid side effects
  //missing = np.isnan(g)
  //values = g[True - missing]
  //mean = values.mean()            # Global mean value
  //stddev = np.sqrt(values.var())  # Global stddev
  //g[missing] = mean               # Plug-in mean values for missing data
  //if stddev == 0:
  //    g = g - mean                # Subtract the mean
  //else:
  //    g = (g - mean) / stddev     # Normalize the deviation
  //return g
  return 1;
}
module faster_lmm_d.genotype;
import faster_lmm_d.helpers;
import faster_lmm_d.dmatrix;
import std.math;

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
module faster_lmm_d.prune;

import faster_lmm_d.dmatrix;
import faster_lmm_d.gemma_io;
import faster_lmm_d.optmatrix;
import faster_lmm_d.gemma_param;

import std.conv;
import std.file;
import std.process;
import std.stdio;
import std.string;

void down_sampler(const string geno_fn, const ulong ni_total, const DMatrix W, const int[] indicator_idv,
                          string[] setSnps, string[string] mapRS2chr, size_t[string] mapRS2bp, double[string] mapRS2cM){

  writeln("ReadFile_geno", geno_fn);
  int[] indicator_snp;

  File outfile = File("geno.txt", "w");

  size_t ns_test;
  SNPINFO[] snpInfo;
  const double maf_level = 0.01;
  const double miss_level = 0.05;
  const double hwe_level = 0;
  const double r2_level = 0.9999;


  string filename = geno_fn;
  auto pipe = pipeShell("gunzip -c " ~ filename);
  File input = pipe.stdout;

  double[] genotype = new double[W.shape[0]];
  double[] genotype_miss = new double[W.shape[0]];

  // W refers to covariates
  double WtWi= 1/vector_ddot(W, W);

  int c_idv = 0;
  int n_0, n_1, n_2, flag_poly;
  long b_pos;
  double v_x, v_w, maf, geno, geno_old, cM;
  string rs, chr, major, minor;
  size_t file_pos, n_miss;

  int ni_test = 0;
  foreach (element; indicator_idv) {
    ni_test += element;
  }
  ns_test = 0;

  file_pos = 0;
  auto count_warnings = 0;
  foreach (line ; input.byLine) {
    auto ch_ptr = to!string(line).split(",");
    rs = ch_ptr[0];
    minor = ch_ptr[1];
    major = ch_ptr[2];
    auto chr_val = ch_ptr[3..$];

    if (setSnps.length != 0 && setSnps.count(rs) == 0) {

      // if SNP in geno but not in -snps we add an missing value
      SNPINFO sInfo = SNPINFO("-9", rs, -9, -9, minor, major,
                                0,  -9, -9, 0, 0, file_pos);
      snpInfo ~= sInfo;
      indicator_snp ~= 0;

      file_pos++;

      continue;
    }
    if (mapRS2bp.get(rs, 0) == 0) { // check
      if (count_warnings++ < 10) {
        writeln("Can't figure out position for ");
      }
      chr = "-9";
      b_pos = -9;
      cM = -9;
    } else {
      b_pos = mapRS2bp[rs];
      chr = mapRS2chr[rs];
      cM = mapRS2cM[rs];
    }

    maf = 0;
    n_miss = 0;
    flag_poly = 0;
    geno_old = -9;
    n_0 = 0;
    n_1 = 0;
    n_2 = 0;
    c_idv = 0;
    foreach(ref ele; genotype_miss){ele = 0;}
    foreach (i, idv; indicator_idv) {
      if (idv == 0)
        continue;
      auto digit = to!string(chr_val[i].strip());
      if (digit == "NA") {
        genotype_miss[c_idv] = 1;
        n_miss++;
        c_idv++;
        continue;
      }

      geno = to!double(digit);
      if (geno >= 0   && geno <= 0.5){ n_0++; }
      if (geno > 0.5  && geno <  1.5){ n_1++; }
      if (geno >= 1.5 && geno <= 2.0){ n_2++; }

      genotype[c_idv] = geno;

      if (flag_poly == 0) {
        geno_old = geno;
        flag_poly = 2;
      }

      if (flag_poly == 2 && geno != geno_old) { flag_poly = 1; }

      maf += geno;

      c_idv++;
    }

    maf /= 2.0 * to!double(ni_test - n_miss);

    snpInfo ~= SNPINFO(chr,    rs,
                       cM,     b_pos,
                       minor,  major,
                       n_miss, to!double(n_miss) / to!double(ni_test),
                       maf,    ni_test - n_miss,
                       0,      file_pos);
    file_pos++;

    if (to!double(n_miss) / to!double(ni_test) > miss_level) {
      indicator_snp ~= 0;
      continue;
    }

    if ((maf < maf_level || maf > (1.0 - maf_level)) && maf_level != -1) {
      indicator_snp ~= 0;
      continue;
    }

    if (flag_poly != 1) {
      indicator_snp ~= 0;
      continue;
    }

    if (hwe_level != 0 && maf_level != -1) {
      if (CalcHWE(n_0, n_2, n_1) < hwe_level) {
        indicator_snp ~=0;
        continue;
      }
    }

    // Filter SNP if it is correlated with W unless W has
    // only one column, of 1s.
    for (size_t i = 0; i < genotype.length; ++i) {
      if (genotype_miss[i] == 1) {
        geno = maf * 2.0;
        genotype[i] = geno;
      }
    }

    double Wtx = vector_ddot(W.elements, genotype);

    v_x = vector_ddot(genotype, genotype);
    v_w = Wtx * Wtx * WtWi;

    //r2_level
    if (W.shape[1] != 1 && v_w / v_x >= r2_level) {
      indicator_snp ~= 0;
      continue;
    }

    indicator_snp ~= 1;
    ns_test++;
    outfile.writeln(line);
  }
}
import std.stdio;
import std.string;
import std.array;
import std.csv;
import std.regex;
import std.getopt;
import std.json;
import std.math;
import std.typecons;
import faster_lmm_d.rqtlreader;
import faster_lmm_d.tsvreader;
import faster_lmm_d.lmm;
import faster_lmm_d.gwas;
import faster_lmm_d.dmatrix;
import faster_lmm_d.optmatrix;
import faster_lmm_d.helpers;
import faster_lmm_d.optimize;
//import gperftools_d.profiler;

void main(string[] args)
{
  // Main routine
  //ProfilerStart();

  string ocontrol, okinship, opheno, ogeno, useBLAS, noBLAS, noCUDA;
  int pheno_column;
  string cmd;

  getopt(args, "control", &ocontrol, "kinship", &okinship, "pheno", &opheno, "geno", &ogeno, "useBLAS", &useBLAS, "noBLAS", &noBLAS, "noCUDA", &noCUDA, "pheno_column", &pheno_column, "cmd", &cmd);

  writeln(cmd);
  JSONValue ctrl;

  if(cmd == "rqtl"){
    writeln("import rqtlreader as reader");
  }
  else{
    writeln("import tsvreader as reader");
  }

  if(ocontrol){
    ctrl = control(ocontrol);//type
    writeln(ctrl);
  }

  if(okinship){
    //string k = reader.kinship(kinship);
    //kinship(kinship);
    writeln("k.shape");
  }
  double[] y;
  string[] ynames;

  if(opheno){
    if(cmd == "rqtl"){
      auto pTuple = pheno(opheno, pheno_column);
      y = pTuple[0];
      ynames = pTuple[1];
    }
    else{
      auto pTuple = tsvpheno(opheno, pheno_column);
      y = pTuple[0];
      ynames = pTuple[1];
    }
    writeln(y.sizeof);
  }

  dmatrix g;
  string[] gnames;

  if(ogeno && cmd != "iterator"){
    genoObj g1;
    if(cmd == "rqtl"){
      g1 = geno(ogeno, ctrl);
    }
    else{
      g1 = tsvgeno(ogeno, ctrl);
    }
    g = g1.geno;
    gnames = g1.gnames;
    writeln(g.shape);
  }

  if(useBLAS){
    bool optmatrixUseBLAS = true;
    writeln(optmatrixUseBLAS);
    writeln("Forcing BLAS support");
  }

  if(noBLAS){
    bool optmatrixUseBLAS = false;
    writeln(optmatrixUseBLAS);
    writeln("Disabling BLAS support");
  }

  if(noCUDA){
    bool cudauseCUDA = false;
    writeln("Disabling CUDA support");
  }

  if(cmd){
    writeln("Error: Run command is missing!");
  }

  //geno_callback("data/small.geno");

  int n;
  int m;

  void check_results(double[] ps, double[] ts){
    writeln(ps.length, "\n", sum(ps));
    double p1 = ps[0];
    double p2 = ps[$-1];
    if(ogeno == "data/small.geno"){
      writeln("Validating results for ", ogeno);
      assert(modDiff(p1,0.7387)<0.001);
      assert(modDiff(p2,0.7387)<0.001);
    }
    if(ogeno == "data/small_na.geno"){
      writeln("Validating results for ", ogeno);
      assert(modDiff(p1,0.062)<0.001);
      assert(modDiff(p2,0.062)<0.001);
    }
    if(ogeno == "data/test8000.geno"){
      writeln("Validating results for ",ogeno);
      assert(std.math.round(sum(ps)) == 4071);
      assert(ps.length == 8000);
    }
    writeln("Run completed");
  }


  // If there are less phenotypes than strains, reduce the genotype matrix

  if(g.shape[0] != y.sizeof){
    writeln("Reduce geno matrix to match phenotype strains");
    writeln("gnames and ynames");
    writeln(gnames);
    writeln(ynames);
    writeln("gnames and ynames");
    int[] gidx = [];
    int index = 0;
    foreach(ind; ynames){
      while(gnames[0] != ind)
      {
        gnames.popFront;
        index++;
      }
      gidx ~= index;
      index++;
      gnames.popFront;
    }
    writeln(gidx);
    dmatrix gTranspose = matrixTranspose(g);
    dmatrix slicedMatrix = sliceDmatrix(gTranspose, gidx);
    writeln(slicedMatrix.shape);
    dmatrix g2 = matrixTranspose(slicedMatrix);
    writeln("geno matrix ",g.shape," reshaped to ",g2.shape);
    g = g2;
  }

  if(cmd == "run"){
    //if options.remove_missing_phenotypes{
    //  raise Exception('Can not use --remove-missing-phenotypes with LMM2')
    //}
    n = cast(int)y.length;
    m = g.shape[1];
    dmatrix k;
    auto gwas = run_gwas("other",n,m,k,y,g); //<--- pass in geno by SNP
    double[] ts = gwas[0];
    double[] ps = gwas[1];
    writeln(ts);
    writeln(ps);
    check_results(ps,ts);
  }
  else if(cmd == "rqtl"){
    //if options.remove_missing_phenotypes{
    //  raise Exception('Can not use --remove-missing-phenotypes with LMM2')
    //}
    n = cast(int)y.length;
    m = g.shape[1];
    dmatrix k;
    auto gwas = run_gwas("other",n,m,k,y,g);
    double[] ts = gwas[0];
    double[] ps = gwas[1];
    writeln(ts);
    writeln(ps);
    check_results(ps,ts);
  }
  else if(cmd == "iterator"){
//     if options.remove_missing_phenotypes:
//          raise Exception('Can not use --remove-missing-phenotypes with LMM2')
//      geno_iterator =  reader.geno_iter(options.geno)
//      ps, ts = gn2_load_redis_iter('testrun_iter','other',k,y,geno_iterator)
//      check_results(ps,ts)
  }
  else{
    writeln("Doing nothing");
  }


  if(ogeno =="data/test8000.geno" && opheno == "data/test8000.pheno"){
    //K = kinship_full(G);
    //auto k1 = std.math.round(K[0][0],4);
    //auto k2 = std.math.round(K2[0][0],4);

    //writeln("Genotype",G.shape, "\n", G);
    //auto K3 = kinship(G);
    //writeln("third Kinship method",K3.shape,"\n",K3);
    //sys.stderr.write(options.geno+"\n");
    //auto k3 = std.math.round(K3[0][0],4);
    //assert(k3 == 1.4352);
  }
  //ProfilerStop();
}

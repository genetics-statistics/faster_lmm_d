import std.stdio;
import std.string;
import std.array;
import std.csv;
import std.regex;
import std.getopt;
import std.json;
import std.math : round;
import std.typecons;
import core.stdc.stdlib : exit;
import std.experimental.logger;

import faster_lmm_d.rqtlreader;
import faster_lmm_d.tsvreader;
import faster_lmm_d.lmm;
import faster_lmm_d.gwas;
import faster_lmm_d.dmatrix;
import faster_lmm_d.optmatrix;
import faster_lmm_d.helpers;
import faster_lmm_d.optimize;
//import gperftools_d.profiler;

// LogLevel logLevel;

void main(string[] args)
{
  // Main routine
  //ProfilerStart();

  string ocontrol, okinship, opheno, ogeno, useBLAS, noBLAS, noCUDA, ologging;
  int pheno_column;
  string cmd;

  globalLogLevel(LogLevel.warning); //default

  getopt(args, "control", &ocontrol, "kinship", &okinship, "pheno", &opheno, "geno", &ogeno, "useBLAS", &useBLAS, "noBLAS", &noBLAS, "noCUDA", &noCUDA, "pheno_column", &pheno_column, "cmd", &cmd, "logging", &ologging);

  trace(cmd);
  JSONValue ctrl;

  if(cmd == "rqtl"){
    trace("import rqtlreader as reader");
  }
  else{
    trace("import tsvreader as reader");
  }

  if(ocontrol){
    ctrl = control(ocontrol);//type
    trace(ctrl);
  }

  if(okinship){
    //string k = reader.kinship(kinship);
    //kinship(kinship);
    // trace("k.shape");
  }

  if(ologging) {
    writeln("Setting logger to " ~ ologging);
    switch (ologging){
      case "debug":
        globalLogLevel(LogLevel.trace);
        break;
      case "info":
        globalLogLevel(LogLevel.info);
        break;
      case "warning":
        globalLogLevel(LogLevel.warning);
        break;
      case "critical":
        globalLogLevel(LogLevel.critical);
        break;
      default:
        assert(false); // should never happen
    }
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
    trace(y.sizeof);
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
    trace(g.shape);
  }

  if(useBLAS){
    bool optmatrixUseBLAS = true;
    trace(optmatrixUseBLAS);
    info("Forcing BLAS support");
  }

  if(noBLAS){
    bool optmatrixUseBLAS = false;
    trace(optmatrixUseBLAS);
    info("Disabling BLAS support");
  }

  if(noCUDA){
    bool cudauseCUDA = false;
    info("Disabling CUDA support");
  }

  //geno_callback("data/small.geno");

  int n;
  int m;

  void check_results(double[] ps, double[] ts){
    trace(ps.length, "\n", sum(ps));
    double p1 = ps[0];
    double p2 = ps[$-1];
    if(ogeno == "data/small.geno"){
      info("Validating results for ", ogeno);
      assert(modDiff(p1,0.7387)<0.001);
      assert(modDiff(p2,0.7387)<0.001);
    }
    if(ogeno == "data/small_na.geno"){
      info("Validating results for ", ogeno);
      assert(modDiff(p1,0.062)<0.001);
      assert(modDiff(p2,0.062)<0.001);
    }
    if(ogeno == "data/test8000.geno"){
      info("Validating results for ",ogeno);
      assert(round(sum(ps)) == 4071);
      assert(ps.length == 8000);
    }
    info("Run completed");
  }


  // If there are less phenotypes than strains, reduce the genotype matrix

  if(g.shape[0] != y.sizeof){
    info("Reduce geno matrix to match phenotype strains");
    trace("gnames and ynames");
    trace(gnames);
    trace(ynames);
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
    trace(gidx);
    dmatrix gTranspose = matrixTranspose(g);
    dmatrix slicedMatrix = sliceDmatrix(gTranspose, gidx);
    trace(slicedMatrix.shape);
    dmatrix g2 = matrixTranspose(slicedMatrix);
    trace("geno matrix ",g.shape," reshaped to ",g2.shape);
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
    trace(ts);
    trace(ps);
    writeln("ps : ",ps[0],",",ps[1],",",ps[2],"...",ps[n-3],",",ps[n-2],",",ps[n-1]);
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
    trace(ts);
    trace(ps);
    writeln("ps : ",ps[0],",",ps[1],",",ps[2],"...",ps[n-3],",",ps[n-2],",",ps[n-1]);
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
    trace("Doing nothing");
  }


  if(ogeno =="data/test8000.geno" && opheno == "data/test8000.pheno"){
    //K = kinship_full(G);
    //auto k1 = std.math.round(K[0][0],4);
    //auto k2 = std.math.round(K2[0][0],4);

    //trace("Genotype",G.shape, "\n", G);
    //auto K3 = kinship(G);
    //trace("third Kinship method",K3.shape,"\n",K3);
    //sys.stderr.write(options.geno+"\n");
    //auto k3 = std.math.round(K3[0][0],4);
    //assert(k3 == 1.4352);
  }
  //ProfilerStop();
}

import std.stdio;
import std.string;
import std.array;
import std.csv;
import std.regex;
import std.getopt;
import std.json;
import std.math : round;
import std.typecons;
import std.exception;
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

void printUsage() {
    stderr.writeln("faster_lmm_d ");
    stderr.writeln();
    stderr.writeln("Usage: faster_lmm_d [args...]");
    stderr.writeln("Common options:");
    stderr.writeln();
    stderr.writeln("   --kinship","                  ","Kinship file format 1.0");
    stderr.writeln("   --pheno","                    ","Phenotype file format 1.0");
    stderr.writeln("   --pheno-column","             ","pheno_column (default = 0)");
    stderr.writeln("   --geno","                     ","Genotype file format 1.0");
    stderr.writeln("   --blas","                     ","Use BLAS instead of MIR-GLAS matrix multiplication");
    stderr.writeln("   --no-blas","                  ","Disable BLAS support");
    stderr.writeln("   --no-cuda","                  ","Disable CUDA support");
    stderr.writeln("   --control","                  ","R/qtl control file");
    stderr.writeln("   --cmd","                      ","command  = run|rqtl");
    stderr.writeln("   --logging","                  ","set logging  = debug|info|warning|critical");
    stderr.writeln("   --help","                     ","");
    stderr.writeln();
    stderr.writeln("Leave bug reports and feature requests at");
    stderr.writeln("https://github.com/prasunanand/faster_lmm_d/issues");
    stderr.writeln();
}

void main(string[] args)
{
  //ProfilerStart();

  string option_control, option_kinship, option_pheno, option_geno, useBLAS, noBLAS, noCUDA, option_logging;
  bool option_help = false;
  int option_pheno_column;
  string cmd;

  globalLogLevel(LogLevel.warning); //default

  getopt(
    args,
    "control", &option_control,
    "kinship", &option_kinship,
    "pheno", &option_pheno,
    "pheno-column", &option_pheno_column,
    "geno", &option_geno,
    "blas", &useBLAS,
    "no-blas", &noBLAS,
    "no-cuda", &noCUDA,
    "cmd", &cmd,
    "logging", &option_logging,
    "help", &option_help
  );

  if(option_help){
    printUsage();
    exit(0);
  }

  trace(cmd);
  JSONValue ctrl;

  if(option_control){
    ctrl = control(option_control);//type
    trace(ctrl);
  }

  if(option_logging) {
    writeln("Setting logger to " ~ option_logging);
    switch (option_logging){
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

  if(option_pheno){
    if(cmd == "rqtl"){
      auto pTuple = pheno(option_pheno, option_pheno_column);
      y = pTuple[0];
      ynames = pTuple[1];
    }
    else{
      auto pTuple = tsvpheno(option_pheno, option_pheno_column);
      y = pTuple[0];
      ynames = pTuple[1];
    }
    trace(y.sizeof);
  }

  dmatrix g;
  string[] gnames;

  if(option_geno && cmd != "iterator"){
    genoObj g1;
    if(cmd == "rqtl"){
      g1 = geno(option_geno, ctrl);
    }
    else{
      g1 = tsvgeno(option_geno, ctrl);
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

  void check_results(double[] ps, double[] ts){
    trace(ps.length, "\n", sum(ps));
    double p1 = ps[0];
    double p2 = ps[$-1];
    if(option_geno == "data/small.geno"){
      info("Validating results for ", option_geno);
      enforce(modDiff(p1,0.7387)<0.001);
      enforce(modDiff(p2,0.7387)<0.001);
    }
    if(option_geno == "data/small_na.geno"){
      info("Validating results for ", option_geno);
      enforce(modDiff(p1,0.062)<0.001);
      enforce(modDiff(p2,0.062)<0.001);
    }
    if(option_geno == "data/test8000.geno"){
      info("Validating results for ",option_geno," ",sum(ps));
      enforce(round(sum(ps)) == 4071);
      enforce(ps.length == 8000);
    }
    info("Run completed");
  }

  // ---- If there are less phenotypes than strains, reduce the genotype matrix:
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

  int n = cast(int)y.length;
  int m = g.shape[1];
  dmatrix k;
  auto gwas = run_gwas("other",n,m,k,y,g);
  double[] ts = gwas[0];
  double[] ps = gwas[1];
  trace(ts);
  trace(ps);
  writeln("ps : ",ps[0],",",ps[1],",",ps[2],"...",ps[n-3],",",ps[n-2],",",ps[n-1]);
  check_results(ps,ts);

  //ProfilerStop();
}

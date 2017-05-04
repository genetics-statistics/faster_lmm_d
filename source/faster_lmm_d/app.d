/*
   This code is part of faster_lmm_d and published under the GPLv3
   License (see LICENSE.txt)

   Copyright © 2017 Prasun Anand & Pjotr Prins
*/

import core.stdc.stdlib : exit;
import std.array;
import std.conv;
import std.exception;
import std.experimental.logger;
import std.getopt;
import std.json;
import std.math : round;
import std.stdio;
import std.algorithm : countUntil;

import faster_lmm_d.dmatrix;
import faster_lmm_d.gwas;
import faster_lmm_d.helpers;
import faster_lmm_d.lmm;
import faster_lmm_d.optmatrix;
import faster_lmm_d.rqtlreader;
import faster_lmm_d.tsvreader;

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
    exit(0);
}

void main(string[] args)
{
  //ProfilerStart();

  string cmd, option_control, option_kinship, option_pheno, option_geno, useBLAS, noBLAS, noCUDA, option_logging;
  bool option_help = false;
  ulong option_pheno_column;

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

  if(option_help || !cmd) printUsage();

  trace(cmd);

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

  if(option_logging) {
    stderr.writeln("Setting logger to " ~ option_logging);
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

  // ---- Control
  JSONValue ctrl;
  if(option_control){
    ctrl = control(option_control);//type
    trace(ctrl);
  }

  // ---- Phenotypes
  double[] y, y_temp;
  string[] phenotypes;

  if(option_pheno){
    if(cmd == "rqtl"){
      auto pTuple = pheno(option_pheno, option_pheno_column);
      y = pTuple[0];
      phenotypes = pTuple[1];
    }
    else{
      auto pTuple = tsvpheno(option_pheno, option_pheno_column);
      y = pTuple[0];
      phenotypes = pTuple[1];
    }
    trace("y.size=",y.sizeof);
  }

  // ---- Genotypes
  dmatrix g;
  string[] gnames;
  auto g1 = (cmd == "rqtl" ? geno(option_geno, ctrl) : tsvgeno(option_geno, ctrl ));
  g = g1.geno;
  gnames = g1.gnames;
  auto ynames = g1.ynames;
  trace(g.shape);

  // ---- If there are less phenotypes than strains, reduce the genotype matrix:
  if(g.shape[0] != y.sizeof){
    info("Reduce geno matrix to match # strains in phenotype");
    trace("gnames and phenotypes");
    trace("gnames=",gnames);
    trace("ynames=",ynames);
    ulong[] gidx = [];
    ulong index = 0;

    foreach(i, ind; phenotypes){
      auto a = countUntil(ynames, ind);
      if(a != -1){
        gidx ~= a;
        y_temp ~= y[i];
      }
    }
    y = y_temp;

    dmatrix gTranspose = matrixTranspose(g);
    dmatrix slicedMatrix = sliceDmatrix(gTranspose, gidx);
    trace(slicedMatrix.shape);
    dmatrix g2 = matrixTranspose(slicedMatrix);
    trace("geno matrix ",g.shape," reshaped to ",g2.shape);
    g = g2;
  }

  // ---- Run GWAS
  auto n = y.length;
  auto m = g.shape[1];
  dmatrix k;
  auto gwas = run_gwas(n,m,k,y,g);
  double[] ts = gwas[0];
  double[] p_values = gwas[1];
  trace(ts);
  trace(p_values);
  log("p_values : ",p_values[0],",",p_values[1],",",p_values[2],"...",p_values[n-3],",",p_values[n-2],",",p_values[n-1]);

  void check_results(double[] p_values, double[] ts){
    trace(p_values.length, "\n", sum(p_values));
    double p1 = p_values[0];
    double p2 = p_values[$-1];
    if(option_geno == "data/small.geno"){
      info("Validating results for ", option_geno);
      enforce(modDiff(p1,0.7141)<0.001);
      enforce(modDiff(p2,0.7141)<0.001);
    }
    if(option_geno == "data/small_na.geno"){
      info("Validating results for ", option_geno);
      enforce(modDiff(p1,0)<0.001);
      enforce(modDiff(p2,0)<0.001);
    }
    if(option_geno == "data/genenetwork/BXD.csv"){
      info("Validating results for ", option_geno);
      enforce(round(sum(p_values)) == 1901);
      enforce(p_values.length == 3811,"size is " ~ to!string(p_values.length));
    }
    if(option_geno == "data/test8000.geno"){
      info("Validating results for ",option_geno," ",sum(p_values));
      enforce(round(sum(p_values)) == 4069);
      enforce(p_values.length == 8000);
    }
    info("Run completed");
  }

  check_results(p_values,ts);

  writeln("Marker\tP-value\tt-test");
  foreach(i, p ; p_values) {
    writeln(gnames[i],"\t",p,"\t",ts[i]);
  }
  //ProfilerStop();
}

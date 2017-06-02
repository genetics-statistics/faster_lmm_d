/*
   This code is part of faster_lmm_d and published under the GPLv3
   License (see LICENSE.txt)

   Copyright © 2017 Prasun Anand & Pjotr Prins
*/

import core.stdc.stdlib : exit;
import std.algorithm;
import std.array;
import std.conv;
import std.exception;
import std.experimental.logger;
import std.getopt;
import std.json;
import std.math : round;
import std.stdio;

import faster_lmm_d.dmatrix;
import faster_lmm_d.gwas;
import faster_lmm_d.helpers : modDiff;
import faster_lmm_d.memory;
import faster_lmm_d.optmatrix;
import faster_lmm_d.output;
import faster_lmm_d.rqtlreader;
import faster_lmm_d.tsvreader;

version(CUDA) {
  import faster_lmm_d.cuda : cuda_init, cuda_destroy;
}

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
  check_memory("App: Start");

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
    bool optmatrix_use_BLAS = true;
    trace(optmatrix_use_BLAS);
    info("Forcing BLAS support");
  }

  if(noBLAS){
    bool optmatrix_use_BLAS = false;
    trace(optmatrix_use_BLAS);
    info("Disabling BLAS support");
  }

  if(noCUDA){
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

  version(CUDA) {
    cuda_init();
    scope(exit) cuda_destroy();
  }
  // ---- Phenotypes
  double[] pheno_vector;

  check_memory("App: Phenotypes");
  auto pTuple = ( cmd == "rqtl" ? pheno(option_pheno, option_pheno_column) : tsvpheno(option_pheno, option_pheno_column ));
  const double[] y = pTuple[0];
  auto phenotypes = pTuple[1];
  trace("y.size=",y.sizeof);

  // ---- Genotypes
  DMatrix geno_matrix;

  auto g1 = ( cmd == "rqtl" ? geno(option_geno, ctrl) : tsvgeno(option_geno, ctrl ));
  auto g = g1.geno;
  auto gnames = g1.gnames;
  auto ynames = g1.ynames;
  trace(g.shape);

  // ---- If there are less phenotypes than strains, reduce the genotype matrix:
  check_memory("App: reduce genotype matrix");
  if(g.rows != y.sizeof){
    info("Reduce geno matrix to match # strains in phenotype");
    trace("gnames and phenotypes");
    pretty_print("gnames",gnames);
    pretty_print("ynames",ynames);
    ulong[] gidx = [];
    ulong index = 0;

    foreach(i, ind; phenotypes){
      auto a = countUntil(ynames, ind);
      if(a != -1){
        gidx ~= a;
        pheno_vector ~= y[i];
      }
    }

    DMatrix g_transposed = slow_matrix_transpose(g);
    DMatrix sliced_mat = slice_dmatrix(g_transposed, gidx);
    trace(sliced_mat.shape);
    geno_matrix = slow_matrix_transpose(sliced_mat);
    trace("geno matrix ", g.shape, " reshaped to ", geno_matrix.shape);
  }

  // ---- Run GWAS
  check_memory("App: run GWAS");
  immutable m_items n = pheno_vector.length;
  immutable m_items m = geno_matrix.m_geno;
  DMatrix k;
  auto tstats = run_gwas(n,m,k,cast(immutable)pheno_vector, geno_matrix);
  auto p_values = map!"a.p_value"(tstats);
  pretty_print("p_values",p_values.array);

  void check_results(T)(T p_values){
    trace(p_values.length, "\n", sum(p_values));
    double p1 = p_values[0];
    double p2 = p_values[$-1];
    if(option_geno == "data/small.geno"){
      info("Validating results for ", option_geno);
      enforce(modDiff(p1,0.738682)<0.001);
      enforce(modDiff(p2,0.738682)<0.001);
    }
    if(option_geno == "data/small_na.geno"){
      info("Validating results for ", option_geno);
      enforce(modDiff(p1, 0.0620106)<0.001);
      enforce(modDiff(p2, 0.0620106)<0.001);
    }
    if(option_geno == "data/genenetwork/BXD.csv"){
      info("Validating results for ", option_geno);
      enforce(round(sum(p_values),"Got ",to!string(p_values)) == 1922);
      enforce(p_values.length == 3811,"size is " ~ to!string(p_values.length));
      enforce(round(p_values[3]*10000) == 8073,"P-value[3] " ~ to!string(round(p_values[3]*10000)));
    }
    if(option_geno == "data/test8000.geno"){
      info("Validating results for ",option_geno," ",sum(p_values));
      enforce(round(sum(p_values)) == 4070);
      enforce(p_values.length == 8000);
      enforce(round(p_values[3]*10000) == 7503,"P-value[3] " ~ to!string(round(p_values[3]*10000)));
    }
    info("Run completed");
  }

  check_results(p_values);

  writefln("%20s\t%9s\t%9s\t%9s\t%9s", "Marker","P-value","t-test","LOD","LRS");
  foreach(i, ts; tstats) {
    writefln("%20s\t%9.5f\t%9.5f\t%9.5f\t%9.5f", gnames[i],ts.p_value,ts.ts,ts.lod,ts.lod/4.61);
  }
  check_memory("Exit");
  //ProfilerStop();
}

import std.stdio;
import std.string;
import std.array;
import std.csv;
import std.regex;
import dyaml.all;
import std.getopt;
import simplelmm.rqtlreader;

void main(string[] args)
{
  string control;
  string okinship;
  string opheno;
  string geno;
  string useBLAS;
  string noBLAS;
  string noCUDA;
  int pheno_column;
  string cmd;

  getopt(args, "control", &control, "kinship", &okinship, "opheno", &opheno, "geno", &geno, "useBLAS", &useBLAS, "noBLAS", &noBLAS, "noCUDA", &noCUDA, "pheno_column", &pheno_column, "cmd", &cmd);

  writeln(cmd);

  if(cmd == "rqtl"){
    writeln("import rqtlreader as reader");
  }
  else{
    writeln("import tsvreader as reader");
  }

  if(control){
    //string ctrl = reader.control(control);//type
    writeln("ctrl");
  }

  if(okinship){
    //string k = reader.kinship(kinship);
    //kinship(kinship);
    writeln("k.shape");
  }

  if(opheno){
    string y = "y";
    string ynames = "ynames";
    //y,
    auto ynam = pheno(opheno, pheno_column);
    writeln("y.shape");
  }

  if(geno && cmd != "iterator"){
    string g = "reader.geno";
    string gnames = "reader.gnames";
    //g,gnames = reader.geno(options.geno, ctrl);
    writeln("g.shape");
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

  //int a = pheno("./data/rqtl/iron_covar.csv");
 // writeln(a);
	//writeln("Edit source/app.d to start your project.");


  //Node root = control("./source/input.yaml");
  //writeln("The answer is ", root["crosstype"].as!int);

  //kinship("./data/rqtl/iron_covar.csv");

 // //Dump the loaded document to output.yaml.
 // Dumper("output.yaml").dump(root);
}
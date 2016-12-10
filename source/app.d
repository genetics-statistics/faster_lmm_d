import std.stdio;
import std.string;
import std.array;
import std.csv;
import std.regex;
import dyaml.all;
import std.getopt;
import std.json;
import simplelmm.rqtlreader;

void main(string[] args)
{
  string ocontrol;
  string okinship;
  string opheno;
  string ogeno;
  string useBLAS;
  string noBLAS;
  string noCUDA;
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
    writeln(".....///////////////////////////////..................");
    writeln(ctrl);
    writeln(".....///////////////////////////////..................");
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

  if(ogeno && cmd != "iterator"){
    string g = "reader.geno";
    string gnames = "reader.gnames";
    //g,
    int gname = geno(ogeno, ctrl);
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
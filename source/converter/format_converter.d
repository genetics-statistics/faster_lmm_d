/*
   This code is part of faster_lmm_d and published under the GPLv3
   License (see LICENSE.txt)

   Copyright © 2017 Prasun Anand & Pjotr Prins
*/

import std.conv;
import std.file;
import std.stdio;
import std.string;

void call(string[] args){
  string option =  args[1];
  if(option == "geno_tsv_to_rqtl"){
    geno_tsv_to_rqtl(args[2]);
  }
  else if(option == "pheno_tsv_to_rqtl"){
    pheno_tsv_to_rqtl(args[2]);
  }
  else{
    writeln("option unrecognized");
  }
}

void geno_tsv_to_rqtl(string filename){
  string input = (to!string(std.file.read(filename))).strip();
  string[] tsv = input.split("\n");
  int individuals = to!int(tsv[1].split(" ")[$-1]);
  int snps = to!int(tsv[2].split(" ")[$-1]);
  write("id");
  for(int i = 0; i < individuals; i++){
    write(",",i+1);
  }
  write("\n");
  foreach(row; tsv[5..$]){
    string[] vec = row.split("\t");
    string[] a = [vec[0]];
    foreach(dchar item; vec[1]){
      a ~= to!string(item);
    }
    string line = a.join(",");
    writeln(line);
  }
}

void pheno_tsv_to_rqtl(string filename){
  string input = (to!string(std.file.read(filename))).strip();
  string[] tsv = input.split("\n");
  int phenotypes = to!int(tsv[2].split(" ")[$-1]);
  write("id");
  for(int i = 0; i < phenotypes; i++){
    write(",",i+1);
  }
  write("\n");
  foreach(row; tsv[4..$]){
    writeln(row.split("\t").join(","));
  }
}

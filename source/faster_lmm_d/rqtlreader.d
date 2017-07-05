/*
   This code is part of faster_lmm_d and published under the GPLv3
   License (see LICENSE.txt)

   Copyright © 2017 Prasun Anand & Pjotr Prins
*/

module faster_lmm_d.rqtlreader;

import std.csv;
import std.conv;
import std.experimental.logger;
import std.file;
import std.json;
import std.regex;
import std.typecons;
import std.string;

import dyaml.all;

import faster_lmm_d.dmatrix;

JSONValue control(const string fn){
  string input = to!string(std.file.read(fn));
  JSONValue j = parseJSON(input);
  return j;
}

auto pheno(const string fn, const ulong p_column= 1){
  Regex!char Pattern = regex("\\.json$", "i");
  double[] y;
  string[] phenotypes;

  if(!match(fn, Pattern).empty)
  {
    Node gn2_pheno = Loader(fn).load();
    foreach(Node strain; gn2_pheno){
      y ~= ( strain[2] == "NA" ? double.nan : strain[2].as!double);
      phenotypes ~= strain[1].as!string;
    }
  }else{
    string input = (to!string(std.file.read(fn))).strip();
    string[] tsv = input.split("\n");
    foreach(row; tsv[1..$]){
      auto vec = row.split(",");
      y ~= ( vec[p_column] == "NA" ? double.nan : to!double(vec[p_column] ));
      phenotypes ~= vec[0];
    }
  }
  return Tuple!(const double[], immutable(string[]))(y, cast(immutable)phenotypes);
}

GenoObj geno(const string fn, JSONValue ctrl){

  trace("in geno function");
  const(JSONValue)* na_strings = "na.strings" in ctrl;
  if(na_strings  == null){
    ctrl["na.strings"] = JSONValue(["-" ,"NA"]);
  }
  int[string] hab_mapper;
  int idx = 0;

  foreach( key, value; ctrl["genotypes"].object){
    string a  = to!string(key);
    string b = to!string(value);
    int c = to!int(b);
    hab_mapper[a] = c;
    idx++;
  }


  assert(idx == 3);
  double[] faster_lmm_d_mapper = [double.nan, 0.0, 0.5, 1.0];

  foreach( JSONValue key; ctrl["na.strings"].array){
    idx += 1;
    hab_mapper[key.str] = idx;
    faster_lmm_d_mapper ~= double.nan;
  }

  log("hab_mapper", hab_mapper);
  log("faster_lmm_d_mapper", faster_lmm_d_mapper);
  string input = to!string(std.file.read(fn));
  auto tsv = csvReader!(string, Malformed.ignore)(input, null);

  auto ynames = cast(immutable(string[]))tsv.header[1..$];

  DMatrix geno;
  string[] gnames = [];
  auto rowCount = 0;
  auto colCount = ynames.length;

  double[][] matrix = [];
  foreach(row; tsv){
    auto r = new double[colCount];
    gnames ~= row.front;
    row.popFront(); // remove header line
    auto count = 0;
    foreach(item; row){
      r[count++] = faster_lmm_d_mapper[hab_mapper[item]];
    }
    geno.elements ~= r;
    rowCount++;
  }
  geno.elements.length = rowCount * colCount;

  geno.shape = [rowCount, colCount];

  DMatrix genotype_matrix;
  if(const(JSONValue)* p = "geno_transposed" in ctrl){
    if(to!bool(p.toString())){
      genotype_matrix = geno;
    }
  }
  else{
    genotype_matrix= geno.T;
  }

  GenoObj geno_obj = GenoObj(genotype_matrix, cast(immutable)gnames, ynames);
  info("Genotype Matrix created");
  return geno_obj;
}

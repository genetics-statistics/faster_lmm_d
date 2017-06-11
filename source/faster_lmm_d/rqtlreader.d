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

import dyaml.all;

import faster_lmm_d.dmatrix;

JSONValue control(const string fn){
  string input = to!string(std.file.read(fn));
  JSONValue j = parseJSON(input);
  return j;
}

auto pheno(const string fn, const ulong p_column= 0){
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
  }
  return Tuple!(const double[], immutable(string[]))(y, cast(immutable)phenotypes);
}

GenoObj geno(const string fn, JSONValue ctrl){

  trace("in geno function");
  //FIXME
  string s = `{"-" : "0","NA": "0", "U": "0"}`;
  ctrl["na-strings"] = parseJSON(s);
  int[string] hab_mapper;
  int idx = 0;

  foreach( key, value; ctrl["genotypes"].object){
    string b = to!string(value);
    int c = to!int(b);
    hab_mapper[key] = c;
    idx++;
  }

  assert(idx == 3);
  double[] faster_lmm_d_mapper = [double.nan, 0.0, 0.5, 1.0];

  foreach( key,value; ctrl["na-strings"].object){
    idx += 1;
    hab_mapper[to!string(key)] = idx;
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

  double[][] lines = [];
  m_items pos = 0;
  foreach(row; tsv){
    geno.elements.length += colCount;
    gnames ~= row.front;
    row.popFront();
    foreach(item; row){
      geno.elements[pos++] = faster_lmm_d_mapper[hab_mapper[item]];
    }
    rowCount++;
  }
  geno.shape = [rowCount, colCount];
  GenoObj geno_obj = GenoObj(geno, cast(immutable)gnames, ynames);

  info("Genotype Matrix created");
  return geno_obj;
}

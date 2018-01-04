/*
   This code is part of faster_lmm_d and published under the GPLv3
   License (see LICENSE.txt)

   Copyright © 2017 - 2018 Prasun Anand & Pjotr Prins
*/

module faster_lmm_d.tsvreader;

import std.csv;
import std.conv;
import std.experimental.logger;
import std.file;
import std.json;
import std.regex;
import std.string;
import std.typecons;

import faster_lmm_d.dmatrix;
import faster_lmm_d.optmatrix;

auto tsvpheno(const string fn, const ulong p_column= 0){
	trace("In tsvpheno");
	double[] y;
  string[] phenotypes;
  string input = to!string(std.file.read(fn));

  string[] lines = input.split("\n");
  assert(lines[0] == "# Phenotype format version 1.0", "test for Phenotype format version 1.0");
  lines = lines[4..$];
  foreach(line; lines){
  	if(line != ""){
  		string[] row = line.split("\t");
      y ~= ( row[1] == "NA" ? double.nan : to!double(row[1]) );
	  	phenotypes ~= to!string(row[0]);
  	}

  }
  return Tuple!(const double[], immutable(string[]))(y, cast(immutable)phenotypes);
}// # FIXME: column not used

GenoObj tsvgeno(const string fn, JSONValue ctrl){

  trace("in geno function");
  string s = `{"A":0,"H":1,"B":2,"-":3}`;
  ctrl["na-strings"] = parseJSON(s);

  int[dchar] hab_mapper;

  foreach( key, value; ctrl["na-strings"].object){
    dchar a  = to!dchar(key);
    string b = to!string(value);
    int c = to!int(b);
    hab_mapper[a] = c;
  }

  double[] faster_lmm_d_mapper = [ 0.0, 0.5, 1.0, double.nan,];

  log("hab_mapper", hab_mapper);
  log("faster_lmm_d_mapper", faster_lmm_d_mapper);

  string input = to!string(std.file.read(fn));
  string[] rows = input.split("\n");

  immutable(string[]) ynames = rows[4].split("\t");

  auto rowCount = rows.length - 6;
  auto colCount = ynames.length - 1;

  DMatrix geno;
  string[] gnames = [];
  geno.elements.length = rowCount * colCount;

  m_items pos = 0;
  foreach(line; rows[5..$]){
    if(line != ""){
      string[] row = line.split("\t");
      gnames ~= row[0];
      foreach(dchar item; row[1]){
        geno.elements[pos++] = faster_lmm_d_mapper[hab_mapper[item]];
      }
    }
  }

  geno.shape = [rowCount, colCount];
  GenoObj geno_obj = GenoObj(geno, cast(immutable)gnames,ynames[1..$]);

  info("Genotype Matrix created");
  return geno_obj;
}

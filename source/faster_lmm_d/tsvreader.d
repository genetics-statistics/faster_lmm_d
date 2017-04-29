/*
   This code is part of faster_lmm_d and published under the GPLv3
   License (see LICENSE.txt)

   Copyright © 2017 Prasun Anand & Pjotr Prins
*/

module faster_lmm_d.tsvreader;

import std.stdio;
import std.json;
import std.conv;
import std.range;
import std.file;
import std.typecons;
import std.string;
import std.array;
import std.csv;
import std.regex;
import std.experimental.logger;

import faster_lmm_d.dmatrix;
import faster_lmm_d.optmatrix;

auto tsvpheno(string fn, int p_column= 0){
	trace("In tsvpheno");
	double[] y;
  string[] ynames;
  //ynames = None
  string input = cast(string)std.file.read(fn);

  string[] lines = input.split("\n");
  //with open(fn,'r') as tsvin:
  assert(lines[0] == "# Phenotype format version 1.0");
  lines = lines[4..$];
  foreach(line; lines){
  	if(line != ""){
  		string[] row = line.split("\t");
	  	y ~=  to!double(row[1]);// <--- slow
	  	ynames ~= to!string(row[0]);
  	}

  }
  return Tuple!(double[], string[])(y, ynames);
}// # FIXME: column not used

genoObj tsvgeno(string fn, JSONValue ctrl){

  trace("in geno function");
  //log(ctrl["genotypes"].object);
  //string ptr = ("na-strings" in ctrl.object).str;
  //log(ctrl.object);
  //string s = `{"-" : "0","NA": "0"}`;
  //FIXME
  string s = `{"A":0,"H":1,"B":2,"-":3}`;
  ctrl["na-strings"] = parseJSON(s);
  //log(ctrl.object);
  int[string] hab_mapper;
  int idx = 0;

  foreach( key, value; ctrl["na-strings"].object){
    string a  = to!string(key);
    string b = to!string(value);
    int c = to!int(b);
    hab_mapper[a] = c;
    log(hab_mapper);
    //log(a);
    log(b);

    idx++;
  }
  //log(hab_mapper);
  double[] faster_lmm_d_mapper = [ 0.0, 0.5, 1.0, double.nan,];
  //foreach(s; ctrl["na.strings"]){

  log("hab_mapper", hab_mapper);
  log("faster_lmm_d_mapper", faster_lmm_d_mapper);
  //log(fn);

  string input = cast(string)std.file.read(fn);
  string[] rows = input.split("\n");
  rows = rows[4..$];
  //log(tsv.header[1..$]);
  string[] gnames = rows[0].split("\t");
  gnames = gnames[1..$];
  double[] gs2;
  int rowCount = 0;
  int colCount;
  string[] gs;
  int allal= 0;
  foreach(line; rows[1..$]){
  	if(line != ""){
	  	string[] row = line.split("\t");
	    string id = row[0];
	    //log(id);
	    //gs = [];
	    colCount = 0;
	    foreach(dchar item; row[1]){
	      gs ~= to!string(item);
	      gs2 ~= faster_lmm_d_mapper[hab_mapper[to!string(item)]];
	      colCount++;
	      allal++;
	    }
	    rowCount++;
	  }
  }
  info("MATRIX CREATED");
  return genoObj(dmatrix([rowCount, colCount], gs2), gnames);
}

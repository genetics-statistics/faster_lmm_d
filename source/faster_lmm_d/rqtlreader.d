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
import std.string;
import std.stdio;
import std.typecons;

import yaml;

import faster_lmm_d.dmatrix;

JSONValue control(const string fn){
  string input = to!string(std.file.read(fn));
  JSONValue j = parseJSON(input);
  return j;
}

auto pheno(const string fn, const ulong p_column){
  ulong pheno_column = p_column + 1;
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
      y ~= ( vec[pheno_column] == "NA" ? double.nan : to!double(vec[pheno_column] ));
      phenotypes ~= vec[0];
    }
  }
  return Tuple!(const double[], immutable(string[]))(y, cast(immutable)phenotypes);
}

GenoObj geno(const string fn, JSONValue ctrl){

  trace("in geno function");
  int[string] hab_mapper;
  double[] faster_lmm_d_mapper;

  if(ctrl.type == JSON_TYPE.OBJECT){
    const(JSONValue)* na_strings = "na.strings" in ctrl;
    if(na_strings  == null){
      ctrl["na.strings"] = JSONValue(["-" ,"NA"]);
    }

    int idx = 0;

    foreach( key, value; ctrl["genotypes"].object){
      string a  = to!string(key);
      if(value.type() == JSON_TYPE.INTEGER){
        hab_mapper[a] = to!int(value.integer);
      }
      else{
        string b = value.str;
        int c = to!int(b);
        hab_mapper[a] = c;
      }

      idx++;
    }

    assert(idx == 3);
    faster_lmm_d_mapper = [double.nan, 0.0, 0.5, 1.0];

    foreach( JSONValue key; ctrl["na.strings"].array){
      idx += 1;
      hab_mapper[key.str] = idx;
      faster_lmm_d_mapper ~= double.nan;
    }

  }
  else{
    trace("in geno function");
    string s = `{"A":0,"H":1,"B":2,"-":3}`;
    ctrl["na-strings"] = parseJSON(s);


    foreach( key, value; ctrl["na-strings"].object){
      string a  = key;
      string b = to!string(value);
      int c = to!int(b);
      hab_mapper[a] = c;
    }

    faster_lmm_d_mapper = [ 0.0, 0.5, 1.0, double.nan,];
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

struct Covariate{
  string id;
  string[string] items;
}

DMatrix covar(const string fn, JSONValue ctrl){

  Covariate[] covars;
  int num_covars = 0;
  foreach(key, value; ctrl.object){
    writeln(key.str);
    if (value.type() == JSON_TYPE.OBJECT){
      const(JSONValue)* p = "covar" in value.object;
      if(p == null){
        continue;
      }else{
        num_covars += 1;
        Covariate c;
        c.id = p.str;
        foreach(value_key, value_value; value.object){
          if(value_key.str != "covar"){
            if(value_value.type() == JSON_TYPE.STRING){
              c.items[value_key.str] = value_value.str;
            }
            else{
              c.items[value_key.str] = to!string(value_value);
            }
          }
        }
        covars ~= c;
      }
    }

  }
  writeln(covars);
  double[] covar_elements;
  string input = (to!string(std.file.read(fn))).strip();
  string[] tsv = input.split("\n");
  foreach(row; tsv[1..$]){
    auto vec = row.split(",");
    covar_elements ~= to!double(vec[0]);
    covar_elements ~= (vec[1] == "male") ? 1 : 0;
    //covar_elements ~= (vec[2] == "(BxS)x(BxS)") ? 1 : 0;
  }

  return DMatrix([covar_elements.length/num_covars, num_covars], covar_elements);
}

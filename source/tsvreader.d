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

import faster_lmm_d.dmatrix;
import faster_lmm_d.optmatrix;

void tsvpheno(string fn,  ref double[] y, ref string[] ynames, int p_column= 0){
	writeln("In tsvpheno");
	double[] Y1;
  //ynames = None
  string input = cast(string)std.file.read(fn);

  string[] lines = input.split("\n");
  //with open(fn,'r') as tsvin:
  assert(lines[0] == "# Phenotype format version 1.0");
  lines = lines[4..$];
  foreach(line; lines){
  	if(line != ""){
  		string[] row = line.split("\t");
	  	Y1 ~=  to!double(row[1]);// <--- slow
	  	ynames ~= to!string(row[0]);
  	}
  	
  }
  y = Y1;
  //writeln("ynames");
  //writeln(y,ynames);
}// # FIXME: column not used

void tsvgeno(string fn, JSONValue ctrl, ref dmatrix g, ref string[] gnames){

  writeln("in geno function");
  //writeln(ctrl["genotypes"].object);
  //string ptr = ("na-strings" in ctrl.object).str;
  //writeln(ctrl.object);
  //string s = `{"-" : "0","NA": "0"}`;
  //FIXME
  string s = `{"A":0,"H":1,"B":2,"-":3}`;
  ctrl["na-strings"] = parseJSON(s);
  //writeln(ctrl.object);
  int[string] hab_mapper;
  int idx = 0;

  foreach( key, value; ctrl["na-strings"].object){
    string a  = to!string(key);
    string b = to!string(value);
    int c = to!int(b);
    hab_mapper[a] = c;
    writeln(hab_mapper);
    //writeln(a);
    writeln(b);

    idx++;
  }
  //writeln(hab_mapper);
  double[] faster_lmm_d_mapper = [ 0.0, 0.5, 1.0, double.nan,];
  //foreach(s; ctrl["na.strings"]){

  writeln("hab_mapper", hab_mapper);
  writeln("faster_lmm_d_mapper", faster_lmm_d_mapper);
  //writeln(fn);

  string input = cast(string)std.file.read(fn);
  string[] rows = input.split("\n");
  rows = rows[4..$];
  //writeln(tsv.header[1..$]);
  gnames = rows[0].split("\t");
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
	    //writeln(id);
	    //gs = [];
	    colCount = 0;
	    foreach(dchar item; row[1]){
	      gs ~= to!string(item);
	      gs2 ~= faster_lmm_d_mapper[hab_mapper[to!string(item)]];
	      colCount++;
	      allal++;
	    }
	    //writeln(rowCount);
	    //writeln(gs2);
	    rowCount++;


	    //writeln(gs);
	    ////# print id,gs
	    ////# Convert all items to genotype values
	    //gs2 = [faster_lmm_d_mapper[hab_mapper[g]] for g in gs]
	    //core.exception.RangeError@source/rqtlreader.d(137): Range violation
	    //foreach(gval;gs){
	    //  gs2 ~= faster_lmm_d_mapper[hab_mapper[gval]];
	    //}
	    //# print id,gs2
	    //# ns = np.genfromtxt(row[1:])
	    //G1.append(gs2) # <--- slow
	    //G = np.array(G1)
	  }
  }
  writeln("MATRIX CREATED");
  g = dmatrix([rowCount, colCount], gs2);
  //pPrint(g);
    //writeln(y);
  ////# print(row)

  //string* ptr;

  //ptr = ("na.strings" in ctrl);
  //if(ptr !is null && ctrl['geno_transposed']){
  //  //return G,gnames
  //}
  ////return G.T,gnames



  //return 5;
 writeln("leaving geno function");
}

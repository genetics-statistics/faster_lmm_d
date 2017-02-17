module simplelmm.tsvreader;

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
  writeln("ynames");
  writeln(y,ynames);
}// # FIXME: column not used

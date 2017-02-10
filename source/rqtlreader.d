module simplelmm.rqtlreader;

import std.stdio;
import std.string;
import std.array;
import std.csv;
import std.regex;
import dyaml.all;
import std.getopt;
import std.typecons;
import std.json;
import std.conv;
import std.range;
import std.file;

import simplelmm.dmatrix;

JSONValue control(string fn){
  //Node root = Loader(fn).load();
  //writeln("in control function");
  string input = cast(string)std.file.read(fn);
  JSONValue j = parseJSON(input);
  //writeln(j);

  return j;
}

int kinship(string fn){
  auto K1 = [][];
  writeln(fn);
  //string input = cast(string)std.file.read(fn);

  //string[] lines = input.split("\n");

  //int i = 0;
  //foreach(line; lines[3..$])
  //{
  //  writeln(line);
  //  //csvReader!int(lines[i],"\t");
  //  i++;
  //}
  auto file = File(fn);
  //writeln(file.byLine());

  return 1;
}

void pheno(string fn,  ref double[] y, ref string[] ynames, int p_column= 0){
  // read recla_geno.csv
  //writeln(fn);

  Regex!char Pattern = regex("\\.json$", "i");

  if(!match(fn, Pattern).empty)
  {
    Node gn2_pheno = Loader(fn).load();
    foreach(Node strain; gn2_pheno){
      //writeln(strain.as!string);
      y ~= strain[2].as!double;
      ynames ~= strain[1].as!string;
    }
    writeln("interest");
    writeln(y);
    writeln(ynames);
    writeln("interest");
  }


  //string input = cast(string)std.file.read(fn);
  ////auto tsv = csvReader!(string[string])(input, null);
  //auto tsv = csvReader!(string)(input, null);
  //writeln(tsv[0]);
  //auto ynames = tsv[1..$];

  //auto p = ctRegex!(`^.+\.$`);
  //foreach(n;ynames){
  //  assert!match(n,p);
  //}
  //foreach(row; tsv){

  //}
}

void geno(string fn, JSONValue ctrl, ref dmatrix g, ref string[] gnames){

  writeln("in geno function");
  //writeln(ctrl["genotypes"].object);
  //string ptr = ("na-strings" in ctrl.object).str;
  //writeln(ctrl.object);
  //string s = `{"-" : "0","NA": "0"}`;
  //FIXME
  string s = `{"-" : "0","NA": "0", "U": "0"}`;
  ctrl["na-strings"] = parseJSON(s);
  //writeln(ctrl.object);
  int[string] hab_mapper;
  int idx = 0;

  foreach( key, value; ctrl["genotypes"].object){
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
  writeln(idx);
  assert(idx == 3);
  double[] simplelmm_mapper = [double.nan, 0.0, 0.5, 1.0];
  //foreach(s; ctrl["na.strings"]){

  foreach( key,value; ctrl["na-strings"].object){
    idx += 1;
    hab_mapper[to!string(key)] = idx;
    simplelmm_mapper ~= double.nan;
  }
  writeln("hab_mapper", hab_mapper);
  writeln("simplelmm_mapper", simplelmm_mapper);
  //writeln(fn);

  string input = cast(string)std.file.read(fn);
  auto tsv = csvReader!(string, Malformed.ignore)(input, null);

  //writeln(tsv.header[1..$]);
  gnames = tsv.header[1..$];
  double[] gs2;
  int rowCount = 0;
  int colCount;
  string[] gs;
  int allal= 0;
  foreach(row; tsv){
    string id = row.front;
    //writeln(id);
    row.popFront();
    gs = [];
    colCount = 0;
    foreach(item; row){
      gs ~= item;
      gs2 ~= simplelmm_mapper[hab_mapper[item]];
      colCount++;
      allal++;
    }
    //writeln(rowCount);
    //writeln(gs2);
    rowCount++;


    //writeln(gs);
    ////# print id,gs
    ////# Convert all items to genotype values
    //gs2 = [simplelmm_mapper[hab_mapper[g]] for g in gs]
    //core.exception.RangeError@source/rqtlreader.d(137): Range violation
    //foreach(gval;gs){
    //  gs2 ~= simplelmm_mapper[hab_mapper[gval]];
    //}
    //# print id,gs2
    //# ns = np.genfromtxt(row[1:])
    //G1.append(gs2) # <--- slow
    //G = np.array(G1)
  }
  writeln("MATRIX CREATED");
  g = dmatrix([rowCount, colCount], gs2);
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


int geno_callback(string fn){
  int[string] hab_mapper;
  hab_mapper["A"] = 0;
  hab_mapper["H"] = 1;
  hab_mapper["B"] = 2;
  hab_mapper["-"] = 3;
  auto simplelmm_mapper = [ 0.0, 0.5, 1.0, double.nan];

  //  raise "NYI"
  writeln(fn);

  auto file = File(fn); // Open for reading
  assert(file.readln().strip() == "# Genotype format version 1.0");
  file.readln();
  file.readln();
  file.readln();
  file.readln();

  string input = cast(string)std.file.read(fn);
  auto tsv = csvReader!(string, Malformed.ignore)(input, '\t');
  writeln(tsv);
  //      for row in tsv:
  //          id = row[0]
  //          gs = list(row[1])
  //          gs2 = [simplelmm_mapper[hab_mapper[g]] for g in gs]
  //          func(id,gs2)
  return 5;
}

int geno_iter(string fn){
  int[string] hab_mapper;
  hab_mapper["A"] = 0;
  hab_mapper["H"] = 1;
  hab_mapper["B"] = 2;
  hab_mapper["-"] = 3;
  auto simplelmm_mapper = [ 0.0, 0.5, 1.0, double.nan];

  writeln(fn);


  auto file = File(fn); // Open for reading
  assert(file.readln() == "# Genotype format version 1.0");
  file.readln();
  file.readln();
  file.readln();
  file.readln();
  writeln(file);

  string input = cast(string)std.file.read(fn);
  auto tsv = csvReader!(string, Malformed.ignore)(input, '\t');
  writeln(tsv);

  return 5;
}

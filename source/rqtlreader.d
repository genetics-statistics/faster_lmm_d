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

JSONValue control(string fn){
  //Node root = Loader(fn).load();
  //writeln("in control function");
  string input = cast(string)std.file.read(fn);
  JSONValue j = parseJSON(input);
  writeln(j);

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
  writeln(file.byLine());

  return 1;
}

int pheno(string fn, int p_column){
  // read recla_geno.csv
  string[] Y1;
  writeln(fn);

  Regex!char Pattern = regex("\\.json$", "i");

  if(!match(fn, Pattern).empty)
  {
    string[] ynames;
    Node gn2_pheno = Loader(fn).load();
    foreach(Node strain; gn2_pheno){
      writeln(strain.as!string);
      Y1 ~= strain[2].as!string;
      ynames ~= strain[1].as!string;
    }
    writeln(Y1);
    writeln(ynames);
    return 6;
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


  return 5;
}

int geno(string fn, JSONValue ctrl){

  writeln("in geno function");
  writeln(ctrl["genotypes"].object);
  //string ptr = ("na-strings" in ctrl.object).str;
  writeln(ctrl.object);

  string s = `{"-" : "0","NA": "0"}`;
  ctrl["na-strings"] = parseJSON(s);
  writeln(ctrl.object);
  int[string] hab_mapper;
  int idx = 0;

  foreach( key, value; ctrl["genotypes"].object){
    hab_mapper[to!string(key)] = to!int(value.str);
    idx++;
  }
  writeln(hab_mapper);
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

  //string input = cast(string)std.file.read(fn);
  //auto tsv = csvReader!(string)(input, null);

  //auto gnames = tsv[1..$];

  //foreach(row; tsv){
  //  id = row[0]
  //  gs = row[1:]
  //  //# print id,gs
  //  //# Convert all items to genotype values
  //  gs2 = [pylmm_mapper[hab_mapper[g]] for g in gs]
  //  //# print id,gs2
  //  //# ns = np.genfromtxt(row[1:])
  //  //G1.append(gs2) # <--- slow
  //  //G = np.array(G1)
  //}
  ////# print(row)

  //string* ptr;

  //ptr = ("na.strings" in ctrl);
  //if(ptr !is null && ctrl['geno_transposed']){
  //  //return G,gnames
  //}
  ////return G.T,gnames



  return 5;
}


//int geno_callback(){
//  int[string] hab_mapper;
//  hab_mapper["A"] = 0;
//  hab_mapper["H"] = 1;
//  hab_mapper["B"] = 2;
//  hab_mapper["-"] = 3;
//  auto pylmm_mapper = [ 0.0, 0.5, 1.0, float.nan]

//  //  raise "NYI"
//  writeln(fn);

//  string input = cast(string)std.file.read(fn);
//  auto tsv = csvReader!(string)(input, null);
//  //  with open(fn,'r') as csvin:
//  //      assert(csvin.readline().strip() == "# Genotype format version 1.0")
//  //      csvin.readline()
//  //      csvin.readline()
//  //      csvin.readline()
//  //      csvin.readline()
//  //      tsv = csv.reader(csvin, delimiter='\t')
//  //      for row in tsv:
//  //          id = row[0]
//  //          gs = list(row[1])
//  //          gs2 = [pylmm_mapper[hab_mapper[g]] for g in gs]
//  //          func(id,gs2)
//  return 5;
//}

//int geno_iter(){
//  nt[string] hab_mapper;
//  hab_mapper["A"] = 0;
//  hab_mapper["H"] = 1;
//  hab_mapper["B"] = 2;
//  hab_mapper["-"] = 3;
//  auto pylmm_mapper = [ 0.0, 0.5, 1.0, float.nan]

//  writeln(fn);

//  string input = cast(string)std.file.read(fn);
//  auto tsv = csvReader!(string)(input, null);
//  //  with open(fn,'r') as csvin:
//  //      assert(csvin.readline().strip() == "# Genotype format version 1.0")
//  //      csvin.readline()
//  //      csvin.readline()
//  //      csvin.readline()
//  //      csvin.readline()
//  //      tsv = csv.reader(csvin, delimiter='\t')
//  //      for row in tsv:
//  //          id = row[0]
//  //          gs = list(row[1])
//  //          gs2 = [pylmm_mapper[hab_mapper[g]] for g in gs]
//  //          yield (id,gs2)
//  return 5;
//}

import std.stdio;
import std.string;
import std.array;
import dyaml.all;

Node control(string fn){
  Node root = Loader(fn).load();
  return root;
}

int kinship(string fn){
  //K1 = []
  writeln(fn);
  string input = cast(string)std.file.read(fn);
  foreach (line; input.split("\n"))
    writeln(line);
  return 1;
}


void main(string[] argv)
{

  Node root = control("./source/input.yaml");
  //writeln("The answer is ", root["crosstype"].as!int);

  kinship("./data/rqtl/iron_covar.csv");

  //Dump the loaded document to output.yaml.
  Dumper("output.yaml").dump(root);
}
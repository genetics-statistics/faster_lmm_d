module test.geno_generator;

import std.conv;
import std.random;
import std.stdio;

void main(string[] args)
{
  writeln("# Genotype format version 1.0");
  writeln("# Individuals = 1219");
  writeln("# SNPs = 8000");
  writeln("# Encoding = HAB");
  char[] encoding = ['A', 'B', 'H', '-'];
  int side = to!int(args[1]) + 1;
  for(int i = 1; i < side ; i++){
    write("\t");
    write(i);
  }
  write("\n");
  for(int i = 1; i < side ; i++){
    write(i);
    write("\t");
    for(int j = 0; j < side ; j++){
      write(encoding[uniform(0,4)]);
    }
    write("\n");
  }
}
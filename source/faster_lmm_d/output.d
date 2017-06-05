/*
   This code is part of faster_lmm_d and published under the GPLv3
   License (see LICENSE.txt)

   Copyright © 2017 Prasun Anand & Pjotr Prins
*/

module faster_lmm_d.output;

import std.stdio;

import faster_lmm_d.dmatrix;

void print(T...)(T args) {
  stderr.write(args);
}

void println(T...)(T args) {
  stderr.writeln(args);
}


void pretty_print(string msg, const DMatrix input) {
  m_items cols = input.cols();
  m_items rows = input.rows();
  auto e = input.elements;
  stderr.writeln("[");
  if(rows>6 && cols>6) {
    foreach(row; 0..3) {
      stderr.write(e[row*cols+0],",",e[row*cols+1],",",e[row*cols+2]);
      stderr.write("...");
      stderr.write(e[row*cols+cols-2],",",e[row*cols+cols-1],",",e[row*cols+cols-2]);
      stderr.writeln();
    }
    stderr.writeln("...");
    foreach(row; rows-3..rows) {
      stderr.write(e[row*cols+0],",",e[row*cols+1],",",e[row*cols+2]);
      stderr.write("...");
      stderr.write(e[row*cols+cols-2],",",e[row*cols+cols-1],",",e[row*cols+cols-2]);
      stderr.writeln();
    }
  }
  else{
    foreach(i, c; e) {
      stderr.write(c,",");
      if (i>6)
        break;
    }
  }

  stderr.writeln("]");
}

void pretty_print(T)(string msg, T[] list) {
  stderr.writeln(msg,":",list[0],",",
          list[1],",",list[2],"...",
          list[$-3],",",list[$-2],",",list[$-1]);
}

void pretty_print(T)(T input) {
  pretty_print("",input);
}

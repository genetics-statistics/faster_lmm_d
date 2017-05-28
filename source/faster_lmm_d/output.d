/*
   This code is part of faster_lmm_d and published under the GPLv3
   License (see LICENSE.txt)

   Copyright © 2017 Prasun Anand & Pjotr Prins
*/

module faster_lmm_d.output;

import std.stdio;

import faster_lmm_d.dmatrix;

void pretty_print(string msg, const DMatrix input) {
  m_items cols = input.cols();
  m_items rows = input.rows();
  auto e = input.elements;
  writeln("[");
  if(rows>6 && cols>6) {
    foreach(row; 0..3) {
      write(e[row*cols+0],",",e[row*cols+1],",",e[row*cols+2]);
      write("...");
      write(e[row*cols+cols-2],",",e[row*cols+cols-1],",",e[row*cols+cols-2]);
      writeln();
    }
    writeln("...");
    foreach(row; rows-3..rows) {
      write(e[row*cols+0],",",e[row*cols+1],",",e[row*cols+2]);
      write("...");
      write(e[row*cols+cols-2],",",e[row*cols+cols-1],",",e[row*cols+cols-2]);
      writeln();
    }
  }
  else{
    foreach(i, c; e) {
      write(c,",");
      if (i>6)
        break;
    }
  }

  writeln("]");
}

void pretty_print(const DMatrix input) {
  pretty_print("",input);
}

void pretty_print(T)(string msg, T[] list) {
  writeln(msg,":",list[0],",",
          list[1],",",list[2],"...",
          list[$-3],",",list[$-2],",",list[$-1]);
}

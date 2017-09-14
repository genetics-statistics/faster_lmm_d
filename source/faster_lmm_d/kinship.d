/*
   This code is part of faster_lmm_d and published under the GPLv3
   License (see LICENSE.txt)

   Copyright © 2017 Prasun Anand & Pjotr Prins
*/

module faster_lmm_d.kinship;

import std.csv;
import std.conv;
import std.exception;
import std.experimental.logger;
import std.file;

import faster_lmm_d.cuda;
import faster_lmm_d.dmatrix;
import faster_lmm_d.helpers;
import faster_lmm_d.optmatrix;
import faster_lmm_d.output;
import faster_lmm_d.memory;

DMatrix kinship_full(const DMatrix G)
{
  info("Full kinship matrix used");
  println("Compute K");
  check_memory();
  m_items m = G.rows(); // snps
  m_items n = G.cols(); // inds
  log(m," SNPs");

  DMatrix temp = slow_matrix_transpose(G);
  DMatrix mmT = matrix_mult(temp, G);
  info("normalize K");
  DMatrix K = divide_dmatrix_num(mmT, m);

  log("kinship_full K sized ",n," ",K.elements.length);
  log(K.elements[0],",",K.elements[1],",",K.elements[2],"...",K.elements[n-3],",",K.elements[n-2],",",K.elements[n-1]);
  ulong row = n;
  ulong lr = n*n-1;
  ulong ll = (n-1)*n;
  log(K.elements[ll],",",K.elements[ll+1],",",K.elements[ll+2],"...",K.elements[lr-2],",",K.elements[lr-1],",",K.elements[lr]);
  //if(test_kinship){ check_kinship(p_values); }
  check_memory();
  return K;
}

EighTuple kvakve(const DMatrix K)
{
  tracef("Obtaining eigendecomposition for %dx%d matrix",K.rows(),K.cols());
  return eigh(K);
}

DMatrix kinship_from_file(const string fn){
  string input = to!string(std.file.read(fn));
  auto csv = csvReader!(string, Malformed.ignore)(input);
  double[] elements;
  foreach(row; csv){
    foreach(cell; row){
      elements ~= to!double(cell);
    }
  }
  return DMatrix([elements.length/261, 261], elements);
}

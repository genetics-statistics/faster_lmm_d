/*
   This code is part of faster_lmm_d and published under the GPLv3
   License (see LICENSE.txt)

   Copyright © 2017 Prasun Anand & Pjotr Prins
*/

module faster_lmm_d.dmatrix;

import std.math;
import std.stdio;
import std.typecons;

alias ulong m_items;

struct DMatrix{
  m_items[] shape; // dimensions are never negative
  double[] elements;
  bool init = false;

  this(const ulong[] s, const double[] e) {
    shape    = s.dup();
    elements = e.dup();
    init     = true;
  }

  const m_items cols() { return shape[1]; }
  const m_items rows() { return shape[0]; }
  const m_items size() { return shape[0] * shape[1]; }
  const m_items n_pheno() { return cols; }
  const m_items m_geno() { return rows; }
  const bool is_square() { return rows == cols; };

  double acc(ulong row, ulong col) {
    return this.elements[row*this.cols()+col];
  }
}

alias Tuple!(DMatrix, "geno", string[], "gnames", immutable(string[]), "ynames") genoObj;

DMatrix logDMatrix(const DMatrix inDmat) {
  m_items total_items = inDmat.size();
  double[] elements = new double[total_items];
  for(auto i = 0; i < total_items; i++) {
    elements[i] = log(inDmat.elements[i]);
  }
  return DMatrix(inDmat.shape, elements);
}

DMatrix addDMatrix(const DMatrix lha, const DMatrix rha) {
  assert(lha.rows() == rha.rows());
  assert(lha.cols() == rha.cols());
  m_items total_items = lha.size();
  double[] elements = new double[total_items];
  for(auto i = 0; i < total_items; i++) {
    elements[i] = lha.elements[i] + rha.elements[i];
  }
  return DMatrix(lha.shape, elements);
}

DMatrix subDMatrix(const DMatrix lha, const DMatrix rha) {
  assert(lha.rows() == rha.rows());
  assert(lha.cols() == rha.cols());
  m_items total_items = lha.size();
  double[] elements = new double[total_items];
  for(auto i = 0; i < total_items; i++) {
    elements[i] = lha.elements[i] - rha.elements[i];
  }
  return DMatrix(lha.shape, elements);
}

DMatrix multiplyDMatrix(const DMatrix lha, const DMatrix rha) {
  ulong[] rha_shape = rha.shape.dup;
  if(lha.rows() != rha.rows()){
    ulong[] temp = rha.shape.dup;
    rha_shape = [temp[1], temp[0]];
  }
  m_items total_items = lha.size();
  double[] elements = new double[total_items];
  if(lha.cols() == rha_shape[1]) {
    for(auto i = 0; i < total_items; i++) {
      elements[i] = lha.elements[i] * rha.elements[i];
    }
  }
  else{
    for(auto i = 0; i < total_items; i++) {
      elements[i] = lha.elements[i] * rha.elements[i%(rha_shape[0]*rha_shape[1])];
    }
  }

  return DMatrix(lha.shape, elements);
}

DMatrix addDMatrixNum(const DMatrix input, const double num) {
  m_items total_items = input.size();
  double[] elements = new double[total_items];
  for(auto i = 0; i < total_items; i++) {
    elements[i] = input.elements[i] + num;
  }
  return DMatrix(input.shape, elements);
}

DMatrix subDMatrixNum(const DMatrix input, const double num) {
  m_items total_items = input.size();
  double[] elements = new double[total_items];
  for(auto i = 0; i < total_items; i++) {
    elements[i] = input.elements[i] - num;
  }
  return DMatrix(input.shape, elements);
}

DMatrix multiplyDMatrixNum(const DMatrix input, const double num) {
  m_items total_items = input.size();
  double[] elements = new double[total_items];
  for(auto i = 0; i < total_items; i++) {
    elements[i] = input.elements[i] * num;
  }
  return DMatrix(input.shape, elements);
}

DMatrix divideNumDMatrix(const double num, const DMatrix input) {
  m_items total_items = input.size();
  double[] elements = new double[total_items];
  for(auto i = 0; i < total_items; i++) {
    elements[i] =  num /input.elements[i];
  }
  return DMatrix(input.shape, elements);
}

DMatrix divideDMatrixNum(const DMatrix input, const double factor) {
  m_items total_items = input.size();
  double[] elements = new double[total_items];
  for(auto i = 0; i < total_items; i++) {
    elements[i] = input.elements[i]/factor;
  }
  return DMatrix(input.shape, elements);
}

DMatrix zerosMatrix(const ulong rows, const ulong cols) {
  double[] elements = new double[rows * cols];
  for(auto i = 0; i < rows*cols; i++) {
    elements[i] = 0;
  }
  return DMatrix([rows, cols], elements);
}

DMatrix onesMatrix(const ulong rows, const ulong cols) {
  double[] elements = new double[rows * cols];
  for(auto i = 0; i < rows*cols; i++) {
    elements[i] = 1;
  }
  return DMatrix([rows, cols], elements);
}

DMatrix horizontallystack(const DMatrix a, const DMatrix b) {
  auto n = a.rows();
  m_items a_cols = a.cols();
  m_items b_cols = b.cols();
  double[] arr;
  for(auto i = 0; i < n; i++) {
    arr ~= a.elements[(a_cols*i)..(a_cols*(i+1))];
    arr ~= b.elements[(b_cols*i)..(b_cols*(i+1))];
  }
  return DMatrix([n, a_cols+b_cols], arr);
}

bool[] compareGt(const DMatrix input, const double val) {
  m_items total_items = input.size;
  bool[] result = new bool[total_items];
  for(auto i = 0; i < total_items; i++) {
    if(input.elements[i] > val) {
      result[i] = true;
    }
    else{
      result[i] = false;
    }
  }
  return result;
}

bool eqeq(const DMatrix lha, const DMatrix rha) {
  auto index = 0;
  foreach(s; lha.shape) {
    if(s != rha.shape[index]) {
      return false;
    }
    index++;
  }
  index = 0;
  foreach(s; lha.elements) {
    double rem = s -rha.elements[index];
    if(rem < 0) {rem *= -1;}
    if(rem > 0.001) {
      return false;
    }
    index++;
  }
  return true;
}

DMatrix getCol(const DMatrix input, const ulong colNo) {
  double[] arr;
  m_items rows = input.rows();
  m_items cols = input.cols();
  for(auto i=0; i < rows; i++) {
    arr ~= input.elements[i*cols+colNo];
  }
  return DMatrix([rows,1],arr);
}

DMatrix getRow(const DMatrix input, const ulong rowNo) {
  m_items cols = input.cols();
  double[] arr = input.elements[rowNo*cols..(rowNo+1)*cols].dup;
  return DMatrix([1,cols],arr);
}


void setCol(ref DMatrix input, const ulong colNo, const DMatrix arr) {
  m_items rows = input.rows();
  m_items cols = input.cols();
  for(auto i=0; i < rows; i++) {
    input.elements[i*cols + colNo] = arr.elements[i];
  }
}

void setRow(ref DMatrix input, const ulong rowNo, const DMatrix arr) {
  auto index =  rowNo*input.cols();
  auto end =  (rowNo+1)*input.cols();
  auto k = 0;
  for(auto i=index; i<end; i++) {
    input.elements[i] = arr.elements[k];
    k++;
  }
}

void nanCounter(const DMatrix input) {
  auto nanCounter = 0;
  foreach(ref ele; input.elements) {
    if(std.math.isNaN(ele)) {
      writeln("Encountered a NaN");
      nanCounter++;
    }
  }
  if(nanCounter>0){
    writefln("NaNs encountered => %d", nanCounter);
  }
}


unittest{
  auto d = DMatrix([2,2],[1,2,3,4]);

  // Test equality of two DMatrixes
  auto lha = DMatrix([3,3], [1,    2, 3, 4, 5, 6, 7, 8, 9]);
  auto rha = DMatrix([3,3], [1.001,2, 3, 4, 5, 6, 7, 8, 9]);
  assert(eqeq(lha, rha));

  // Test the fields of a DMatrix
  assert(d.shape == [2,2]);
  assert(d.elements == [1,2,3,4]);

  // Test elementwise operations
  auto d2 = DMatrix([2,2],[2,4,5,6]);
  auto d3 = DMatrix([2,2],[3,6,8,10]);
  assert(addDMatrix(d, d2) == d3);
  assert(addDMatrixNum(d, 0) == d);

  assert(subDMatrix(d3, d2) == d);
  assert(subDMatrixNum(d, 0) == d);

  auto d4 = DMatrix([2,2],[6,24,40,60]);
  assert(multiplyDMatrix(d2,d3) == d4);
  assert(multiplyDMatrixNum(d2,1) == d2);

  assert(divideDMatrixNum(d2,1) == d2);

  auto zeroMat = DMatrix([3,3], [0,0,0, 0,0,0, 0,0,0]);
  assert(zerosMatrix(3,3) == zeroMat);

  auto onesMat = DMatrix([3,3], [1,1,1, 1,1,1, 1,1,1]);
  assert(onesMatrix(3,3) == onesMat);

  auto colMatrix = DMatrix([2,1], [2,4]);
  assert( getCol(d, 1) == colMatrix );

  auto rowMatrix = DMatrix([1,2], [3,4]);
  assert( getRow(d, 1) == rowMatrix );

  auto row = DMatrix([1,2], [3.5,4.2]);
  setRow(d, 1, row);
  auto newD =  DMatrix([2,2], [1, 2, 3.5, 4.2]);
  assert( d == newD );

  auto left = DMatrix([3, 1], [1,
                               4,
                               5]);

  auto right = DMatrix([3, 2], [2, 4,
                                4, 8,
                                7, 12]);

  auto stacked = DMatrix([3, 3], [1, 2, 4,
                                  4, 4, 8,
                                  5, 7, 12]);
  assert(horizontallystack(left, right) == stacked);
}

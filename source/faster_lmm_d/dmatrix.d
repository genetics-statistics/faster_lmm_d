/*
   This code is part of faster_lmm_d and published under the GPLv3
   License (see LICENSE.txt)

   Copyright © 2017 Prasun Anand & Pjotr Prins
*/

module faster_lmm_d.dmatrix;

import std.algorithm;
import std.conv;
import std.math;
import std.stdio;
import std.typecons;

import faster_lmm_d.optmatrix;

alias size_t m_items; // dimensions are never negative

struct DMatrix{
  m_items[] shape;
  double[] elements;
  bool init = false;

  this(const DMatrix m) {
    shape    = m.shape.dup;
    elements = m.elements.dup;
    init     = true;
  }
  this(const ulong[] shape_in, const double[] e) {
    shape    = shape_in.dup;
    elements = e.dup;
    init     = true;
  }

  const sum() { return reduce!"a + b"(0.0, elements); }

  pragma(inline) const m_items cols() { return shape[1]; }
  pragma(inline) const m_items rows() { return shape[0]; }
  pragma(inline) const m_items size() { return rows() * cols(); }
  pragma(inline) const size_t byte_size() { return size() * double.sizeof; }
  pragma(inline) const m_items n_pheno() { return cols; }
  pragma(inline) const m_items m_geno() { return rows; }
  pragma(inline) const bool is_square() { return rows == cols; };

  /*
   * Validate by comparing two Dmatrices.
   * Params:
   *	other     = matrix to compare
   *	threshold = threshold is used to compare the sum of contents,
   *                created on different hardware. Typically the
   *                difference will be very small
  */
  void validate(const DMatrix other, const double threshold=1.0) {
    stderr.write("Other result:");
    pretty_print(other);
    stderr.write("Result:");
    pretty_print(this);
    assert(rows == other.rows, "rows mismatch, expected "~to!string(other.rows)~" but got "~to!string(rows));
    assert(cols == other.cols, "cols mismatch, expected "~to!string(other.cols)~" but got "~to!string(cols));
    assert(elements.length == other.elements.length);
    assert(abs(elements[$-1]-other.elements[$-1])<threshold,"elements.last mismatch");
    assert(abs(elements[0]-other.elements[0])<threshold,"elements[0] mismatch");
    assert(abs(sum-other.sum)<threshold, "sum mismatch, expected "~to!string(other.sum)~" but got "~to!string(sum));
  }
}

alias Tuple!(const DMatrix, "geno", immutable string[], "gnames", immutable string[], "ynames") GenoObj;

double accessor(const DMatrix input, ulong row, ulong col) {
  return input.elements[row*input.cols()+col];
}

DMatrix log_dmatrix(const DMatrix input) {
  m_items total_items = input.size();
  double[] elements = new double[total_items];
  for(auto i = 0; i < total_items; i++) {
    elements[i] = log(input.elements[i]);
  }
  return DMatrix(input.shape, elements);
}

DMatrix add_dmatrix(const DMatrix lha, const DMatrix rha) {
  assert(lha.rows() == rha.rows());
  assert(lha.cols() == rha.cols());
  m_items total_items = lha.size();
  double[] elements = new double[total_items];
  foreach(i ,lha_element; lha.elements) {
    elements[i] = lha_element + rha.elements[i];
  }
  return DMatrix(lha.shape, elements);
}

DMatrix sub_dmatrix(const DMatrix lha, const DMatrix rha) {
  assert(lha.rows() == rha.rows());
  assert(lha.cols() == rha.cols());
  m_items total_items = lha.size();
  double[] elements = new double[total_items];
  foreach(i ,lha_element; lha.elements) {
    elements[i] = lha_element - rha.elements[i];
  }
  return DMatrix(lha.shape, elements);
}

DMatrix multiply_dmatrix(const DMatrix lha, const DMatrix rha) {
  ulong[] rha_shape = rha.shape.dup;
  if(lha.rows() != rha.rows()){
    ulong[] temp = rha.shape.dup;
    rha_shape = [temp[1], temp[0]];
  }
  m_items total_items = lha.size();
  double[] elements = new double[total_items];
  if(lha.cols() == rha_shape[1]) {
    foreach(i ,lha_element; lha.elements) {
      elements[i] = lha_element * rha.elements[i];
    }
  }
  else{
    foreach(i ,lha_element; lha.elements) {
      elements[i] = lha_element * rha.elements[i%(rha_shape[0]*rha_shape[1])];
    }
  }

  return DMatrix(lha.shape, elements);
}

DMatrix add_dmatrix_num(const DMatrix input, const double num) {
  m_items total_items = input.size();
  double[] elements = new double[total_items];
  foreach(i ,input_element; input.elements) {
    elements[i] = input_element + num;
  }
  return DMatrix(input.shape, elements);
}

DMatrix sub_dmatrix_num(const DMatrix input, const double num) {
  m_items total_items = input.size();
  double[] elements = new double[total_items];
  foreach(i ,input_element; input.elements) {
    elements[i] = input_element - num;
  }
  return DMatrix(input.shape, elements);
}

DMatrix multiply_dmatrix_num(const DMatrix input, const double num) {
  m_items total_items = input.size();
  double[] elements = new double[total_items];
  foreach(i ,input_element; input.elements) {
    elements[i] = input_element * num;
  }
  return DMatrix(input.shape, elements);
}

DMatrix divide_num_dmatrix(const double num, const DMatrix input) {
  m_items total_items = input.size();
  double[] elements = new double[total_items];
  foreach(i ,input_element; input.elements) {
    elements[i] = num/input_element;
  }
  return DMatrix(input.shape, elements);
}

DMatrix divide_dmatrix_num(const DMatrix input, const double factor) {
  m_items total_items = input.size();
  double[] elements = new double[total_items];
  foreach(i ,input_element; input.elements) {
    elements[i] = input_element/factor;
  }
  return DMatrix(input.shape, elements);
}

DMatrix zeros_dmatrix(const ulong rows, const ulong cols) {
  double[] elements = new double[rows * cols];
  for(auto i = 0; i < rows*cols; i++) {
    elements[i] = 0;
  }
  return DMatrix([rows, cols], elements);
}

DMatrix ones_dmatrix(const ulong rows, const ulong cols) {
  double[] elements = new double[rows * cols];
  for(auto i = 0; i < rows*cols; i++) {
    elements[i] = 1;
  }
  return DMatrix([rows, cols], elements);
}

DMatrix horizontally_stack(const DMatrix a, const DMatrix b) {
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

bool[] compare_gt(const DMatrix input, const double val) {
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
  foreach(index, s; lha.shape) {
    if(s != rha.shape[index]) {
      return false;
    }
  }

  foreach(index, s; lha.elements) {
    double rem = s -rha.elements[index];
    if(abs(rem) > 0.001) {
      return false;
    }
  }
  return true;
}

DMatrix get_col(const DMatrix input, const ulong colNo) {
  double[] arr;
  m_items rows = input.rows();
  m_items cols = input.cols();
  for(auto i=0; i < rows; i++) {
    arr ~= input.elements[i*cols+colNo];
  }
  return DMatrix([rows,1],arr);
}

DMatrix get_row(const DMatrix input, const ulong row_no) {
  m_items cols = input.cols();
  double[] arr = input.elements[row_no*cols..(row_no+1)*cols].dup;
  return DMatrix([1,cols],arr);
}


DMatrix set_col(const DMatrix input, const ulong colNo, const DMatrix arr) {
  m_items rows = input.rows();
  m_items cols = input.cols();
  auto result = input.elements.dup;
  for(auto i=0; i < rows; i++) {
    result[i*cols + colNo] = arr.elements[i];
  }
  return DMatrix(input.shape.dup, result);
}

DMatrix set_row(const DMatrix input, const ulong row_no, const DMatrix arr) {
  auto index =  row_no*input.cols();
  auto end =  (row_no+1)*input.cols();
  auto result = input.elements.dup;
  auto k = 0;
  for(auto i=index; i<end; i++) {
    result[i] = arr.elements[k];
    k++;
  }
  return DMatrix(input.shape.dup, result);
}

void nan_counter(const DMatrix input) {
  auto nan_counter = 0;
  foreach(element; input.elements) {
    if(std.math.isNaN(element)) {
      writeln("Encountered a NaN");
      nan_counter++;
    }
  }
  if(nan_counter>0){
    writefln("NaNs encountered => %d", nan_counter);
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
  assert(add_dmatrix(d, d2) == d3);
  assert(add_dmatrix_num(d, 0) == d);

  assert(sub_dmatrix(d3, d2) == d);
  assert(sub_dmatrix_num(d, 0) == d);

  auto d4 = DMatrix([2,2],[6,24,40,60]);
  assert(multiply_dmatrix(d2,d3) == d4);
  assert(multiply_dmatrix_num(d2,1) == d2);

  assert(divide_dmatrix_num(d2,1) == d2);

  auto zero_mat = DMatrix([3,3], [0,0,0, 0,0,0, 0,0,0]);
  assert(zeros_dmatrix(3,3) == zero_mat);

  auto ones_mat = DMatrix([3,3], [1,1,1, 1,1,1, 1,1,1]);
  assert(ones_dmatrix(3,3) == ones_mat);

  auto col_matrix = DMatrix([2,1], [2,4]);
  assert( get_col(d, 1) == col_matrix );

  auto row_matrix = DMatrix([1,2], [3,4]);
  assert( get_row(d, 1) == row_matrix );

  auto row = DMatrix([1,2], [3.5,4.2]);

  auto new_dmatrix =  DMatrix([2,2], [1, 2, 3.5, 4.2]);

  assert( set_row(d, 1, row) == new_dmatrix );

  auto left = DMatrix([3, 1], [1,
                               4,
                               5]);

  auto right = DMatrix([3, 2], [2, 4,
                                4, 8,
                                7, 12]);

  auto stacked = DMatrix([3, 3], [1, 2, 4,
                                  4, 4, 8,
                                  5, 7, 12]);
  assert(horizontally_stack(left, right) == stacked);
}

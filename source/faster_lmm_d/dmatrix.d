/*
   This code is part of faster_lmm_d and published under the GPLv3
   License (see LICENSE.txt)

   Copyright © 2017 - 2018 Prasun Anand & Pjotr Prins
*/

module faster_lmm_d.dmatrix;

import std.algorithm;
import std.conv;
import std.math;
import std.stdio;
import std.typecons;

import faster_lmm_d.helpers;
import faster_lmm_d.optmatrix;
import faster_lmm_d.output;

alias size_t m_items; // dimensions are never negative

struct DMatrix{
  m_items[] shape;
  double[] elements;

  this(const DMatrix m) {
    this(m.shape,m.elements);
  }

  this(const m_items[] shape_in, const double[] e) {
    shape    = shape_in.dup_fast;
    elements = e.dup_fast;
  }

  this(const double[] list) {
    this([list.length,1],list);
  }

  const sum() { return reduce!"a + b"(0.0, elements); }

  pragma(inline) const DMatrix opBinary(string s : "+")(DMatrix other){ return add_dmatrix          (this, other); }
  pragma(inline) const DMatrix opBinary(string s : "-")(DMatrix other){ return subtract_dmatrix     (this, other); }
  pragma(inline) const DMatrix opBinary(string s : "*")(DMatrix other){ return slow_multiply_dmatrix(this, other); }
  pragma(inline) const DMatrix opBinary(string s : "/")(DMatrix other){ return divide_dmatrix       (this, other); }

  pragma(inline) const m_items cols() { return shape[1]; }
  pragma(inline) const m_items rows() { return shape[0]; }
  pragma(inline) const m_items size() { return rows() * cols(); }
  pragma(inline) const size_t byte_size() { return size() * double.sizeof; }
  pragma(inline) const m_items n_pheno() { return cols; }
  pragma(inline) const m_items m_geno() { return rows; }
  pragma(inline) const bool is_square() { return rows == cols; };
  pragma(inline) const DMatrix T() {
    return slow_matrix_transpose(this);
  };

  /*
   * Validate by comparing two Dmatrices.
   * Params:
   *	compute   = delegate returns other matrix to compare
   *	threshold = threshold is used to compare the sum of contents,
   *                created on different hardware. Typically the
   *                difference will be very small
  */
  void validate(DMatrix delegate() compute,
                const double threshold=1.0) {
    version(VALIDATE) {
      auto other = compute();
      writeln("Other result:");
      pretty_print(other);
      writeln("Result:");
      pretty_print(this);
      assert(rows == other.rows, "rows mismatch, expected "~to!string(other.rows)~" but got "~to!string(rows));
      assert(cols == other.cols, "cols mismatch, expected "~to!string(other.cols)~" but got "~to!string(cols));
      assert(elements.length == other.elements.length);
      assert(abs(elements[$-1]-other.elements[$-1])<threshold,"elements.last mismatch");
      assert(abs(elements[0]-other.elements[0])<threshold,"elements[0] mismatch");
      assert(abs(sum-other.sum)<threshold, "sum mismatch, expected "~to!string(other.sum)~" but got "~to!string(sum));
    }
  }
}

alias Tuple!(const DMatrix, "geno", immutable string[], "gnames", immutable string[], "ynames") GenoObj;

alias Tuple!(DMatrix, "first", DMatrix, "last")MatrixSplit;

DMatrix dup_dmatrix(const DMatrix input){
  return DMatrix(input.shape.dup, input.elements.dup);
}

void set(ref DMatrix mat, size_t row, size_t col, double value){
  mat.elements[row* mat.cols + col] = value;
}

double accessor(const DMatrix input, const ulong row, const ulong col) {
  return input.elements[row*input.cols()+col];
}

DMatrix get_diagonal(const DMatrix input){
  assert(input.rows == input.cols);
  double[] elements = new double[input.rows];
  foreach(i; 0..input.rows){
    elements[i] = input.elements[i*input.rows + i];
  }
  return DMatrix([input.rows, 1], elements);
}

DMatrix sigmoid(const DMatrix input) {
  m_items total_items = input.size();
  double[] elements = new double[total_items];
  for(auto i = 0; i < total_items; i++) {
    elements[i] = 1/ (1+exp(-1 * input.elements[i]));
  }
  return DMatrix(input.shape, elements);
}

DMatrix log_dmatrix(const DMatrix input) {
  m_items total_items = input.size();
  double[] elements = new double[total_items];
  for(auto i = 0; i < total_items; i++) {
    elements[i] = log(input.elements[i]);
  }
  return DMatrix(input.shape, elements);
}

DMatrix abs_dmatrix(const DMatrix input) {
  m_items total_items = input.size();
  double[] elements = new double[total_items];
  for(auto i = 0; i < total_items; i++) {
    elements[i] = abs(input.elements[i]);
  }
  return DMatrix(input.shape, elements);
}

DMatrix sqrt_dmatrix(const DMatrix input) {
  m_items total_items = input.size();
  double[] elements = new double[total_items];
  for(auto i = 0; i < total_items; i++) {
    elements[i] = sqrt(input.elements[i]);
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

DMatrix subtract_dmatrix(const DMatrix lha, const DMatrix rha) {
  assert(lha.rows() == rha.rows());
  assert(lha.cols() == rha.cols());
  m_items total_items = lha.size();
  double[] elements = new double[total_items];
  foreach(i ,lha_element; lha.elements) {
    elements[i] = lha_element - rha.elements[i];
  }
  return DMatrix(lha.shape, elements);
}

DMatrix slow_multiply_dmatrix(const DMatrix lha, const DMatrix rha) {
  ulong[] rha_shape = rha.shape.dup_fast;
  if(lha.rows() != rha.rows()){
    ulong[] temp = rha.shape.dup_fast;
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

DMatrix divide_dmatrix(const DMatrix lha, const DMatrix rha) {
  assert(lha.rows() == rha.rows());
  assert(lha.cols() == rha.cols());
  m_items total_items = lha.size();
  double[] elements = new double[total_items];
  foreach(i ,lha_element; lha.elements) {
    elements[i] = lha_element / rha.elements[i];
  }
  return DMatrix(lha.shape, elements);
}

/*
 * Add a number to all elements
 */

DMatrix add_dmatrix_num(const DMatrix input, const double num) {
  m_items total_items = input.size();
  double[] elements = new double[total_items];
  foreach(i ,input_element; input.elements) {
    elements[i] = input_element + num;
  }
  return DMatrix(input.shape, elements);
}

DMatrix subtract_num_dmatrix(const double num, const DMatrix input) {
  m_items total_items = input.size();
  double[] elements = new double[total_items];
  foreach(i ,input_element; input.elements) {
    elements[i] = num - input_element;
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

DMatrix identity_dmatrix(const ulong rows, const ulong cols) {
  assert(rows == cols);
  double[] elements = new double[rows * cols];
  for(auto i = 0; i < rows*cols; i++) {
    elements[i] = 0;
  }

  for(auto i = 0; i < rows; i++){
    elements[i * i] = 1;
  }
  return DMatrix([rows, cols], elements);
}

DMatrix zeros_dmatrix(const ulong rows, const ulong cols) {
  double[] elements = new double[rows * cols];
  for(auto i = 0; i < rows*cols; i++) {
    elements[i] = 0;
  }
  return DMatrix([rows, cols], elements);
}

DMatrix set_zeros_dmatrix(const DMatrix a) {
  ulong rows = a.rows;
  ulong cols = a.cols;
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

DMatrix ones_dmatrix(const ulong[] shape) {
  return ones_dmatrix(shape[0], shape[1]);
}

DMatrix set_ones_dmatrix(const DMatrix a) {
  ulong rows = a.rows;
  ulong cols = a.cols;
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
  m_items rows = input.rows;
  m_items cols = input.cols;
  double[] arr = new double[rows];
  foreach(i; 0..rows) {
    arr[i] = input.elements[i*cols+colNo];
  }
  return DMatrix([rows,1],arr);
}

DMatrix get_row(const DMatrix input, const ulong row_no) {
  m_items cols = input.cols();
  double[] arr = input.elements[row_no*cols..(row_no+1)*cols].dup_fast;
  return DMatrix([1,cols],arr);
}

/*
 * set_col sets the column at colNo to arr
 */
DMatrix set_col(const DMatrix input, const ulong col_no, const DMatrix arr) {
  assert(arr.cols == 1);
  assert(arr.rows == input.rows);
  auto result = input.elements.dup;
  foreach(row; 0..input.rows) {
    result[row*input.cols + col_no] = arr.elements[row];
  }
  return DMatrix(input.shape, result);
}

void set_col2(ref DMatrix input, const ulong col_no, const DMatrix arr) {
  assert(arr.cols == 1);
  assert(arr.rows == input.rows);
  //auto result = input.elements.dup;
  foreach(row; 0..input.rows) {
    input.elements[row*input.cols + col_no] = arr.elements[row];
  }
}

DMatrix set_row(const DMatrix input, const ulong row_no, const DMatrix arr) {
  //assert(arr.rows == 1);
  //assert(arr.cols == input.cols);
  auto result = input.elements.dup;
  auto i = 0;
  foreach(col; row_no*input.cols..(row_no+1)*input.cols) {
    result[col] = arr.elements[i++];
  }
  return DMatrix(input.shape, result);
}

void set_row2(ref DMatrix input, const ulong row_no, const DMatrix arr) {
  //assert(arr.rows == 1);
  //assert(arr.cols == input.cols);
  auto i = 0;
  foreach(col; row_no*input.cols..(row_no+1)*input.cols) {
    input.elements[col] = arr.elements[i++];
  }
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

MatrixSplit mat_split_along_row(DMatrix matrix) {
  DMatrix top, bottom;
  ulong rows = matrix.rows;
  ulong cols = matrix.cols;

  top.shape = [rows/2, cols];
  bottom.shape = [rows - rows/2, cols];

  ulong half = (rows/2)*cols;
  top.elements = matrix.elements[0..half];
  bottom.elements = matrix.elements[half..$];

  return MatrixSplit(top, bottom);
}

MatrixSplit mat_split_along_col(DMatrix matrix){
  DMatrix left, right;
  ulong rows = matrix.rows;
  ulong cols = matrix.cols;
  left.shape = [rows, cols/2];
  right.shape = [rows, cols - cols/2];

  ulong half;
  for(ulong i = 0; i < rows; i++){
    half = i*cols + cols/2;
    left.elements ~= matrix.elements[i*cols..half];
    right.elements ~= matrix.elements[half..(i+1)*cols];
  }

  return MatrixSplit(left, right);
}

DMatrix matrix_join(DMatrix ul, DMatrix ur, DMatrix dl, DMatrix dr){
  DMatrix result;
  result.shape = [ul.rows + dl.rows, ul.cols + ur.cols];
  ulong rows = result.rows;
  ulong cols = result.cols;
  for(ulong i = 0; i < ul.rows; i++){
    result.elements ~= ul.get_row(i).elements;
    result.elements ~= ur.get_row(i).elements;
  }
  for(ulong i = 0; i < dl.rows; i++){
    result.elements ~= dl.get_row(i).elements;
    result.elements ~= dr.get_row(i).elements;
  }
  return result;
}

DMatrix get_subvector_dmatrix(const DMatrix H, const size_t offset, const size_t n){
  double[] elements = H.elements.dup;
  return DMatrix([1, n], elements[offset..(offset+n)]);
}


DMatrix get_sub_dmatrix_old(const DMatrix H, const size_t a, const size_t b, const size_t n1, const size_t n2){
  size_t index = 0, cols = H.cols;
  double[] elements = new double[n1*n2];
  foreach(i; 0..n1){
    foreach(j; 0..n2){
      elements[index++] = H.elements[i*cols + j];
    }
  }
  return DMatrix([n1, n2], elements);
}

DMatrix get_sub_dmatrix(const DMatrix H, const size_t k1, const size_t k2, const size_t n1, const size_t n2){
  size_t index = 0, cols = H.cols;
  double[] elements = new double[n1*n2];
  foreach(i; 0..n1){
    foreach(j; 0..n2){
      elements[index++] = H.elements[(i*cols) + j + k1*cols + k2];
    }
  }
  return DMatrix([n1, n2], elements);
}

DMatrix set_sub_dmatrix(const DMatrix H, const size_t a, const size_t b, const size_t n1, const size_t n2, const DMatrix H_Sub){
  double[] elements = H.elements.dup;
  size_t index = 0, cols = H.cols;
  foreach(i; 0..n1){
    foreach(j; 0..n2){
      elements[(i*cols) + j] = H_Sub.elements[index++];
    }
  }
  return DMatrix(H.shape, elements);
}

void set_sub_dmatrix2_old(ref DMatrix H,  size_t a, size_t b, size_t n1, size_t n2, DMatrix H_Sub){
  size_t index = 0, cols = H.cols;
  foreach(i; 0..n1){
    foreach(j; 0..n2){
     H.elements[(i*cols) + j] = H_Sub.elements[index++];
    }
  }
}

void set_sub_dmatrix2(ref DMatrix H,  size_t k1, size_t k2, size_t n1, size_t n2, DMatrix H_Sub){
  size_t index = 0, cols = H.cols;
  //m'(i,j) = m->data[(k1*m->tda + k2) + i*m->tda + j] // tda ~ cols
  foreach(i; 0..n1){
    foreach(j; 0..n2){
      H.elements[(i*cols) + j + k1*cols + k2] = H_Sub.elements[index++];
    }
  }
}

struct DMatrix_int{
  size_t[] shape;
  int[] elements;

  pragma(inline) const m_items rows() { return shape[0]; }
  pragma(inline) const m_items cols() { return shape[1]; }
  pragma(inline) const m_items size() { return rows() * cols(); }

  this(const size_t[] shape_in, const int[] e) {
    shape    = shape_in.dup_fast;
    elements = e.dup_fast;
  }

  this(const DMatrix_int m) {
    this(m.shape,m.elements);
  }
}

int accessor(const DMatrix_int input, size_t row, size_t col){
  return input.elements[row * input.shape[0] + col];
}

void set(ref DMatrix_int input, size_t row, size_t col, int val){
  input.elements[row * input.shape[0] + col] = val;
}

DMatrix_int zeros_dmatrix_int(const ulong rows, const ulong cols) {
  int[] elements = new int[rows * cols];
  for(auto i = 0; i < rows*cols; i++) {
    elements[i] = 0;
  }
  return DMatrix_int([rows, cols], elements);
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

  assert(subtract_dmatrix(d3, d2) == d);

  auto d4 = DMatrix([2,2],[6,24,40,60]);
  assert(slow_multiply_dmatrix(d2,d3) == d4);
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

  auto row = DMatrix([2,1], [3.5,4.2]);

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

  DMatrix lmatrix = DMatrix([2,3],[1,2,3,
                                  4,5,6]);

  DMatrix rmatrix = DMatrix([3,5],[1,2,3,4,5,
                                   4,5,6,4,6,
                                   7,8,9,7,8]);

  MatrixSplit mat =  mat_split_along_row(lmatrix);
  MatrixSplit nat =  mat_split_along_col(rmatrix);

  DMatrix res_ul = matrix_mult(mat.first, nat.first);
  DMatrix res_ur = matrix_mult(mat.first, nat.last );
  DMatrix res_dl = matrix_mult(mat.last,  nat.first);
  DMatrix res_dr = matrix_mult(mat.last,  nat.last) ;

  assert(matrix_mult(lmatrix, rmatrix) == matrix_join(res_ul, res_ur, res_dl, res_dr));
}

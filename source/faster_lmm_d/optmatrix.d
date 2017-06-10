/*
   This code is part of faster_lmm_d and published under the GPLv3
   License (see LICENSE.txt)

   Copyright © 2017 Prasun Anand & Pjotr Prins
*/

module faster_lmm_d.optmatrix;

import std.experimental.logger;
import std.algorithm: min, max, reduce;
import std.exception: enforce;
import std.math: sqrt, round;
import std.stdio;
import std.typecons; // for Tuples

import cblas : gemm, Transpose, Order;

import faster_lmm_d.dmatrix;
import faster_lmm_d.helpers;

extern (C) {
  void dgetrf_ (int* m, int* n, double* a, int* lda, int* ipiv, int* info);
  void dgetri_ (int* n, double* a, int* lda, const(int)* ipiv, double* work, int* lwork, int* info);
  int LAPACKE_dgetrf (int matrix_layout, int m, int n, double* a, int lda, int* ipiv);
  int LAPACKE_dsyevr (int matrix_layout, char jobz, char range, char uplo, int n,
                      double* a, int lda, double vl, double vu, int il, int iu, double abstol,
                      int* m, double* w, double* z, int ldz, int* isuppz);
  int LAPACKE_dgetri (int matrix_layout, int n, double* a, int lda, const(int)* ipiv);
}

version(CUDA) {
  import faster_lmm_d.cuda;
  DMatrix matrix_mult(const DMatrix lha, const DMatrix rha) {
    auto cuda_result = cuda_matrix_mult(lha,rha);
    return cuda_result;
  }
} else version(ARRAYFIRE) {
  import faster_lmm_d.arrayfire;
  DMatrix matrix_mult(const DMatrix lha, const DMatrix rha) {
    af_array device_lha, device_rha, device_result;
    const long[] ldims = [to!long(lha.cols), to!long(lha.rows)];
    const long[] rdims = [to!long(rha.cols), to!long(rha.rows)];
    af_create_array(&device_lha, cast(void *)lha.elements, 2,  ldims.ptr, af_dtype.f64);
    af_create_array(&device_rha, cast(void *)rha.elements, 2,  rdims.ptr, af_dtype.f64);
    af_matmul(&device_result, device_rha, device_lha, af_mat_prop.AF_MAT_NONE, af_mat_prop.AF_MAT_NONE);
    double[] host_result = new double[lha.rows * rha.cols];
    af_get_data_ptr(host_result.ptr, device_result);
    af_release_array(device_lha);
    af_release_array(device_rha);
    af_release_array(device_result);
    auto res_shape = [lha.rows,rha.cols];
    return DMatrix(res_shape, host_result);
  }
} else { // default
  DMatrix matrix_mult(const DMatrix lha,const DMatrix rha) {
    return cpu_matrix_mult(lha,rha);
  }
}

DMatrix cpu_matrix_mult(const DMatrix lha,const DMatrix rha) {
  double[] C = new double[lha.rows()*rha.cols()];
  gemm(Order.RowMajor, Transpose.NoTrans, Transpose.NoTrans, to!int(lha.rows), to!int(rha.cols), to!int(lha.cols), /*no scaling*/
       1,lha.elements.ptr, to!int(lha.cols), rha.elements.ptr, to!int(rha.cols), /*no addition*/0, C.ptr, to!int(rha.cols));
  auto res_shape = [lha.rows(),rha.cols()];
  return DMatrix(res_shape, C);
}

DMatrix cpu_matrix_mult_transpose(const DMatrix lha, const DMatrix rha) {
  double[] C = new double[lha.rows()*rha.rows()];
  gemm(Order.RowMajor, Transpose.NoTrans, Transpose.NoTrans, to!int(lha.rows), to!int(rha.rows), to!int(lha.cols), /*no scaling*/
       1,lha.elements.ptr, to!int(lha.cols), rha.elements.ptr, to!int(rha.rows), /*no addition*/0, C.ptr, to!int(rha.rows));
  auto res_shape = [lha.rows(),rha.rows()];
  return DMatrix(res_shape, C);
}

DMatrix large_matrix_mult(DMatrix lha, DMatrix rha) {

  MatrixSplit mat =  mat_split_along_row(lha);
  MatrixSplit nat =  mat_split_along_col(rha);

  DMatrix res_ul = matrix_mult(mat.first, nat.first);
  DMatrix res_ur = matrix_mult(mat.first, nat.last );
  DMatrix res_dl = matrix_mult(mat.last,  nat.first);
  DMatrix res_dr = matrix_mult(mat.last,  nat.last) ;

  return matrix_join(res_ul, res_ur, res_dl, res_dr);
}

/*
 * Base call for matrix multiplication
 */

DMatrix matrix_mult(string lname, const DMatrix lha, string rname, const DMatrix rha) {
  trace(lname," x ",rname," (",lha.cols,",",lha.rows," x ",rha.cols,",",rha.rows,")");
  return matrix_mult(lha,rha);
}

DMatrix slow_matrix_transpose(const DMatrix m) {
  assert(m.cols > 0 && m.rows > 0);
  if (m.cols == 1 || m.rows == 1) {
    trace("Fast ",m.cols," x ",m.rows);
    return DMatrix([m.cols, m.rows],m.elements);
  }
  else {
    trace("Slow ", m.cols," x ",m.rows);
    double[] output = new double[m.size];
    auto index = 0;
    auto e = m.elements;
    auto cols = m.cols;
    auto rows = m.rows;
    foreach(i ; 0..cols) {
      foreach(j ; 0..rows) {
        output[index++] = e[j*cols + i];
      }
    }
    return DMatrix([cols, rows],output);
  }
}

DMatrix slice_dmatrix(const DMatrix input, const ulong[] along) {
  trace("In slice_dmatrix");
  double[] output;
  foreach(row_index; along) {
    for(auto i=to!ulong(row_index*input.cols); i < (row_index+1)*input.cols(); i++) {
      output ~= input.elements[i];
    }
  }
  return DMatrix([along.length,input.cols()],output);
}

DMatrix slice_dmatrix_keep(const DMatrix input, const bool[] along) {
  trace("In slice_dmatrix_keep");
  assert(along.length == input.rows());
  m_items cols = input.cols();
  double[] output;
  auto row_index = 0;
  auto shape0 = 0;
  foreach(bool toKeep; along) {
    if(toKeep) {
      for(auto i=row_index*cols; i < (row_index+1)*cols; i++) {
        output ~= input.elements[i];
      }
      shape0++;
    }
    row_index++;

  }
  return DMatrix([shape0,cols],output);
}

DMatrix normalize_along_row(const DMatrix input) {
  double[] largeArr;
  double[] arr;
  log(input.shape);
  m_items rows = input.rows();
  m_items cols = input.cols();
  for(auto i = 0; i < rows; i++) {
    arr = input.elements[(cols*i)..(cols*(i+1))].dup;
    bool[] missing = is_nan(arr);
    bool[] values_arr = negate_bool(missing);
    double[] values = get_num_array(arr,values_arr);
    double mean = global_mean(values);
    double variation = get_variation(values, mean);
    double std_dev = sqrt(variation);

    double[] num_arr = replace_nan(arr, values_arr, mean);
    if(std_dev == 0) {
      foreach(ref elem; num_arr) {
        elem -= mean;
      }
    }else{
      foreach(ref elem; num_arr) {
        elem = (elem - mean) / std_dev;
      }
    }
    largeArr ~= num_arr;
  }
  return DMatrix(input.shape, largeArr);
}

DMatrix remove_cols(const DMatrix input, const bool[] keep) {
  immutable col_length = sum(cast(bool[])keep);
  m_items rows = input.rows();
  m_items cols = input.cols();
  double[] arr = new double[rows*col_length];
  auto index = 0;
  for(auto i= 0; i < rows; i++) {
    for(auto j = i*cols, count = 0; j < (i+1)*cols; j++) {
      if(keep[count] == true) {
        arr[index] = input.elements[j];
        index++;
      }
      count++;
    }
  }
  auto shape = [rows, col_length];
  return DMatrix(shape, arr);
}

DMatrix rounded_nearest(const DMatrix input) {
  m_items total_elements = input.elements.length;
  double[] arr = new double[total_elements];
  for(auto i = 0; i < total_elements; i++) {
    arr[i] = round(input.elements[i]*1000)/1000;
  }
  return DMatrix(input.shape, arr);
}

alias Tuple!(DMatrix,"kva",DMatrix,"kve") EighTuple;

/*
 *  Obtain eigendecomposition for K and return Kva,Kve where Kva is
 *  cleaned of small values < 1e-6 (notably smaller than zero)
 */

EighTuple eigh(const DMatrix input) {
  int n = to!int(input.rows);
  int m = to!int(input.cols);
  double[] elements = input.elements.dup; // will contain output
  int lda = n;
  assert(elements.length >= lda*n); // dimension (LDA, N)

  double vl_unused, vu_unused;
  int il_unused, iu_unused;
  double abstol = 0.001;

  int out_m;
  auto out_m_array = new int[n];
  int ldz = n;
  auto z = new double[m*n];
  auto isuppz = new int[2*n];
  double[] w = new double[n];  // eigenvectors

  int lwork = n*2;
  auto work = new double[lwork];
  int liwork = n*10;
  auto iwork = new int[lwork];

  int il = 1;
  int iu = m;

  auto info = LAPACKE_dsyevr(101, 'V', 'A', 'L', n,
                elements.ptr, n, vl_unused, vu_unused, il, iu, abstol,
                out_m_array.ptr, w.ptr, z.ptr, ldz, isuppz.ptr);
  enforce(info==0);

  DMatrix kva = DMatrix([n,1], w);
  DMatrix kve = DMatrix(input.shape, z);
  for(auto zq = 0 ; zq < kva.elements.length; zq++){
    if(kva.elements[zq]< 1e-6){
      kva.elements[zq] = 0;
    }
  }
  EighTuple e;
  e.kva = kva;
  e.kve = kve;
  return e;
}

double det(const DMatrix input)
in {
  assert(input.is_square, "Input matrix should be square");
}
body {
  auto rf = getrf(input);
  // odd permutations => negative:
  auto idx=1;
  auto num_perm = reduce!((a,b) => a + ( b != idx++ ? 1 : 0 ) )(0,rf.ipiv);
  auto prod = (num_perm % 2 ? 1.0 : -1.0 );
  auto m    = rf.arr;
  foreach(i; 0..min(input.rows,input.cols)) {
    prod *= m[input.cols*i + i];
  }
  return prod;
}

alias Tuple!(int[],"ipiv",double[],"arr") LUtup;

LUtup getrf(const double[] arr, const m_items cols, const m_items rows = 1) {
  auto m = to!int(cols);
  assert(m>=1);
  auto n = to!int(rows);
  auto arr2 = arr.dup; // to store the result
  auto ipiv = new int[min(m,n)+1];
  int info;
  int i_cols = to!int(cols);
  int lda = m;
  enforce(LAPACKE_dgetrf(101,m,n,arr2.ptr,lda,ipiv.ptr)==0);
  LUtup t;
  t.ipiv = ipiv;
  t.arr = arr2;
  return t;
}

auto getrf(const DMatrix m) {
  return getrf(m.elements, m.cols, m.rows);
}

DMatrix inverse(const DMatrix input) {
  auto res = getrf(input);
  auto n = to!int(input.cols);
  auto m = to!int(input.rows);
  assert(m == n); // this implementation
  auto lda = n;
  enforce(LAPACKE_dgetri(101, n, res.arr.ptr, lda, res.ipiv.ptr)==0);
  return DMatrix(input.shape, res.arr);
}

import std.conv;

unittest{
  DMatrix d1 = DMatrix([3,4],[2,4,5,6, 7,8,9,10, 2,-1,-4,3]);
  DMatrix d2 = DMatrix([4,2],[2,7,8,9, -5,2,-1,-4]);
  DMatrix d3 = DMatrix([3,2], [5,36,23, 99,13,-15]);
  assert(matrix_mult(d1,d2) == d3);

  // > m <- matrix(c(2,-1,-4,3),2,2)
  // > ginv(m)
  //      [,1] [,2]
  // [1,]  1.5    2
  // [2,]  0.5    1

  DMatrix d4 = DMatrix([2,2], [2, -1, -4, 3]);
  DMatrix d5 = DMatrix([2,2], [1.5, 0.5, 2, 1]);
  assert(inverse(d4) == d5, to!string(inverse(d4)));

  // > m <- matrix(c(2,-1,-4,3,1,2),2,3)
  // > m
  //      [,1] [,2] [,3]
  // [1,]    2   -4    1
  // [2,]   -1    3    2
  // > ginv(m)
  //            [,1] [,2]
  // [1,]  0.1066667 0.02
  // [2,] -0.1333333 0.10
  // [3,]  0.2533333 0.36

  DMatrix d6 = DMatrix([2,2],[2, -4, -1, 3]);
  assert(d4.T == d6);

  DMatrix m = DMatrix([3,4],[10, 11, 12, 13,
                             14, 15, 16, 17,
                             18, 19, 20, 21]);

  DMatrix matrix = DMatrix([4,3],[10,14,18,
                              11,15,19,
                              12,16,20,
                              13,17,21]);

  auto transposed_mat = m.T;
  auto result_mat = matrix.T;
  assert(transposed_mat == matrix,to!string(transposed_mat));
  assert(result_mat == m,to!string(result_mat));

  DMatrix d7 = DMatrix([4,2],[-3,13,7, -5, -12, 26, 2, -8]);

  // DMatrix d4 = DMatrix([2,2], [2, -1, -4, 3]);
  assert(det(d4) == 2,"det(d4) expected 2, but have "~to!string(det(d4)));  // 2*3 - -1*-4 = 6 - 4 = 2

  auto d8 = DMatrix([3,3], [21, 14, 12, -11, 22, 1, 31, -11, 42]);
  auto eigh_matrix = eigh(d8);
  auto kva_matrix = DMatrix([3, 1], [0, 17.322, 69.228]);
  auto kve_matrix = DMatrix([3, 3], [-0.823, 0.075, 0.563, -0.126, 0.943, -0.310, 0.554, 0.326, 0.766]);
  assert( rounded_nearest(eigh_matrix.kva) == kva_matrix);
  assert( rounded_nearest(eigh_matrix.kve) == kve_matrix);

  auto mat = DMatrix([3,3], [4,  6,  11,
                             5,  5,  5,
                             11, 12, 13]);

  auto rmMat = DMatrix([3,1], [11,
                                5,
                               13]);
  assert(remove_cols(mat, [false, false, true]) == rmMat);

  auto sliced_mat = DMatrix([2,3], [4, 6, 11,
                                   5, 5, 5, ]);

  assert(slice_dmatrix(mat, [0,1]) == sliced_mat);
  assert(slice_dmatrix_keep(mat, [true, true, false]) == sliced_mat);

  auto norm_mat = DMatrix([3,3], [-1.01905, -0.339683, 1.35873,
                                        0,         0,       0,
                                 -1.22474,         0, 1.22474]);
  assert(eqeq(normalize_along_row(mat), norm_mat));

  DMatrix lmatrix = DMatrix([2,3],[1,2,3,
                                  4,5,6]);

  DMatrix rmatrix = DMatrix([3,5],[1,2,3,4,5,
                                   4,5,6,4,6,
                                   7,8,9,7,8]);

  assert(matrix_mult(lmatrix, rmatrix) == large_matrix_mult(lmatrix, rmatrix));
}

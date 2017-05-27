/*
   This code is part of faster_lmm_d and published under the GPLv3
   License (see LICENSE.txt)

   Copyright © 2017 Prasun Anand & Pjotr Prins
*/

module faster_lmm_d.cuda;

version(CUDA) {

import core.stdc.stdlib : exit;
import std.conv;

import cuda_d.cublas_api;
import cuda_d.cublas_v2;
import cuda_d.cuda;
import cuda_d.cuda_runtime_api;

import faster_lmm_d.dmatrix;
import faster_lmm_d.memory;


 void cuda_init() {};
 void cuda_destroy() {};

void gpu_blas_mmul(const double *A, const double *B, double *C, const int m, const int k, const int n) {
  int lda=m,ldb=k,ldc=m;
  const double alf = 1;
  const double bet = 0;
  const double *alpha = &alf;
  const double *beta = &bet;

  // Create a handle for CUBLAS
  cublasHandle_t handle;
  cublasCreate(&handle);

  // Do the actual multiplication
  cublasDgemm(handle, cublasOperation_t.CUBLAS_OP_N, cublasOperation_t.CUBLAS_OP_N, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);

  // Destroy the handle
  cublasDestroy(handle);
}

DMatrix cuda_matrix_mult(const DMatrix rha, const DMatrix lha){
  // Allocate 3 arrays on CPU
  check_memory();

  int nr_rows_A, nr_cols_A, nr_rows_B, nr_cols_B, nr_rows_C, nr_cols_C;
  nr_rows_A = to!int(lha.cols);
  nr_cols_A = to!int(lha.rows);
  nr_rows_B = to!int(rha.cols);
  nr_cols_B = to!int(rha.rows);
  nr_rows_C = to!int(lha.cols);
  nr_cols_C = to!int(rha.rows);

  auto h_C = new double[nr_rows_C * nr_cols_C];

  double* d_A, d_B, d_C;

  cudaMalloc(cast(void **)&d_A,nr_rows_A * nr_cols_A * to!int(double.sizeof));
  cudaMalloc(cast(void **)&d_B,nr_rows_B * nr_cols_B * to!int(double.sizeof));
  cudaMalloc(cast(void **)&d_C,nr_rows_C * nr_cols_C * to!int(double.sizeof));

  cudaMemcpy(cast(void*)d_A, cast(void*)lha.elements, nr_rows_A * nr_cols_A * to!int(double.sizeof), cudaMemcpyKind.cudaMemcpyHostToDevice);
  cudaMemcpy(cast(void*)d_B, cast(void*)rha.elements, nr_rows_B * nr_cols_B * to!int(double.sizeof), cudaMemcpyKind.cudaMemcpyHostToDevice);

  // Multiply A and B on GPU

  gpu_blas_mmul(d_A, d_B, d_C, nr_rows_A, nr_cols_A, nr_cols_B);

  // Copy (and print) the result on host memory
  cudaMemcpy(h_C.ptr,d_C,nr_rows_C * nr_cols_C * to!int(double.sizeof), cudaMemcpyKind.cudaMemcpyDeviceToHost);

  //Free GPU memory
  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);

  return DMatrix([rha.rows, lha.cols], h_C);
}

} // CUDA

/*
   This code is part of faster_lmm_d and published under the GPLv3
   License (see LICENSE.txt)

   Copyright © 2017 Prasun Anand & Pjotr Prins
*/

module faster_lmm_d.cuda;

version(CUDA) {

  import std.experimental.logger;
  import std.conv;
  import std.exception;

  import cuda_d.cublas_api;
  import cuda_d.cublas_v2;
  import cuda_d.cuda;
  import cuda_d.cuda_runtime_api;

  import faster_lmm_d.dmatrix;
  import faster_lmm_d.optmatrix;
  import faster_lmm_d.memory;

  const ulong MB = 1024*1024;

void gpu_blas_mmul(double *A, const double *B, double *C, const m_items _m, const m_items _k, const m_items _n) {
  auto m=to!int(_m), k=to!int(_k), n=to!int(_n);
  auto lda=m,ldb=k,ldc=m;
  const double alf = 1;
  const double bet = 0;
  const double *alpha = &alf;
  const double *beta = &bet;

  // Create a handle for CUBLAS
  cublasHandle_t handle;
  cublasCreate(&handle);

  trace("Running cublasDgemm");
  cublasDgemm(handle, cublasOperation_t.CUBLAS_OP_N, cublasOperation_t.CUBLAS_OP_N, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);

  // Destroy the handle
  cublasDestroy(handle);
}

DMatrix cuda_matrix_mult(const DMatrix rha, const DMatrix lha){
  trace("Entering cuda_matrix_mult");
  check_memory();

  double *gpu_malloc(ulong size) {
    void *gpu_ptr;
    trace("Allocating GPU RAM of ",(size * to!uint(double.sizeof))/MB,"MB");
    auto cudaStat = cudaMalloc(&gpu_ptr, size * to!uint(double.sizeof));
    enforce(cudaStat == cudaError.cudaSuccess,"CUDA device memory allocation failed");
    return cast(double *)gpu_ptr;
  }

  auto nr_rows_A = lha.cols;
  auto nr_cols_A = lha.rows;
  auto nr_rows_B = rha.cols;
  auto nr_cols_B = rha.rows;
  auto nr_rows_C = lha.cols;
  auto nr_cols_C = rha.rows;

  trace("CUDA result matrix size =",nr_rows_C,",",nr_cols_C);
  if (nr_rows_C * nr_cols_C < 10000 || nr_rows_C < 10 || nr_cols_C < 10) {
    trace("Matrix is small, so running the CPU version instead");
    return cpu_matrix_mult(rha,lha);
  }

  trace("CUDA A matrix size =",nr_rows_A,",",nr_cols_A);
  auto d_A = gpu_malloc(nr_rows_A * nr_cols_A);
  trace("CUDA A matrix size =",nr_rows_B,",",nr_cols_B);
  auto d_B = gpu_malloc(nr_rows_B * nr_cols_B);
  auto d_C = gpu_malloc(nr_rows_C * nr_cols_C);

  cudaMemcpy(cast(void*)d_A, cast(void*)lha.elements, nr_rows_A * nr_cols_A * double.sizeof, cudaMemcpyKind.cudaMemcpyHostToDevice);
  cudaMemcpy(cast(void*)d_B, cast(void*)rha.elements, nr_rows_B * nr_cols_B * double.sizeof, cudaMemcpyKind.cudaMemcpyHostToDevice);

  // Multiply A and B on GPU

  gpu_blas_mmul(d_A, d_B, d_C, nr_rows_A, nr_cols_A, nr_cols_B);

  // Copy (and print) the result on host memory
  auto h_C = new double[nr_rows_C * nr_cols_C];
  cudaMemcpy(h_C.ptr,d_C,nr_rows_C * nr_cols_C * double.sizeof, cudaMemcpyKind.cudaMemcpyDeviceToHost);

  //Free GPU memory
  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);

  check_memory();
  trace("Exit cuda_matrix_mult");

  return DMatrix([rha.rows, lha.cols], h_C);
}

} // CUDA

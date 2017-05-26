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
  import std.parallelism;

  import cuda_d.cublas_api;
  import cuda_d.cublas_v2;
  import cuda_d.cuda;
  import cuda_d.cuda_runtime_api;

  import faster_lmm_d.dmatrix;
  import faster_lmm_d.optmatrix;
  import faster_lmm_d.memory;

  const ulong MB = 1024*1024;
  const cudaSuccess = cudaError.cudaSuccess;

// Do not call this function outside cuda_init
static void cuda_startup() {
  void *dummy;
  // allocate some CUDA RAM to force initialization
  enforce(cudaMalloc(&dummy, 8000)==cudaSuccess,"CUDA failed to initialize");
  enforce(cudaFree(dummy)==cudaSuccess);
}

void cuda_init() {
  trace("Initializing CUDA on separate thread");
  auto t = task!cuda_startup();
  t.executeInNewThread();
  trace("Back to main thread...");
}

void gpu_blas_mmul(double *A, const double *B, double *C, const m_items _m, const m_items _k, const m_items _n) {
  auto m=to!int(_m), k=to!int(_k), n=to!int(_n);
  auto lda=m,ldb=k,ldc=m;
  const double alf = 1;
  const double bet = 0;
  const double *alpha = &alf;
  const double *beta = &bet;

  // Create a handle for CUBLAS
  cublasHandle_t handle;
  enforce(cublasCreate(&handle) == cublasStatus_t.CUBLAS_STATUS_SUCCESS, "CUBLAS initialization failed");
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
    auto cudaStat = cudaMalloc(&gpu_ptr, to!size_t(size) * double.sizeof);
    enforce(cudaStat == cudaSuccess,"CUDA device memory allocation failed");
    return cast(double *)gpu_ptr;
  }

  void copy_ram_to_gpu(double *dest, const DMatrix src) {
    enforce(cudaMemcpy(dest, cast(void*)src.elements, src.byte_size, cudaMemcpyKind.cudaMemcpyHostToDevice)==cudaSuccess);
  }

  //auto nr_rows_A = lha.cols;
  //auto nr_cols_A = lha.rows;
  //auto nr_rows_B = rha.cols;
  //auto nr_cols_B = rha.rows;
  auto A = lha, B = rha;
  auto nr_rows_C = lha.cols;
  auto nr_cols_C = rha.rows;

  trace("CUDA result matrix size =",nr_rows_C,",",nr_cols_C);
  if (nr_rows_C * nr_cols_C < 10000 || nr_rows_C < 10 || nr_cols_C < 10) {
    trace("Matrix is small, so running the CPU version instead");
    return cpu_matrix_mult(rha,lha);
  }

  auto d_A = gpu_malloc(A.size);
  auto d_B = gpu_malloc(B.size);
  auto d_C = gpu_malloc(nr_rows_C * nr_cols_C);

  // ---- Initialize GPU matrices
  copy_ram_to_gpu(d_A,lha);
  copy_ram_to_gpu(d_B,rha);
  enforce(cudaMemset(d_C,0,to!size_t(nr_rows_C) * nr_cols_C * double.sizeof)==cudaSuccess);

  //gpu_blas_mmul(d_A, d_B, d_C, nr_rows_A, nr_cols_A, nr_cols_B);
  gpu_blas_mmul(d_A, d_B, d_C, A.cols, A.rows, B.rows);

  // ---- Copy result to RAM
  auto h_C = new double[nr_rows_C * nr_cols_C];
  cudaMemcpy(h_C.ptr,d_C,nr_rows_C * nr_cols_C * double.sizeof, cudaMemcpyKind.cudaMemcpyDeviceToHost);

  enforce(cudaFree(d_A)==cudaSuccess,"Error freeing d_A");
  enforce(cudaFree(d_B)==cudaSuccess,"Error freeing d_B");
  enforce(cudaFree(d_C)==cudaSuccess,"Error freeing d_C");

  check_memory("Exit CUDA matrix multiply");

  return DMatrix([nr_cols_C, nr_rows_C], h_C);
}

} // CUDA

/*
   This code is part of faster_lmm_d and published under the GPLv3
   License (see LICENSE.txt)

   Copyright © 2017 Prasun Anand & Pjotr Prins
*/

module faster_lmm_d.cuda;

version(CUDA) {

  import std.experimental.logger;
  import std.algorithm;
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

  alias GPU_PTR = double *;
  const ulong MB = 1024*1024;
  const cudaSuccess = cudaError.cudaSuccess;
  __gshared cublasHandle_t cublas_handle = null;

// Do not call this function outside cuda_init
static void cuda_startup() {
  void *dummy;
  // allocate some CUDA RAM to force initialization
  enforce(cudaMalloc(&dummy, 8000)==cudaSuccess,"CUDA failed to initialize");
  enforce(cudaFree(dummy)==cudaSuccess);
  // Create a handle for CUBLAS
  enforce(cublasCreate(&cublas_handle) == cublasStatus_t.CUBLAS_STATUS_SUCCESS, "CUBLAS initialization failed");
}

void cuda_init() {
  trace("Initializing CUDA on separate thread");
  auto t = task!cuda_startup();
  t.executeInNewThread();
  trace("Back to main thread...");
}

void cuda_destroy() {
  trace("Close CUDA environment");
  if (cublas_handle)
    cublasDestroy(cublas_handle);
}

/*
 * Matrix multiplication using CUDA.
 */

DMatrix cuda_matrix_mult(const DMatrix _B, const DMatrix _A){
  trace("Entering cuda_matrix_mult");
  check_memory();

  GPU_PTR gpu_malloc(ulong size) {
    void *gpu_mem;
    auto bytes = (size * to!uint(double.sizeof));
    trace("Allocating GPU RAM of ",(bytes>MB ? to!string(bytes/MB)~"MB" : to!string(bytes)));
    enforce(cudaMalloc(&gpu_mem, bytes) == cudaSuccess,"CUDA device memory allocation failed");
    return cast(GPU_PTR)gpu_mem;
  }

  void copy_ram_to_gpu(GPU_PTR dest, const DMatrix src) {
    enforce(cudaMemcpy(dest, cast(void*)src.elements, src.byte_size, cudaMemcpyKind.cudaMemcpyHostToDevice)==cudaSuccess);
  }

  // CUDA is column major ordered by default
  auto A = DMatrix([_A.shape[1],_A.shape[0]],_A.elements);
  auto B = DMatrix([_B.shape[1],_B.shape[0]],_B.elements);

  // auto C_cols = B.cols;
  // auto C_rows = A.rows;
  auto C_cols = _B.rows;
  auto C_rows = _A.cols;

  auto C_size = C_cols * C_rows;
  auto C_byte_size = to!size_t(C_size) * double.sizeof;

  trace("CUDA result matrix size =",C_rows,",",C_cols);
  //if (C_size < 1000) {
  //  trace("Matrix is small, so running the CPU version instead");
  //  return cpu_matrix_mult(A,B);
  //}

  auto d_A = gpu_malloc(A.size);
  auto d_B = gpu_malloc(B.size);
  auto d_C = gpu_malloc(C_size);

  // ---- Initialize GPU matrices
  copy_ram_to_gpu(d_A,A);
  copy_ram_to_gpu(d_B,B);
  // enforce(cudaMemset(d_C,0,C_byte_size)==cudaSuccess); // skip because beta == 0.0

  // C = αAxB + βC
  int m = to!int(A.rows); // number of rows of matrix op(A) and C.
  enforce(A.rows == C_rows);
  int n = to!int(B.cols); // number of columns of matrix op(B) and C.
  enforce(B.cols == C_cols);
  int k = to!int(A.cols); // number of columns of op(A) and rows of op(B).
  enforce(A.cols == B.rows);
  auto alpha = 1.0;    // scalar used for multiplication.
  auto beta = 0.0;     // scalar used for multiplication. If beta==0, C does not have to be a valid input.
  int lda = to!int(m); // leading dimension of two-dimensional array used to store A.
  int ldb = to!int(k); // leading dimension of two-dimensional array used to store B.
  int ldc = to!int(m); // leading dimension of a two-dimensional array used to store the matrix C.

  trace("m=",m," n=",n," k=",k);
  // for A array of dimensions lda x k with lda>=max(1,m) if transa == CUBLAS_OP_N and lda x m with lda>=max(1,k) otherwise.
  enforce(lda >= max(1,m));
  enforce(lda * k == A.size);

  // for B array of dimension ldb x n with ldb>=max(1,k) if transa == CUBLAS_OP_N and ldb x k with ldb>=max(1,n) otherwise
  enforce(ldb >= max(1,k));
  enforce(ldb * n == B.size);

  // for C array array of dimensions ldc x n with ldc>=max(1,m).
  enforce(ldc >= max(1,m));
  enforce(ldc * n == C_size);

  //cublasHandle_t cublas_handle2 = null;
  //enforce(cublasCreate(&cublas_handle2) == cublasStatus_t.CUBLAS_STATUS_SUCCESS, "CUBLAS initialization failed");
  enforce(cublasDgemm(cublas_handle,
                      cublasOperation_t.CUBLAS_OP_N,
                      cublasOperation_t.CUBLAS_OP_N,
                      m, n, k, &alpha, d_A, lda, d_B, ldb, &beta, d_C, ldc)==cublasStatus_t.CUBLAS_STATUS_SUCCESS, "cublasDgemm failed");

  // ---- Copy result to RAM
  auto result = new double[C_size];
  enforce(cudaMemcpy(result.ptr,d_C,C_byte_size,cudaMemcpyKind.cudaMemcpyDeviceToHost)==cudaSuccess,"cudaMemcpy failed with size "~to!string(C_size));

  enforce(cudaFree(d_A)==cudaSuccess,"cudaFree error d_A");
  enforce(cudaFree(d_B)==cudaSuccess,"cudaFree error d_B");
  enforce(cudaFree(d_C)==cudaSuccess,"cudaFree error d_C");

  check_memory("Exit CUDA matrix multiply");

  return DMatrix([C_cols, C_rows], result);
}

} // CUDA

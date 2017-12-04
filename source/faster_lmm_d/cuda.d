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
  import std.stdio;
  import std.typecons;

  import cuda_d.cublas_api;
  import cuda_d.cublas_v2;
  import cuda_d.cuda;
  import cuda_d.cuda_runtime_api;

  import faster_lmm_d.dmatrix;
  import faster_lmm_d.optmatrix;
  import faster_lmm_d.memory;

  const ulong MB = 1024*1024;
  const RUN_CUDA_AT_SIZE =5*MB;
  const cudaSuccess = cudaError.cudaSuccess;

  alias GPU_PTR = double *;
  __gshared cublasHandle_t cublas_handle = null;

  alias GPU_PTRS = Tuple!(GPU_PTR, GPU_PTR, GPU_PTR);
  alias GPU_PTRS_SIZE = Tuple!(size_t, size_t, size_t);

  GPU_PTRS ptr_cache;
  GPU_PTRS_SIZE ptr_cache_size;
  bool ptr_cache_initialized = false;

  // Do not call this function outside cuda_init - it actually runs on a separate
  // initialisation thread
  static void gpu_startup() {
    // allocate some CUDA RAM to force early initialization
    auto dummy = gpu_malloc(8000);
    enforce(cudaFree(dummy)==cudaSuccess);
    // Create a handle for CUBLAS
    enforce(cublasCreate(&cublas_handle) == cublasStatus_t.CUBLAS_STATUS_SUCCESS, "CUBLAS initialization failed");
  }

  GPU_PTR gpu_malloc(ulong size) {
    void *gpu_mem;
    auto bytes = (size * to!uint(double.sizeof));
    trace("Allocating GPU RAM of ",(bytes>MB ? to!string(bytes/MB)~"MB" : to!string(bytes)));
    enforce(cudaMalloc(&gpu_mem, bytes) == cudaSuccess,"CUDA device memory allocation failed");
    return cast(GPU_PTR)gpu_mem;
  }

  static GPU_PTRS gpu_malloc(size_t size1, size_t size2, size_t size3) {
    if (!ptr_cache_initialized) {
      ptr_cache = tuple(gpu_malloc(size1),gpu_malloc(size2),gpu_malloc(size3));
      ptr_cache_size = tuple(size1,size2,size3);
      ptr_cache_initialized = true;
    }
    else {
      // realloc if size is too small
      foreach (i, size; tuple(size1,size2,size3)) {
        if (size > ptr_cache_size[i]) {
          gpu_free(ptr_cache[i]);
          ptr_cache[i] = gpu_malloc(size);
          ptr_cache_size[i] = size;
        }
      }
    }
    return ptr_cache;
  }

  void gpu_free(GPU_PTR p) {
    trace("gpu_free");
    enforce(cudaFree(p)==cudaSuccess,"cudaFree error");
  }

  static void gpu_free()
  {
    if (ptr_cache_initialized) {
      gpu_free(ptr_cache[0]);
      gpu_free(ptr_cache[1]);
      gpu_free(ptr_cache[2]);
      ptr_cache_initialized = false;
    }
  }

  void cuda_get_free_memory() {
    ulong free, total;
    int gpuCount;
    CUresult res;
    CUdevice dev;
    CUcontext ctx;

    cuInit(0);
    cuDeviceGetCount(&gpuCount);
    trace("Detected "~to!string(gpuCount)~" GPUs");
    foreach (i; 0..gpuCount) {
      cuDeviceGet(&dev,i);
      ulong bytes;
      enforce(cuDeviceTotalMem(&bytes,dev) == cudaSuccess);
      cuCtxCreate(&ctx, 0, dev);
      enforce(cuMemGetInfo(&free, &total) == cudaSuccess);
      trace("^^^^ Device: ",i," ",free/MB,"MB free out of ",total/MB,"MB (",(free*1.0)/total,"% used) ",bytes/MB,"MB total");
      cuCtxDetach(ctx);
    }
  }

  void cuda_copy_ram_to_gpu(GPU_PTR dest, const(double *)src, size_t size) {
    enforce(cudaMemcpy(dest, cast(void*)src, size, cudaMemcpyKind.cudaMemcpyHostToDevice)==cudaSuccess);
  }

  void cuda_copy_ram_to_gpu(GPU_PTR dest, const DMatrix src) {
    cuda_copy_ram_to_gpu(dest, src.elements.ptr, src.byte_size);
  }

  void cuda_init() {
    trace("Initializing CUDA on separate thread");
    offload_init(0);
    cuda_get_free_memory();
    auto t = task!gpu_startup();
    t.executeInNewThread();
  }

  void cuda_destroy() {
    trace("Close CUDA environment");
    offload_destroy(0);
    if (cublas_handle)
      cublasDestroy(cublas_handle);
    gpu_free();
    cuda_get_free_memory();
  }

  /*
   * Matrix multiplication using CUDA.
   */

  double cuda_ddot(const DMatrix lha, const DMatrix rha){
    double result;
    auto d_lha = gpu_malloc(lha.size);
    auto d_rha = gpu_malloc(rha.size);

    cuda_copy_ram_to_gpu(d_lha, lha);
    cuda_copy_ram_to_gpu(d_rha, rha);

    cublasDdot(cublas_handle, to!int(lha.elements.length), d_lha, 1, d_rha, 1, &result);
    gpu_free(d_lha);
    gpu_free(d_rha);
    return result;
  }

  DMatrix cuda_matrix_mult(const DMatrix _B, const DMatrix _A){

    if (_A.byte_size < RUN_CUDA_AT_SIZE && _B.byte_size < RUN_CUDA_AT_SIZE) {
      trace("Matrix is small: running CPU multiplication instead");
      return cpu_matrix_mult(_B,_A);
    }

    // Check for cached matrices
    auto cached_B = cast(GPU_PTR)offload_get_ptr(_B);
    if (cached_B)
      trace("CACHE HIT!! ",cached_B);

    // CUDA is column major ordered by default
    auto A = DMatrix([_A.shape[1],_A.shape[0]],_A.elements);
    auto B = DMatrix([_B.shape[1],_B.shape[0]],_B.elements);

    auto C_cols = _B.rows;
    auto C_rows = _A.cols;

    auto C_size = C_cols * C_rows;
    auto C_byte_size = to!size_t(C_size) * double.sizeof;

    trace("CUDA multiply A",A.rows,"x",A.cols," B",B.rows,"x",B.cols," into C",C_rows,",",C_cols);

    auto ptrs = gpu_malloc(A.size, B.size, C_size);
    auto d_A = ptrs[0];
    auto d_B = ptrs[1];
    auto d_C = ptrs[2];

    // ---- Initialize GPU matrices
    cuda_copy_ram_to_gpu(d_A,A);
    if (cached_B)
      d_B = cached_B;
    else
      cuda_copy_ram_to_gpu(d_B,B);
    // enforce(cudaMemset(d_C,0,C_byte_size)==cudaSuccess); // skip because beta == 0.0

    // C = αAxB + βC
    int m = to!int(A.rows); // number of rows of matrix op(A) and C.
    assert(A.rows == C_rows);
    int n = to!int(B.cols); // number of columns of matrix op(B) and C.
    assert(B.cols == C_cols);
    int k = to!int(A.cols); // number of columns of op(A) and rows of op(B).
    assert(A.cols == B.rows);
    auto alpha = 1.0;       // scalar used for multiplication.
    auto beta = 0.0;        // scalar used for multiplication. If beta==0, C does not have to be a valid input.
    int lda = to!int(m);    // leading dimension of two-dimensional array used to store A.
    int ldb = to!int(k);    // leading dimension of two-dimensional array used to store B.
    int ldc = to!int(m);    // leading dimension of a two-dimensional array used to store the matrix C.

    // trace("m=",m," n=",n," k=",k);
    // for A array of dimensions lda x k with lda>=max(1,m) if transa == CUBLAS_OP_N and lda x m with lda>=max(1,k) otherwise.
    assert(lda >= max(1,m));
    assert(lda * k == A.size);

    // for B array of dimension ldb x n with ldb>=max(1,k) if transa == CUBLAS_OP_N and ldb x k with ldb>=max(1,n) otherwise
    assert(ldb >= max(1,k));
    assert(ldb * n == B.size);

    // for C array array of dimensions ldc x n with ldc>=max(1,m).
    assert(ldc >= max(1,m));
    assert(ldc * n == C_size);

    enforce(cublasDgemm(cublas_handle,
                        cublasOperation_t.CUBLAS_OP_N,
                        cublasOperation_t.CUBLAS_OP_N,
                        m, n, k, &alpha, d_A, lda, d_B, ldb, &beta, d_C, ldc)==cublasStatus_t.CUBLAS_STATUS_SUCCESS, "cublasDgemm failed");

    auto result = new double[C_size];
    enforce(cudaMemcpy(result.ptr,d_C,C_byte_size,cudaMemcpyKind.cudaMemcpyDeviceToHost)==cudaSuccess,"cudaMemcpy failed with size "~to!string(C_size));

    debug { check_memory("Exit CUDA matrix multiply"); }

    auto cuda_result = DMatrix([C_cols, C_rows], result);
    cuda_result.validate( () => cpu_matrix_mult(_B,_A));
    return cuda_result;
  }

} // CUDA

/*
   This code is part of faster_lmm_d and published under the GPLv3
   License (see LICENSE.txt)

   Copyright © 2017 Prasun Anand & Pjotr Prins
*/

module faster_lmm_d.arrayfire;

extern ( C ){

  const string VERSION = "3.4.0";
  const int AF_VERSION_MAJOR = 3;
  const int AF_VERSION_MINOR = 4;
  const int AF_VERSION_PATCH = 0;
  const int AF_API_VERSION_CURRENT = 34;

  int AF_API_VERSION = AF_API_VERSION_CURRENT;

  alias void * af_array;

  alias long dim_t;

  enum af_dtype{
    f32,    ///< 32-bit floating point values
    c32,    ///< 32-bit complex floating point values
    f64,    ///< 64-bit complex floating point values
    c64,    ///< 64-bit complex floating point values
    b8,    ///< 8-bit boolean values
    s32,    ///< 32-bit signed integral values
    u32,    ///< 32-bit unsigned integral values
    u8,    ///< 8-bit unsigned integral values
    s64,    ///< 64-bit signed integral values
    u64,    ///< 64-bit unsigned integral values
    s16,    ///< 16-bit signed integral values
    u16,    ///< 16-bit unsigned integral values
  } ;


  enum af_err{
    ///
    /// The function returned successfully
    ///
    AF_SUCCESS            =   0,

    // 100-199 Errors in environment

    ///
    /// The system or device ran out of memory
    ///
    AF_ERR_NO_MEM         = 101,

    ///
    /// There was an error in the device driver
    ///
    AF_ERR_DRIVER         = 102,

    ///
    /// There was an error with the runtime environment
    ///
    AF_ERR_RUNTIME        = 103,

    // 200-299 Errors in input parameters

    ///
    /// The input array is not a valid af_array object
    ///
    AF_ERR_INVALID_ARRAY  = 201,

    ///
    /// One of the function arguments is incorrect
    ///
    AF_ERR_ARG            = 202,

    ///
    /// The size is incorrect
    ///
    AF_ERR_SIZE           = 203,

    ///
    /// The type is not suppported by this function
    ///
    AF_ERR_TYPE           = 204,

    ///
    /// The type of the input arrays are not compatible
    ///
    AF_ERR_DIFF_TYPE      = 205,

    ///
    /// Function does not support GFOR / batch mode
    ///
    AF_ERR_BATCH          = 207,


    //if(AF_API_VERSION >= 33){
      ///
      /// Input does not belong to the current device.
      ///
      AF_ERR_DEVICE         = 208,
    //}

    // 300-399 Errors for missing software features

    ///
    /// The option is not supported
    ///
    AF_ERR_NOT_SUPPORTED  = 301,

    ///
    /// This build of ArrayFire does not support this feature
    ///
    AF_ERR_NOT_CONFIGURED = 302,

    //if(AF_API_VERSION >= 32){
      ///
      /// This build of ArrayFire is not compiled with "nonfree" algorithms
      ///
      AF_ERR_NONFREE        = 303,
    //}

    // 400-499 Errors for missing hardware features

    ///
    /// This device does not support double
    ///
    AF_ERR_NO_DBL         = 401,

    ///
    /// This build of ArrayFire was not built with graphics or this device does
    /// not support graphics
    ///
    AF_ERR_NO_GFX         = 402,

    // 500-599 Errors specific to heterogenous API

    //if(AF_API_VERSION >= 32){
      ///
      /// There was an error when loading the libraries
      ///
      AF_ERR_LOAD_LIB       = 501,
    //}

    //if(AF_API_VERSION >= 32){
      ///
      /// There was an error when loading the symbols
      ///
      AF_ERR_LOAD_SYM       = 502,
    //}

    //if(AF_API_VERSION >= 32){
      ///
      /// There was a mismatch between the input array and the active backend
      ///
      AF_ERR_ARR_BKND_MISMATCH    = 503,
    //}

    // 900-999 Errors from upstream libraries and runtimes

    ///
    /// There was an internal error either in ArrayFire or in a project
    /// upstream
    ///
    AF_ERR_INTERNAL       = 998,

    ///
    /// Unknown Error
    ///
    AF_ERR_UNKNOWN        = 999
  };

  enum af_source{
    afDevice,   ///< Device pointer
    afHost,     ///< Host pointer
  };

  const uint AF_MAX_DIMS = 4;

  enum af_interp_type{
    AF_INTERP_NEAREST,         ///< Nearest Interpolation
    AF_INTERP_LINEAR,          ///< Linear Interpolation
    AF_INTERP_BILINEAR,        ///< Bilinear Interpolation
    AF_INTERP_CUBIC,           ///< Cubic Interpolation
    AF_INTERP_LOWER,           ///< Floor Indexed
    //#if AF_API_VERSION >= 34
      AF_INTERP_LINEAR_COSINE,   ///< Linear Interpolation with cosine smoothing
    //#endif
    //#if AF_API_VERSION >= 34
      AF_INTERP_BILINEAR_COSINE, ///< Bilinear Interpolation with cosine smoothing
    //#endif
    //#if AF_API_VERSION >= 34
      AF_INTERP_BICUBIC,         ///< Bicubic Interpolation
    //#endif
    //#if AF_API_VERSION >= 34
      AF_INTERP_CUBIC_SPLINE,    ///< Cubic Interpolation with Catmull-Rom splines
    //#endif
    //#if AF_API_VERSION >= 34
      AF_INTERP_BICUBIC_SPLINE,  ///< Bicubic Interpolation with Catmull-Rom splines
    //#endif
  };

  enum af_border_type{
    ///
    /// Out of bound values are 0
    ///
    AF_PAD_ZERO = 0,

    ///
    /// Out of bound values are symmetric over the edge
    ///
    AF_PAD_SYM
  };

  enum af_connectivity{
    ///
    /// Connectivity includes neighbors, North, East, South and West of current pixel
    ///
    AF_CONNECTIVITY_4 = 4,

    ///
    /// Connectivity includes 4-connectivity neigbors and also those on Northeast, Northwest, Southeast and Southwest
    ///
    AF_CONNECTIVITY_8 = 8
  };

  enum af_conv_mode{

    ///
    /// Output of the convolution is the same size as input
    ///
    AF_CONV_DEFAULT,

    ///
    /// Output of the convolution is signal_len + filter_len - 1
    ///
    AF_CONV_EXPAND,
  } ;

  enum af_conv_domain{
    AF_CONV_AUTO,    ///< ArrayFire automatically picks the right convolution algorithm
    AF_CONV_SPATIAL, ///< Perform convolution in spatial domain
    AF_CONV_FREQ,    ///< Perform convolution in frequency domain
  };

  enum af_match_type{
    AF_SAD = 0,   ///< Match based on Sum of Absolute Differences (SAD)
    AF_ZSAD,      ///< Match based on Zero mean SAD
    AF_LSAD,      ///< Match based on Locally scaled SAD
    AF_SSD,       ///< Match based on Sum of Squared Differences (SSD)
    AF_ZSSD,      ///< Match based on Zero mean SSD
    AF_LSSD,      ///< Match based on Locally scaled SSD
    AF_NCC,       ///< Match based on Normalized Cross Correlation (NCC)
    AF_ZNCC,      ///< Match based on Zero mean NCC
    AF_SHD        ///< Match based on Sum of Hamming Distances (SHD)
  };

  //#if AF_API_VERSION >= 31
    enum af_ycc_std{
      AF_YCC_601 = 601,  ///< ITU-R BT.601 (formerly CCIR 601) standard
      AF_YCC_709 = 709,  ///< ITU-R BT.709 standard
      AF_YCC_2020 = 2020  ///< ITU-R BT.2020 standard
    };
  //#endif

  enum af_cspace_t{
    AF_GRAY = 0, ///< Grayscale
    AF_RGB,      ///< 3-channel RGB
    AF_HSV,      ///< 3-channel HSV
  //#if AF_API_VERSION >= 31
      AF_YCbCr     ///< 3-channel YCbCr
  //#endif
  };

  enum af_mat_prop{
    AF_MAT_NONE       = 0,    ///< Default
    AF_MAT_TRANS      = 1,    ///< Data needs to be transposed
    AF_MAT_CTRANS     = 2,    ///< Data needs to be conjugate tansposed
    AF_MAT_CONJ       = 4,    ///< Data needs to be conjugate
    AF_MAT_UPPER      = 32,   ///< Matrix is upper triangular
    AF_MAT_LOWER      = 64,   ///< Matrix is lower triangular
    AF_MAT_DIAG_UNIT  = 128,  ///< Matrix diagonal contains unitary values
    AF_MAT_SYM        = 512,  ///< Matrix is symmetric
    AF_MAT_POSDEF     = 1024, ///< Matrix is positive definite
    AF_MAT_ORTHOG     = 2048, ///< Matrix is orthogonal
    AF_MAT_TRI_DIAG   = 4096, ///< Matrix is tri diagonal
    AF_MAT_BLOCK_DIAG = 8192  ///< Matrix is block diagonal
  };

  enum af_norm_type{
    AF_NORM_VECTOR_1,      ///< treats the input as a vector and returns the sum of absolute values
    AF_NORM_VECTOR_INF,    ///< treats the input as a vector and returns the max of absolute values
    AF_NORM_VECTOR_2,      ///< treats the input as a vector and returns euclidean norm
    AF_NORM_VECTOR_P,      ///< treats the input as a vector and returns the p-norm
    AF_NORM_MATRIX_1,      ///< return the max of column sums
    AF_NORM_MATRIX_INF,    ///< return the max of row sums
    AF_NORM_MATRIX_2,      ///< returns the max singular value). Currently NOT SUPPORTED
    AF_NORM_MATRIX_L_PQ,   ///< returns Lpq-norm

    AF_NORM_EUCLID = AF_NORM_VECTOR_2, ///< The default. Same as AF_NORM_VECTOR_2
  };

  //#if AF_API_VERSION >= 31
    enum af_image_format{
      AF_FIF_BMP          = 0,    ///< FreeImage Enum for Bitmap File
      AF_FIF_ICO          = 1,    ///< FreeImage Enum for Windows Icon File
      AF_FIF_JPEG         = 2,    ///< FreeImage Enum for JPEG File
      AF_FIF_JNG          = 3,    ///< FreeImage Enum for JPEG Network Graphics File
      AF_FIF_PNG          = 13,   ///< FreeImage Enum for Portable Network Graphics File
      AF_FIF_PPM          = 14,   ///< FreeImage Enum for Portable Pixelmap (ASCII) File
      AF_FIF_PPMRAW       = 15,   ///< FreeImage Enum for Portable Pixelmap (Binary) File
      AF_FIF_TIFF         = 18,   ///< FreeImage Enum for Tagged Image File Format File
      AF_FIF_PSD          = 20,   ///< FreeImage Enum for Adobe Photoshop File
      AF_FIF_HDR          = 26,   ///< FreeImage Enum for High Dynamic Range File
      AF_FIF_EXR          = 29,   ///< FreeImage Enum for ILM OpenEXR File
      AF_FIF_JP2          = 31,   ///< FreeImage Enum for JPEG-2000 File
      AF_FIF_RAW          = 34    ///< FreeImage Enum for RAW Camera Image File
    };
  //#endif

  //#if AF_API_VERSION >=34
    enum af_moment_type{
      AF_MOMENT_M00 = 1,
      AF_MOMENT_M01 = 2,
      AF_MOMENT_M10 = 4,
      AF_MOMENT_M11 = 8,
      AF_MOMENT_FIRST_ORDER = AF_MOMENT_M00 | AF_MOMENT_M01 | AF_MOMENT_M10 | AF_MOMENT_M11
    };
  //#endif

  //#if AF_API_VERSION >= 32
    enum af_homography_type{
      AF_HOMOGRAPHY_RANSAC = 0,   ///< Computes homography using RANSAC
      AF_HOMOGRAPHY_LMEDS  = 1    ///< Computes homography using Least Median of Squares
    };
  //#endif

  //#if AF_API_VERSION >= 32
    // These enums should be 2^x
    enum af_backend{
      AF_BACKEND_DEFAULT = 0,  ///< Default backend order: OpenCL -> CUDA -> CPU
      AF_BACKEND_CPU     = 1,  ///< CPU a.k.a sequential algorithms
      AF_BACKEND_CUDA    = 2,  ///< CUDA Compute Backend
      AF_BACKEND_OPENCL  = 4,  ///< OpenCL Compute Backend
    };
  //#endif

  // Below enum is purely added for example purposes
  // it doesn't and shoudn't be used anywhere in the
  // code. No Guarantee's provided if it is used.
  enum af_someenum_t{
      AF_ID = 0
  };

  //#if AF_API_VERSION >=34
    enum af_binary_op{
        AF_BINARY_ADD  = 0,
        AF_BINARY_MUL  = 1,
        AF_BINARY_MIN  = 2,
        AF_BINARY_MAX  = 3
    };
  //#endif

  //#if AF_API_VERSION >=34
    enum af_random_engine_type{
      AF_RANDOM_ENGINE_PHILOX_4X32_10     = 100,                                  //Philox variant with N = 4, W = 32 and Rounds = 10
      AF_RANDOM_ENGINE_THREEFRY_2X32_16   = 200,                                  //Threefry variant with N = 2, W = 32 and Rounds = 16
      AF_RANDOM_ENGINE_MERSENNE_GP11213   = 300,                                  //Mersenne variant with MEXP = 11213
      AF_RANDOM_ENGINE_PHILOX             = AF_RANDOM_ENGINE_PHILOX_4X32_10,      //Resolves to Philox 4x32_10
      AF_RANDOM_ENGINE_THREEFRY           = AF_RANDOM_ENGINE_THREEFRY_2X32_16,    //Resolves to Threefry 2X32_16
      AF_RANDOM_ENGINE_MERSENNE           = AF_RANDOM_ENGINE_MERSENNE_GP11213,    //Resolves to Mersenne GP 11213
      AF_RANDOM_ENGINE_DEFAULT            = AF_RANDOM_ENGINE_PHILOX               //Resolves to Philox
    };
  //#endif

////////////////////////////////////////////////////////////////////////////////
// FORGE / Graphics Related Enums
// These enums have values corresponsding to Forge enums in forge defines.h
////////////////////////////////////////////////////////////////////////////////
  enum af_colormap{
      AF_COLORMAP_DEFAULT = 0,    ///< Default grayscale map
      AF_COLORMAP_SPECTRUM= 1,    ///< Spectrum map
      AF_COLORMAP_COLORS  = 2,    ///< Colors
      AF_COLORMAP_RED     = 3,    ///< Red hue map
      AF_COLORMAP_MOOD    = 4,    ///< Mood map
      AF_COLORMAP_HEAT    = 5,    ///< Heat map
      AF_COLORMAP_BLUE    = 6     ///< Blue hue map
  };

  //#if AF_API_VERSION >= 32
    enum af_marker_type{
      AF_MARKER_NONE         = 0,
      AF_MARKER_POINT        = 1,
      AF_MARKER_CIRCLE       = 2,
      AF_MARKER_SQUARE       = 3,
      AF_MARKER_TRIANGLE     = 4,
      AF_MARKER_CROSS        = 5,
      AF_MARKER_PLUS         = 6,
      AF_MARKER_STAR         = 7
    };
  //#endif
////////////////////////////////////////////////////////////////////////////////

  //#if AF_API_VERSION >= 34
    enum af_storage{
      AF_STORAGE_DENSE     = 0,   ///< Storage type is dense
      AF_STORAGE_CSR       = 1,   ///< Storage type is CSR
      AF_STORAGE_CSC       = 2,   ///< Storage type is CSC
      AF_STORAGE_COO       = 3,   ///< Storage type is COO
    };
  //#endif

  af_err af_create_array(af_array *arr, const void * data, const uint ndims, const dim_t * dims, const af_dtype type);

  af_err af_create_handle(af_array *arr, const uint ndims, const dim_t * dims, const af_dtype type);

  af_err af_copy_array(af_array *arr, const af_array input);

  af_err af_write_array(af_array arr, const void *data, const size_t bytes, af_source src);

  af_err af_get_data_ptr(void *data, const af_array arr);

  af_err af_release_array(af_array arr);

  af_err af_retain_array(af_array *output, const af_array input);

  af_err af_get_data_ref_count(int *use_count, const af_array input);

  af_err af_eval(af_array input);

  af_err af_eval_multiple(const int num, af_array *arrays);

  af_err af_set_manual_eval_flag(bool flag);

  af_err af_get_manual_eval_flag(bool *flag);

  af_err af_get_elements(dim_t *elems, const af_array arr);

  af_err af_get_type(af_dtype *type, const af_array arr);

  af_err af_get_dims(dim_t *d0, dim_t *d1, dim_t *d2, dim_t *d3, const af_array arr);

  af_err af_get_numdims(uint *result, const af_array arr);

  af_err af_is_empty        (bool *result, const af_array arr);

  af_err af_is_scalar       (bool *result, const af_array arr);

  af_err af_is_row          (bool *result, const af_array arr);

  af_err af_is_column       (bool *result, const af_array arr);

  af_err af_is_vector       (bool *result, const af_array arr);

  af_err af_is_complex      (bool *result, const af_array arr);

  af_err af_is_real         (bool *result, const af_array arr);

  af_err af_is_double       (bool *result, const af_array arr);

  af_err af_is_single       (bool *result, const af_array arr);

  af_err af_is_realfloating (bool *result, const af_array arr);

  af_err af_is_floating     (bool *result, const af_array arr);

  af_err af_is_integer      (bool *result, const af_array arr);

  af_err af_is_bool         (bool *result, const af_array arr);

  af_err af_is_sparse       (bool *result, const af_array arr);

  //BLAS Routines

  af_err af_matmul(af_array *output , const af_array lhs, const af_array rhs, const af_mat_prop optLhs, const af_mat_prop optRhs);

  af_err af_dot(af_array *output, const af_array lhs, const af_array rhs, const af_mat_prop optLhs, const af_mat_prop optRhs);

  af_err af_transpose(af_array *output, af_array input, const bool conjugate);

  af_err af_transpose_inplace(af_array input, const bool conjugate);


  // LAPACK Routines

  af_err af_svd(af_array *u, af_array *s, af_array *vt, const af_array input);

  af_err af_svd_inplace(af_array *u, af_array *s, af_array *vt, af_array input);

  af_err af_lu(af_array *lower, af_array *upper, af_array *pivot, const af_array input);

  af_err af_lu_inplace(af_array *pivot, af_array input, const bool is_lapack_piv);

  af_err af_qr(af_array *q, af_array *r, af_array *tau, const af_array input);

  af_err af_qr_inplace(af_array *tau, af_array input);

  af_err af_cholesky(af_array *output, int *info, const af_array input, const bool is_upper);

  af_err af_cholesky_inplace(int *info, af_array input, const bool is_upper);

  af_err af_solve(af_array *x, const af_array a, const af_array b, const af_mat_prop options);

  af_err af_solve_lu(af_array *x, const af_array a, const af_array piv, const af_array b, const af_mat_prop options);

  af_err af_inverse(af_array *output, const af_array input, const af_mat_prop options);

  af_err af_rank(uint *rank, const af_array input, const double tol);

  af_err af_det(double *det_real, double *det_imag, const af_array input);

  af_err af_norm(double *output, const af_array input, const af_norm_type type, const double p, const double q);

  af_err af_is_lapack_available(bool *output);


  void af_print_array(af_array arr);

}

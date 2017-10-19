/*
   This code is part of faster_lmm_d and published under the GPLv3
   License (see LICENSE.txt)

   Copyright © 2017 Prasun Anand & Pjotr Prins
*/

module faster_lmm_d.gemma_lmm;

import core.stdc.stdlib : exit;

import std.conv;
import std.exception;
import std.math;
import std.algorithm: min, max, reduce;
alias mlog = std.math.log;
import std.process;
import std.stdio;
import std.typecons;
import std.experimental.logger;
import std.string;

import faster_lmm_d.dmatrix;
import faster_lmm_d.optmatrix;
import faster_lmm_d.gemma_param;
import gsl.permutation;

import gsl.cdf;
import gsl.errno;
import gsl.math;
import gsl.min;
import gsl.roots;

import progress.bar;

DMatrix read_matrix_from_file2(string filename){
  string input = to!string(std.file.read(filename));

  string[] lines = input.split("\n");
  size_t cols;
  size_t rows = lines.length - 1;
  double[] elements;
  foreach(linex; lines[0..$-1]){
    auto line = linex.strip();
    string[] items = line.split("\t");
    foreach(item; items){
      elements ~= to!double(item) ;
    }
  }
  return DMatrix([rows, elements.length/rows], elements);
}

void CenterMatrix(ref DMatrix G) {
  //DMatrix Gw = gsl_vector_alloc(G.shape[0]);
  writeln("CenterMatrix");
  DMatrix w = ones_dmatrix(1, G.shape[0]);

  DMatrix Gw = matrix_mult(G, w);
  double alpha = -1.0 / to!double(G.shape[0]);
  DMatrix G1 =  multiply_dmatrix_num(matrix_mult(Gw, w.T), alpha);
  DMatrix G2 =  multiply_dmatrix_num(matrix_mult(w, Gw.T), alpha);
  G = add_dmatrix(add_dmatrix(G1, G2), G);

  double alpha_2 = -1.0 / (to!double(G.shape[0]* to!double(G.shape[0])));
  double d = matrix_mult(w, Gw.T).elements[0];
  alpha = d / (to!double(G.shape[0]) * to!double(G.shape[0]));
  G1 =  multiply_dmatrix_num(matrix_mult(w, w.T), alpha);
  G = add_dmatrix(G1, G);

  for (size_t i = 0; i < G.shape[0]; ++i) {
    for (size_t j = 0; j < i; ++j) {
      d = accessor(G, j, i);
      G.elements[G.cols * i + j] = d;
    }
  }
  writeln(sqrt(to!double(G.elements.length)));

  return;
}

// Eigenvalue decomposition, matrix A is destroyed. Returns eigenvalues in
// 'eval'. Also returns matrix 'evec' (U).
void lapack_eigen_symmv(ref DMatrix A, ref DMatrix eval, ref DMatrix evec,
                        const size_t flag_largematrix) {
  if (flag_largematrix == 1) {
    int N = to!int(A.shape[0]), LDA = to!int(A.shape[0]), INFO, LWORK = -1;
    char JOBZ = 'V', UPLO = 'L';

    if (N != A.shape[1] || N != eval.elements.length) {
      writeln("Matrix needs to be symmetric and same dimension in lapack_eigen_symmv.");
      return;
    }

    LWORK = 3 * N;
    //double[LWORK] WORK;
    double[] WORK;
    INFO = LAPACKE_dsyev(101, JOBZ, UPLO, N, A.elements.ptr, LDA, eval.elements.ptr);
      //WORK, &LWORK, &INFO);
    if (INFO != 0) {
      writeln("Eigen decomposition unsuccessful in lapack_eigen_symmv.");
      return;
    }

    DMatrix A_sub = get_sub_dmatrix(A, 0, 0, N, N);
    evec.elements = A_sub.elements.dup;
    evec = evec.T;
  } else {
    int N = to!int(A.shape[0]), LDA = to!int(A.shape[0]), LDZ = to!int(A.shape[0]), INFO;
    int LWORK = -1, LIWORK = -1;
    char JOBZ = 'V', UPLO = 'L', RANGE = 'A';
    double ABSTOL = 1.0E-7;

    // VL, VU, IL, IU are not referenced; M equals N if RANGE='A'.
    double VL = 0.0, VU = 0.0;
    int IL = 0, IU = 0, M;

    if (N != A.shape[1] || N != eval.elements.length) {
      writeln("Matrix needs to be symmetric and same dimension in lapack_eigen_symmv.");
      return;
    }

    //int[2 * N] ISUPPZ;
    int[] ISUPPZ;

    double[1] WORK_temp;
    int[1] IWORK_temp;

    INFO = LAPACKE_dsyevr(101, JOBZ, RANGE, UPLO, N, A.elements.ptr, LDA, VL, VU, IL, IU,
            ABSTOL, &M, eval.elements.ptr, evec.elements.ptr, LDZ, ISUPPZ.ptr);
    //, WORK_temp,        &LWORK, IWORK_temp, &LIWORK, &INFO);
    if (INFO != 0) {
      writeln("Work space estimate unsuccessful in lapack_eigen_symmv.");
      return;
    }
    LWORK = to!int(WORK_temp[0]);
    LIWORK = to!int(IWORK_temp[0]);

    double[] WORK;
     //= new double[LWORK];
    int[] IWORK;
    //= new int[LIWORK];

    INFO = LAPACKE_dsyevr(101, JOBZ, RANGE, UPLO, N, A.elements.ptr, LDA, VL, VU, IL, IU,
            ABSTOL, &M, eval.elements.ptr, evec.elements.ptr, LDZ, ISUPPZ.ptr);
    //, WORK, &LWORK, IWORK, &LIWORK, &INFO);
    if (INFO != 0) {
      writeln("Eigen decomposition unsuccessful in lapack_eigen_symmv.");
      return;
    }

    evec = evec.T;

  }

  return;
}

// Does NOT set eigenvalues to be positive. G gets destroyed. Returns
// eigen trace and values in U and eval (eigenvalues).
double EigenDecomp(ref DMatrix G, ref DMatrix U, ref DMatrix eval,
                   const size_t flag_largematrix) {
  lapack_eigen_symmv(G, eval, U, flag_largematrix);

  // Calculate track_G=mean(diag(G)).
  double d = 0.0;
  for (size_t i = 0; i < eval.elements.length; ++i)
    d += eval.elements[i];

  d /= to!double(eval.elements.length);

  return d;
}

// Same as EigenDecomp but zeroes eigenvalues close to zero. When
// negative eigenvalues remain a warning is issued.
double EigenDecomp_Zeroed(ref DMatrix G, ref DMatrix U, ref DMatrix eval,
                          const size_t flag_largematrix) {
  EigenDecomp(G,U,eval,flag_largematrix);
  auto d = 0.0;
  int count_zero_eigenvalues = 0;
  int count_negative_eigenvalues = 0;
  for (size_t i = 0; i < eval.elements.length; i++) {
    if (abs(eval.elements[i]) < EIGEN_MINVALUE)
      eval.elements[i]  =  0.0;
    // checks
    if (eval.elements[i] == 0.0)
      count_zero_eigenvalues += 1;
    if (eval.elements[i] < 0.0) // count smaller than -EIGEN_MINVALUE
      count_negative_eigenvalues += 1;
    d += eval.elements[i];
  }
  d /= to!double(eval.elements.length);
  if (count_zero_eigenvalues > 1) {
    string msg = "Matrix G has ";
    msg ~= to!string(count_zero_eigenvalues);
    msg ~= " eigenvalues close to zero";
    writeln(msg);
  }
  if (count_negative_eigenvalues > 0) {
    writeln("Matrix G has more than one negative eigenvalues!");
  }

  return d;
}

void CalcPab(const size_t n_cvt, const size_t e_mode, const DMatrix Hi_eval,
              const DMatrix Uab, const DMatrix ab, ref DMatrix Pab) {
  size_t index_ab, index_aw, index_bw, index_ww;
  double p_ab;
  double ps_ab, ps_aw, ps_bw, ps_ww;
  Pab = set_zeros_dmatrix(Pab);
  for (size_t p = 0; p <= n_cvt + 1; ++p) {
    for (size_t a = p + 1; a <= n_cvt + 2; ++a) {
      for (size_t b = a; b <= n_cvt + 2; ++b) {
        index_ab = GetabIndex(a, b, n_cvt);
        if (p == 0) {
          DMatrix Uab_col = get_col(Uab, index_ab);
          p_ab = matrix_mult(Hi_eval, Uab_col).elements[0]; 

          if (e_mode != 0) {
            p_ab = ab.elements[index_ab] - p_ab;
          }

          Pab.elements[index_ab] = p_ab;
        } else {
          index_aw = GetabIndex(a, p, n_cvt);
          index_bw = GetabIndex(b, p, n_cvt);
          index_ww = GetabIndex(p, p, n_cvt);

          ps_ab = accessor(Pab, p - 1, index_ab);
          ps_aw = accessor(Pab, p - 1, index_aw);
          ps_bw = accessor(Pab, p - 1, index_bw);
          ps_ww = accessor(Pab, p - 1, index_ww);

          p_ab = ps_ab - ps_aw * ps_bw / ps_ww;
          Pab.elements[p * Pab.cols + index_ab] = p_ab;
        }
      }
    }
  }
  return;
}



void CalcPPab(const size_t n_cvt, const size_t e_mode,
              const DMatrix HiHi_eval, const DMatrix Uab,
              const DMatrix ab, const DMatrix Pab, ref DMatrix PPab) {
  size_t index_ab, index_aw, index_bw, index_ww;
  double p2_ab;
  double ps2_ab, ps_aw, ps_bw, ps_ww, ps2_aw, ps2_bw, ps2_ww;

  PPab = set_zeros_dmatrix(PPab);
  for (size_t p = 0; p <= n_cvt + 1; ++p) {
    for (size_t a = p + 1; a <= n_cvt + 2; ++a) {
      for (size_t b = a; b <= n_cvt + 2; ++b) {
        index_ab = GetabIndex(a, b, n_cvt);
        if (p == 0) {
          DMatrix Uab_col = get_col(Uab, index_ab);
          p2_ab = matrix_mult(HiHi_eval, Uab_col).elements[0];  // check its shape is [1,1] else take transpose of HiHi_eval
          if (e_mode != 0) {
            p2_ab = p2_ab - ab.elements[index_ab] +
                    2.0 * Pab.elements[index_ab];
          }
          PPab.elements[index_ab] = p2_ab;
        } else {
          index_aw = GetabIndex(a, p, n_cvt);
          index_bw = GetabIndex(b, p, n_cvt);
          index_ww = GetabIndex(p, p, n_cvt);

          ps2_ab = accessor(PPab, p - 1, index_ab);
          ps_aw = accessor(Pab, p - 1, index_aw);
          ps_bw = accessor(Pab, p - 1, index_bw);
          ps_ww = accessor(Pab, p - 1, index_ww);
          ps2_aw = accessor(PPab, p - 1, index_aw);
          ps2_bw = accessor(PPab, p - 1, index_bw);
          ps2_ww = accessor(PPab, p - 1, index_ww);

          p2_ab = ps2_ab + ps_aw * ps_bw * ps2_ww / (ps_ww * ps_ww);
          p2_ab -= (ps_aw * ps2_bw + ps_bw * ps2_aw) / ps_ww;
          PPab.elements[p * PPab.cols + index_ab] = p2_ab;
        }
      }
    }
  }
  return;
}

void CalcPPPab(const size_t n_cvt, const size_t e_mode,
               const DMatrix HiHiHi_eval, const DMatrix Uab,
               const DMatrix ab, const DMatrix Pab,
               const DMatrix PPab, ref DMatrix PPPab) {
  size_t index_ab, index_aw, index_bw, index_ww;
  double p3_ab;
  double ps3_ab, ps_aw, ps_bw, ps_ww, ps2_aw, ps2_bw, ps2_ww, ps3_aw, ps3_bw, ps3_ww;
  PPPab = set_zeros_dmatrix(PPPab);

  for (size_t p = 0; p <= n_cvt + 1; ++p) {
    for (size_t a = p + 1; a <= n_cvt + 2; ++a) {
      for (size_t b = a; b <= n_cvt + 2; ++b) {
        index_ab = GetabIndex(a, b, n_cvt);
        if (p == 0) {
          DMatrix Uab_col = get_col(Uab, index_ab);
          p3_ab = matrix_mult(HiHiHi_eval, Uab_col).elements[0];
          if (e_mode != 0) {
            p3_ab = ab.elements[index_ab] - p3_ab +
                    3.0 * accessor(PPab, 0, index_ab) -
                    3.0 * accessor(Pab, 0, index_ab);
          }
          PPPab.elements[0* PPPab.cols + index_ab] = p3_ab;
        } else {
          index_aw = GetabIndex(a, p, n_cvt);
          index_bw = GetabIndex(b, p, n_cvt);
          index_ww = GetabIndex(p, p, n_cvt);
          ps3_ab = accessor(PPPab, p - 1, index_ab);
          ps_aw = accessor(Pab, p - 1, index_aw);
          ps_bw = accessor(Pab, p - 1, index_bw);
          ps_ww = accessor(Pab, p - 1, index_ww);
          ps2_aw = accessor(PPab, p - 1, index_aw);
          ps2_bw = accessor(PPab, p - 1, index_bw);
          ps2_ww = accessor(PPab, p - 1, index_ww);
          ps3_aw = accessor(PPPab, p - 1, index_aw);
          ps3_bw = accessor(PPPab, p - 1, index_bw);
          ps3_ww = accessor(PPPab, p - 1, index_ww);

          p3_ab = ps3_ab -
                  ps_aw * ps_bw * ps2_ww * ps2_ww / (ps_ww * ps_ww * ps_ww);
          p3_ab -= (ps_aw * ps3_bw + ps_bw * ps3_aw + ps2_aw * ps2_bw) / ps_ww;
          p3_ab += (ps_aw * ps2_bw * ps2_ww + ps_bw * ps2_aw * ps2_ww +
                    ps_aw * ps_bw * ps3_ww) /
                   (ps_ww * ps_ww);

          PPPab.elements[p* PPPab.cols + index_ab] = p3_ab;
        }
      }
    }
  }
  return;
}

size_t GetabIndex(size_t a, size_t b, size_t n_cvt) {
  size_t n = n_cvt + 2;
  if (a > n || b > n || a <= 0 || b <= 0) {
    writeln("error in GetabIndex.");
    return 0;
  }

  if (b < a) {
    size_t temp = b;
    b = a;
    a = temp;
  }

  return (2 * n - a + 2) * (a - 1) / 2 + b - a;
}

struct loglikeparam{
  size_t n_cvt;
  size_t ni_test;
  size_t n_index;
  bool calc_null;
  int e_mode;
  DMatrix eval;
  DMatrix Uab;
  DMatrix ab;

  this(bool calc_null, size_t ni_test, size_t n_cvt, DMatrix eval, DMatrix Uab,  DMatrix ab, int e_mode) {
    this.n_cvt = n_cvt;
    this.ni_test = ni_test;
    this.calc_null = calc_null;
    this.e_mode = e_mode;
    this.eval = eval;
    this.Uab = Uab;
    this.ab = ab;
  }
}


double LogL_f(double l, void* params) {

  auto ptr = cast(loglikeparam *)params;
  loglikeparam p = *ptr;

  size_t n_cvt = p.n_cvt;
  size_t ni_test = p.ni_test;
  size_t n_index = (n_cvt + 2 + 1) * (n_cvt + 2) / 2;

  size_t nc_total;
  if (p.calc_null == true) {
    nc_total = n_cvt;
  } else {
    nc_total = n_cvt + 1;
  }

  double f = 0.0;
  double logdet_h = 0.0;
  double d;
  size_t index_yy;

  DMatrix Pab;
  Pab.shape = [n_cvt + 2, n_index];
  DMatrix Hi_eval;
  
  Hi_eval.shape = [1, p.eval.elements.length];
  DMatrix v_temp;
  v_temp.shape = [1, p.eval.elements.length];

  v_temp.elements = p.eval.elements.dup;

  v_temp = multiply_dmatrix_num(v_temp, l);

  if (p.e_mode == 0) {
    Hi_eval = set_ones_dmatrix(Hi_eval);
  } else {
    Hi_eval.elements = v_temp.elements.dup;
  }

  v_temp = add_dmatrix_num(v_temp, 1.0);
  Hi_eval = divide_dmatrix(Hi_eval, v_temp);

  for (size_t i = 0; i < p.eval.elements.length; ++i) {
    d = v_temp.elements[i];
    logdet_h += mlog(fabs(d));
  }

  CalcPab(n_cvt, p.e_mode, Hi_eval, p.Uab, p.ab, Pab);

  double c = 0.5 * to!double(ni_test) * (mlog(to!double(ni_test)) - mlog(2 * M_PI) - 1.0);

  index_yy = GetabIndex(n_cvt + 2, n_cvt + 2, n_cvt);
  double P_yy = accessor(Pab, nc_total, index_yy);
  f = c - 0.5 * logdet_h - 0.5 * to!double(ni_test) * mlog(P_yy);

  return f;
}

double LogRL_f(double l, void* params) {

  auto ptr = cast(loglikeparam *)params;
  loglikeparam p = *ptr;

  size_t n_cvt = p.n_cvt;
  size_t ni_test = p.ni_test;
  size_t n_index = (n_cvt + 2 + 1) * (n_cvt + 2) / 2;

  double df;
  size_t nc_total;
  if (p.calc_null == true) {
    nc_total = n_cvt;
    df = to!double(ni_test) - to!double(n_cvt);
  } else {
    nc_total = n_cvt + 1;
    df = to!double(ni_test) - to!double(n_cvt) - 1.0;
  }

  double f = 0.0, logdet_h = 0.0, logdet_hiw = 0.0, d;
  size_t index_ww;

  DMatrix Pab;
  Pab.shape = [n_cvt + 2, n_index];
  DMatrix Iab;
  Iab.shape = [n_cvt + 2, n_index];
  DMatrix Hi_eval;
  Hi_eval.shape = [1, p.eval.elements.length];
  DMatrix v_temp;
  v_temp.shape = [1, p.eval.elements.length];
  v_temp.elements = p.eval.elements.dup;

  v_temp = multiply_dmatrix_num(v_temp, l);
  if (p.e_mode == 0) {
    Hi_eval = set_ones_dmatrix(Hi_eval);
  } else {
    Hi_eval.elements = v_temp.elements.dup;
  }
  v_temp = add_dmatrix_num(v_temp, 1.0);
  Hi_eval = divide_dmatrix(Hi_eval, v_temp);

  for (size_t i = 0; i < p.eval.elements.length; ++i) {
    d = v_temp.elements[i];
    logdet_h += mlog(fabs(d));
  }

  CalcPab(n_cvt, p.e_mode, Hi_eval, p.Uab, p.ab, Pab);

  v_temp = set_ones_dmatrix(v_temp);
  CalcPab(n_cvt, p.e_mode, v_temp, p.Uab, p.ab, Iab);

  // Calculate |WHiW|-|WW|.
  logdet_hiw = 0.0;
  for (size_t i = 0; i < nc_total; ++i) {
    index_ww = GetabIndex(i + 1, i + 1, n_cvt);
    d = accessor(Pab, i, index_ww);
    logdet_hiw += mlog(d);
    d = accessor(Iab, i, index_ww);
    logdet_hiw -= mlog(d);
  }
  index_ww = GetabIndex(n_cvt + 2, n_cvt + 2, n_cvt);
  double P_yy = accessor(Pab, nc_total, index_ww);
  //writeln(P_yy);

  double c = 0.5 * df * (mlog(df) - mlog(2 * M_PI) - 1.0);
  f = c - 0.5 * logdet_h - 0.5 * logdet_hiw - 0.5 * df * mlog(P_yy);
  return f;
}

extern(C) double LogRL_dev1(double l, void* params) {
  auto ptr = cast(loglikeparam *)params;
  loglikeparam p = *ptr;

  size_t n_cvt = p.n_cvt;
  size_t ni_test = p.ni_test;
  size_t n_index = (n_cvt + 2 + 1) * (n_cvt + 2) / 2;

  double df;
  size_t nc_total;
  //p.calc_null = true;   //  check 
  if (p.calc_null == true) {
    nc_total = n_cvt;
    df = to!double(ni_test) - to!double(n_cvt);
  } else {
    nc_total =  n_cvt + 1;
    df =  to!double(ni_test) - to!double(n_cvt) - 1.0;
    
  }

  double dev1 = 0.0, trace_Hi = 0.0;
  size_t index_ww;

  DMatrix Pab;
  Pab.shape = [n_cvt + 2, n_index];

  DMatrix PPab;
  PPab.shape = [n_cvt + 2, n_index];

  DMatrix Hi_eval;
  Hi_eval.shape = [1, p.eval.elements.length];
  DMatrix HiHi_eval;
  HiHi_eval.shape = [1, p.eval.elements.length];
  DMatrix v_temp;
  v_temp.shape = [1, p.eval.elements.length];
  v_temp.elements = p.eval.elements.dup;

  v_temp = multiply_dmatrix_num(v_temp, l);
  if (p.e_mode == 0) {
    Hi_eval = set_ones_dmatrix(Hi_eval);
  } else {
    Hi_eval.elements = v_temp.elements;
  }

  v_temp = add_dmatrix_num(v_temp, 1.0);
  Hi_eval = divide_dmatrix(Hi_eval, v_temp);

  nan_counter(p.Uab);
  nan_counter(p.ab);
  HiHi_eval.elements =  Hi_eval.elements.dup;
  HiHi_eval = slow_multiply_dmatrix(HiHi_eval, Hi_eval);

  v_temp = set_ones_dmatrix(v_temp);
  trace_Hi = matrix_mult(Hi_eval, v_temp.T).elements[0];

  if (p.e_mode != 0) {
    trace_Hi = to!double(ni_test) - trace_Hi;
  }

  CalcPab(n_cvt, p.e_mode, Hi_eval, p.Uab, p.ab, Pab);
  //writeln(Pab);
  CalcPPab(n_cvt, p.e_mode, HiHi_eval, p.Uab, p.ab, Pab, PPab);

  // Calculate tracePK and trace PKPK.
  double trace_P = trace_Hi;
  double ps_ww, ps2_ww;
  for (size_t i = 0; i < nc_total; ++i) {
    index_ww = GetabIndex(i + 1, i + 1, n_cvt);
    ps_ww = accessor(Pab, i, index_ww);
    ps2_ww = accessor(PPab, i, index_ww);
    trace_P -= ps2_ww / ps_ww;
  }
  double trace_PK = (df - trace_P) / l;
  writeln("trace_PK => ", trace_PK );

  // Calculate yPKPy, yPKPKPy.
  index_ww = GetabIndex(n_cvt + 2, n_cvt + 2, n_cvt);
  double P_yy = accessor(Pab, nc_total, index_ww);
  double PP_yy = accessor(PPab, nc_total, index_ww);
  double yPKPy = (P_yy - PP_yy) / l;
  //writeln("nc_total => ", nc_total);
  //writeln("index_ww => ", index_ww);
  //writeln("PP_yy => ", PP_yy );
  //writeln("yPKPy => ", yPKPy);
  //writeln("P_yy => ", P_yy);
  //writeln("PP_yy => ", PP_yy );
  //writeln("yPKPy => ", yPKPy);

  dev1 = -0.5 * trace_PK + 0.5 * df * yPKPy / P_yy;

  return dev1;

}

extern(C) double LogL_dev1(double l, void* params) {
  auto ptr = cast(loglikeparam *)params;
  loglikeparam p = *ptr;

  size_t n_cvt = p.n_cvt;
  size_t ni_test = p.ni_test;
  size_t n_index = (n_cvt + 2 + 1) * (n_cvt + 2) / 2;

  size_t nc_total;

  if (p.calc_null == true) {
    nc_total = n_cvt;
  } else {
    nc_total = n_cvt + 1;
  }

  double dev1 = 0.0, trace_Hi = 0.0;
  size_t index_yy;

  DMatrix Pab;
  Pab.shape = [n_cvt + 2, n_index];

  DMatrix PPab;
  PPab.shape = [n_cvt + 2, n_index];

  DMatrix Hi_eval;
  Hi_eval.shape = [1, p.eval.elements.length];
  DMatrix HiHi_eval;
  HiHi_eval.shape = [1, p.eval.elements.length];
  DMatrix v_temp;
  v_temp.shape = [1, p.eval.elements.length];
  v_temp.elements = p.eval.elements;

  v_temp = multiply_dmatrix_num(v_temp, l);

  if (p.e_mode == 0) {
    Hi_eval = set_ones_dmatrix(Hi_eval);
  } else {
    Hi_eval.elements = v_temp.elements.dup;
  }
  v_temp = add_dmatrix_num(v_temp, 1.0);
  Hi_eval = divide_dmatrix(Hi_eval, v_temp);

  HiHi_eval.elements = Hi_eval.elements.dup;
  HiHi_eval = slow_multiply_dmatrix(HiHi_eval, Hi_eval);

  v_temp = set_ones_dmatrix(v_temp);
  trace_Hi = matrix_mult(Hi_eval, v_temp.T).elements[0];

  if (p.e_mode != 0) {
    trace_Hi = to!double(ni_test) - trace_Hi;
  }

  CalcPab(n_cvt, p.e_mode, Hi_eval, p.Uab, p.ab, Pab);
  CalcPPab(n_cvt, p.e_mode, HiHi_eval, p.Uab, p.ab, Pab, PPab);

  double trace_HiK = (to!double(ni_test) - trace_Hi) / l;

  index_yy = GetabIndex(n_cvt + 2, n_cvt + 2, n_cvt);

  double P_yy = accessor(Pab, nc_total, index_yy);
  double PP_yy = accessor(PPab, nc_total, index_yy);
  double yPKPy = (P_yy - PP_yy) / l;

  dev1 = -0.5 * trace_HiK + 0.5 * to!double(ni_test) * yPKPy / P_yy;

  return dev1;
}

extern(C) double LogRL_dev2(double l, void* params) {

  auto ptr = cast(loglikeparam *)params;
  loglikeparam p = *ptr;

  size_t n_cvt = p.n_cvt;
  size_t ni_test = p.ni_test;
  size_t n_index = (n_cvt + 2 + 1) * (n_cvt + 2) / 2;

  double df;
  size_t nc_total;
  if (p.calc_null == true) {
    nc_total = n_cvt;
    df = to!double(ni_test) - to!double(n_cvt);
  } else {
    nc_total = n_cvt + 1;
    df = to!double(ni_test) - to!double(n_cvt) - 1.0;
  }

  double dev2 = 0.0, trace_Hi = 0.0, trace_HiHi = 0.0;
  size_t index_ww;

  DMatrix Pab;
  Pab.shape = [n_cvt + 2, n_index];

  DMatrix PPab;
  PPab.shape = [n_cvt + 2, n_index];

  DMatrix PPPab;
  PPPab.shape = [n_cvt + 2, n_index];

  DMatrix Hi_eval;
  Hi_eval.shape = [1, p.eval.elements.length];
  DMatrix HiHi_eval;
  HiHi_eval.shape = [1, p.eval.elements.length];
  DMatrix HiHiHi_eval;
  HiHiHi_eval.shape = [1, p.eval.elements.length];
  DMatrix v_temp;
  v_temp.shape = [1, p.eval.elements.length];
  v_temp.elements = p.eval.elements;

  v_temp = multiply_dmatrix_num(v_temp, l);

  if (p.e_mode == 0) {
    Hi_eval = set_ones_dmatrix(Hi_eval);
  } else {
    Hi_eval.elements = v_temp.elements.dup;
  }
  v_temp = add_dmatrix_num(v_temp, 1.0);
  Hi_eval = divide_dmatrix(Hi_eval, v_temp);

  HiHi_eval.elements = Hi_eval.elements.dup;
  HiHi_eval = slow_multiply_dmatrix(HiHi_eval, Hi_eval);

  HiHiHi_eval.elements = HiHi_eval.elements.dup;
  HiHiHi_eval = slow_multiply_dmatrix(HiHiHi_eval, Hi_eval);

  v_temp = set_ones_dmatrix(v_temp);
  trace_Hi = matrix_mult(Hi_eval, v_temp.T).elements[0];
  trace_HiHi = matrix_mult(HiHi_eval, v_temp.T).elements[0];

  if (p.e_mode != 0) {
    trace_Hi = to!double(ni_test) - trace_Hi;
    trace_HiHi = 2 * trace_Hi + trace_HiHi - to!double(ni_test);
  }

  CalcPab(n_cvt, p.e_mode, Hi_eval, p.Uab, p.ab, Pab);
  CalcPPab(n_cvt, p.e_mode, HiHi_eval, p.Uab, p.ab, Pab, PPab);
  CalcPPPab(n_cvt, p.e_mode, HiHiHi_eval, p.Uab, p.ab, Pab, PPab, PPPab);

  // Calculate tracePK and trace PKPK.
  double trace_P = trace_Hi, trace_PP = trace_HiHi;
  double ps_ww, ps2_ww, ps3_ww;
  for (size_t i = 0; i < nc_total; ++i) {
    index_ww = GetabIndex(i + 1, i + 1, n_cvt);
    ps_ww = accessor(Pab, i, index_ww);
    ps2_ww = accessor(PPab, i, index_ww);
    ps3_ww = accessor(PPPab, i, index_ww);
    trace_P -= ps2_ww / ps_ww;
    trace_PP += ps2_ww * ps2_ww / (ps_ww * ps_ww) - 2.0 * ps3_ww / ps_ww;
  }
  double trace_PKPK = (df + trace_PP - 2.0 * trace_P) / (l * l);

  // Calculate yPKPy, yPKPKPy.
  index_ww = GetabIndex(n_cvt + 2, n_cvt + 2, n_cvt);
  double P_yy = accessor(Pab, nc_total, index_ww);
  double PP_yy = accessor(PPab, nc_total, index_ww);
  double PPP_yy = accessor(PPPab, nc_total, index_ww);
  double yPKPy = (P_yy - PP_yy) / l;
  double yPKPKPy = (P_yy + PPP_yy - 2.0 * PP_yy) / (l * l);

  dev2 = 0.5 * trace_PKPK -
         0.5 * df * (2.0 * yPKPKPy * P_yy - yPKPy * yPKPy) / (P_yy * P_yy);

  return dev2;
}

extern(C) double LogL_dev2(double l, void* params) {

  auto ptr = cast(loglikeparam *)params;
  loglikeparam p = *ptr;

  size_t n_cvt = p.n_cvt;
  size_t ni_test = p.ni_test;
  size_t n_index = (n_cvt + 2 + 1) * (n_cvt + 2) / 2;

  size_t nc_total;
  if (p.calc_null == true) {
    nc_total = n_cvt;
  } else {
    nc_total = n_cvt + 1;
  }

  double dev2 = 0.0, trace_Hi = 0.0, trace_HiHi = 0.0;
  size_t index_yy;
  DMatrix Pab;
  Pab.shape = [n_cvt + 2, n_index];

  DMatrix PPab;
  PPab.shape = [n_cvt + 2, n_index];

  DMatrix PPPab;
  PPPab.shape = [n_cvt + 2, n_index];

  DMatrix Hi_eval;
  Hi_eval.shape = [1, p.eval.elements.length];
  DMatrix HiHi_eval;
  HiHi_eval.shape = [1, p.eval.elements.length];
  DMatrix HiHiHi_eval;
  HiHiHi_eval.shape = [1, p.eval.elements.length];
  DMatrix v_temp;
  v_temp.shape = [1, p.eval.elements.length];
  v_temp.elements = p.eval.elements;

  v_temp = multiply_dmatrix_num(v_temp, l);

  if (p.e_mode == 0) {
    Hi_eval = set_ones_dmatrix(Hi_eval);
  } else {
    Hi_eval.elements = v_temp.elements.dup;
  }
  v_temp = add_dmatrix_num(v_temp, 1.0);
  Hi_eval = divide_dmatrix(Hi_eval, v_temp);

  HiHi_eval.elements = Hi_eval.elements.dup;
  HiHi_eval = slow_multiply_dmatrix(HiHi_eval, Hi_eval); // gsl_vector_mul();
  HiHiHi_eval.elements = HiHi_eval.elements.dup;
  HiHiHi_eval = slow_multiply_dmatrix(HiHiHi_eval, Hi_eval);

  v_temp = set_ones_dmatrix(v_temp);

  trace_Hi = matrix_mult(Hi_eval, v_temp.T).elements[0];

  trace_HiHi = matrix_mult(HiHi_eval, v_temp.T).elements[0];
  if (p.e_mode != 0) {
    trace_Hi = to!double(ni_test) - trace_Hi;
    trace_HiHi = 2 * trace_Hi + trace_HiHi - to!double(ni_test);
  }

  CalcPab(n_cvt, p.e_mode, Hi_eval, p.Uab, p.ab, Pab);
  CalcPPab(n_cvt, p.e_mode, HiHi_eval, p.Uab, p.ab, Pab, PPab);
  CalcPPPab(n_cvt, p.e_mode, HiHiHi_eval, p.Uab, p.ab, Pab, PPab, PPPab);

  double trace_HiKHiK = (to!double(ni_test) + trace_HiHi - 2 * trace_Hi) / (l * l);

  index_yy = GetabIndex(n_cvt + 2, n_cvt + 2, n_cvt);
  double P_yy = accessor(Pab, nc_total, index_yy);
  double PP_yy = accessor(PPab, nc_total, index_yy);
  double PPP_yy = accessor(PPPab, nc_total, index_yy);
  double yPKPy = (P_yy - PP_yy) / l;
  double yPKPKPy = (P_yy + PPP_yy - 2.0 * PP_yy) / (l * l);

  dev2 = 0.5 * trace_HiKHiK -
         0.5 * to!double(ni_test) * (2.0 * yPKPKPy * P_yy - yPKPy * yPKPy) /
             (P_yy * P_yy);

  return dev2;
}

extern(C) void LogL_dev12(double l, void *params, double *dev1, double *dev2) {

  //writeln("In LogL_dev12, l =" , l);
  //writeln(l);

  auto ptr = cast(loglikeparam *)params;
  loglikeparam p = *ptr;

  size_t n_cvt = p.n_cvt;
  size_t ni_test = p.ni_test;
  size_t n_index = (n_cvt + 2 + 1) * (n_cvt + 2) / 2;

  size_t nc_total;
  if (p.calc_null == true) {
    nc_total = n_cvt;
  } else {
    nc_total = n_cvt + 1;
  }

  double trace_Hi = 0.0, trace_HiHi = 0.0;
  size_t index_yy;

  DMatrix Pab;
  Pab.shape = [n_cvt + 2, n_index];

  DMatrix PPab;
  PPab.shape = [n_cvt + 2, n_index];

  DMatrix PPPab;
  PPPab.shape = [n_cvt + 2, n_index];

  DMatrix Hi_eval;
  Hi_eval.shape = [1, p.eval.elements.length];
  DMatrix HiHi_eval;
  HiHi_eval.shape = [1, p.eval.elements.length];
  DMatrix HiHiHi_eval;
  HiHiHi_eval.shape = [1, p.eval.elements.length];
  DMatrix v_temp;
  v_temp.shape = [1, p.eval.elements.length];
  v_temp.elements = p.eval.elements.dup;

  v_temp = multiply_dmatrix_num(v_temp, l);

  if (p.e_mode == 0) {
    Hi_eval = set_ones_dmatrix(Hi_eval);
  } else {
    Hi_eval.elements = v_temp.elements.dup;
  }

  v_temp = add_dmatrix_num(v_temp, 1.0);
  Hi_eval = divide_dmatrix(Hi_eval, v_temp);

  HiHi_eval.elements = Hi_eval.elements.dup;
  HiHi_eval = slow_multiply_dmatrix(HiHi_eval, Hi_eval);
  HiHiHi_eval.elements = HiHi_eval.elements.dup;
  HiHiHi_eval = slow_multiply_dmatrix(HiHiHi_eval, Hi_eval);

  v_temp = set_ones_dmatrix(v_temp);
  trace_Hi = matrix_mult(Hi_eval, v_temp.T).elements[0];
  trace_HiHi = matrix_mult(HiHi_eval, v_temp.T).elements[0];

  if (p.e_mode != 0) {
    trace_Hi = to!double(ni_test) - trace_Hi;
    trace_HiHi = 2 * trace_Hi + trace_HiHi - to!double(ni_test);
  }

  CalcPab(n_cvt, p.e_mode, Hi_eval, p.Uab, p.ab, Pab);
  CalcPPab(n_cvt, p.e_mode, HiHi_eval, p.Uab, p.ab, Pab, PPab);
  CalcPPPab(n_cvt, p.e_mode, HiHiHi_eval, p.Uab, p.ab, Pab, PPab, PPPab);

  double trace_HiK = (to!double(ni_test) - trace_Hi) / l;
  double trace_HiKHiK = (to!double(ni_test) + trace_HiHi - 2 * trace_Hi) / (l * l);

  index_yy = GetabIndex(n_cvt + 2, n_cvt + 2, n_cvt);

  double P_yy = accessor(Pab, nc_total, index_yy);
  double PP_yy = accessor(PPab, nc_total, index_yy);
  double PPP_yy = accessor(PPPab, nc_total, index_yy);

  double yPKPy = (P_yy - PP_yy) / l;
  double yPKPKPy = (P_yy + PPP_yy - 2.0 * PP_yy) / (l * l);

  *dev1 = -0.5 * trace_HiK + 0.5 * to!double(ni_test) * yPKPy / P_yy;
  *dev2 = 0.5 * trace_HiKHiK -
          0.5 * to!double(ni_test) * (2.0 * yPKPKPy * P_yy - yPKPy * yPKPy) /
              (P_yy * P_yy);

  return;
}

extern(C) void LogRL_dev12(double l, void* params, double* dev1, double* dev2) {

  auto ptr = cast(loglikeparam *)params;
  loglikeparam p = *ptr;

  size_t n_cvt = p.n_cvt;
  size_t ni_test = p.ni_test;
  size_t n_index = (n_cvt + 2 + 1) * (n_cvt + 2) / 2;

  double df;
  size_t nc_total;
  if (p.calc_null == true) {
    nc_total = n_cvt;
    df = to!double(ni_test) - to!double(n_cvt);
  } else {
    nc_total = n_cvt + 1;
    df = to!double(ni_test) - to!double(n_cvt) - 1.0;
  }

  double trace_Hi = 0.0, trace_HiHi = 0.0;
  size_t index_ww;

  DMatrix Pab;
  Pab.shape = [n_cvt + 2, n_index];

  DMatrix PPab;
  PPab.shape = [n_cvt + 2, n_index];

  DMatrix PPPab;
  PPPab.shape = [n_cvt + 2, n_index];

  DMatrix Hi_eval;
  Hi_eval.shape = [1, p.eval.elements.length];
  DMatrix HiHi_eval;
  HiHi_eval.shape = [1, p.eval.elements.length];
  DMatrix HiHiHi_eval;
  HiHiHi_eval.shape = [1, p.eval.elements.length];
  DMatrix v_temp;
  v_temp.shape = [1, p.eval.elements.length];
  v_temp.elements = p.eval.elements;

  v_temp = multiply_dmatrix_num(v_temp, l);

  if (p.e_mode == 0) {
    Hi_eval = set_ones_dmatrix(Hi_eval);
  } else {
    Hi_eval.elements = v_temp.elements.dup;
  }
  v_temp = add_dmatrix_num(v_temp, 1.0);
  Hi_eval = divide_dmatrix(Hi_eval, v_temp);


  HiHi_eval.elements = Hi_eval.elements.dup;
  HiHi_eval = slow_multiply_dmatrix(HiHi_eval, Hi_eval);
  HiHiHi_eval.elements = HiHi_eval.elements.dup;
  HiHiHi_eval = slow_multiply_dmatrix(HiHiHi_eval, Hi_eval);

  v_temp = set_ones_dmatrix(v_temp);
  trace_Hi = matrix_mult(Hi_eval, v_temp.T).elements[0];
  trace_HiHi = matrix_mult(HiHi_eval, v_temp.T).elements[0];

  if (p.e_mode != 0) {
    trace_Hi = to!double(ni_test) - trace_Hi;
    trace_HiHi = 2 * trace_Hi + trace_HiHi - to!double(ni_test);
  }

  CalcPab(n_cvt, p.e_mode, Hi_eval, p.Uab, p.ab, Pab);
  CalcPPab(n_cvt, p.e_mode, HiHi_eval, p.Uab, p.ab, Pab, PPab);
  CalcPPPab(n_cvt, p.e_mode, HiHiHi_eval, p.Uab, p.ab, Pab, PPab, PPPab);

  // Calculate tracePK and trace PKPK.
  double trace_P = trace_Hi, trace_PP = trace_HiHi;

  double ps_ww, ps2_ww, ps3_ww;
  for (size_t i = 0; i < nc_total; ++i) {
    index_ww = GetabIndex(i + 1, i + 1, n_cvt);
    ps_ww = accessor(Pab, i, index_ww);
    ps2_ww = accessor(PPab, i, index_ww);
    ps3_ww = accessor(PPPab, i, index_ww);
    trace_P -= ps2_ww / ps_ww;
    trace_PP += ps2_ww * ps2_ww / (ps_ww * ps_ww) - 2.0 * ps3_ww / ps_ww;
  }
  double trace_PK = (df - trace_P) / l;
  double trace_PKPK = (df + trace_PP - 2.0 * trace_P) / (l * l);

  // Calculate yPKPy, yPKPKPy.
  index_ww = GetabIndex(n_cvt + 2, n_cvt + 2, n_cvt);
  double P_yy = accessor(Pab, nc_total, index_ww);
  double PP_yy = accessor(PPab, nc_total, index_ww);
  double PPP_yy = accessor(PPPab, nc_total, index_ww);
  double yPKPy = (P_yy - PP_yy) / l;
  double yPKPKPy = (P_yy + PPP_yy - 2.0 * PP_yy) / (l * l);

  *dev1 = -0.5 * trace_PK + 0.5 * df * yPKPy / P_yy;
  *dev2 = 0.5 * trace_PKPK -
          0.5 * df * (2.0 * yPKPKPy * P_yy - yPKPy * yPKPy) / (P_yy * P_yy);

  return;
}

alias Tuple!(double,"l",double,"h") Lambda_tup;

void CalcLambda(const char func_name, void* params, const double l_min,
                const double l_max, const size_t n_region, ref double lambda,
                ref double logf) {
  //writeln("in CalcLambda for NOT-NULL");
  if (func_name != 'R' && func_name != 'L' && func_name != 'r' &&
      func_name != 'l') {
    writeln("func_name only takes 'R' or 'L': 'R' for
            log-restricted likelihood, 'L' for log-likelihood.");
    return;
  }

  Lambda_tup[] lambda_lh;

  // Evaluate first-order derivates in different intervals.
  double lambda_l, lambda_h;
  double lambda_interval = mlog(l_max / l_min) / to!double(n_region);
  double dev1_l, dev1_h, logf_l, logf_h;
  //writeln("lambda_interval = ", lambda_interval);
  for (size_t i = 0; i < n_region; ++i) {
    lambda_l = l_min * exp(lambda_interval * i);
    lambda_h = l_min * exp(lambda_interval * (i + 1.0));

    if (func_name == 'R' || func_name == 'r') {
      dev1_l = LogRL_dev1(lambda_l, params);
      dev1_h = LogRL_dev1(lambda_h, params);
    } else {
      dev1_l = LogL_dev1(lambda_l, params);
      dev1_h = LogL_dev1(lambda_h, params);
    }

     writeln("dev1_l = ", dev1_l);
    writeln("dev1_h = ", dev1_h);
    if (dev1_l * dev1_h <= 0) {
      writeln("lambda_lh size up");
      lambda_lh ~= Lambda_tup(lambda_l, lambda_h);
    }
  }

  // If derivates do not change signs in any interval.
  if (lambda_lh.length == 0) {
    if (func_name == 'R' || func_name == 'r') {
      logf_l = LogRL_f(l_min, params);
      logf_h = LogRL_f(l_max, params);
    } else {
      logf_l = LogL_f(l_min, params);
      logf_h = LogL_f(l_max, params);
    }

    if (logf_l >= logf_h) {
      lambda = l_min;
      logf = logf_l;
    } else {
      lambda = l_max;
      logf = logf_h;
    }
  } else {
    // If derivates change signs.
    int status;
    int iter = 0, max_iter = 100;
    double l, l_temp;

    gsl_function F;
    gsl_function_fdf FDF;

    F.params = params;
    FDF.params = params;

    if (func_name == 'R' || func_name == 'r') {
      F.function_ = &LogRL_dev1;
      FDF.f = &LogRL_dev1;
      FDF.df = &LogRL_dev2;
      FDF.fdf = &LogRL_dev12;
    } else {
      F.function_ = &LogL_dev1;
      FDF.f = &LogL_dev1;
      FDF.df = &LogL_dev2;
      FDF.fdf = &LogL_dev12;
    }

    //writeln("solver");
    gsl_root_fsolver_type *T_f;
    gsl_root_fsolver *s_f;
    T_f = cast(gsl_root_fsolver_type*)gsl_root_fsolver_brent;
    s_f = gsl_root_fsolver_alloc(T_f);

    gsl_root_fdfsolver_type *T_fdf;
    gsl_root_fdfsolver *s_fdf;
    T_fdf = cast(gsl_root_fdfsolver_type*)gsl_root_fdfsolver_newton;
    s_fdf = gsl_root_fdfsolver_alloc(T_fdf);

    for (int i = 0; i < 1; ++i) {

      lambda_l = lambda_lh[i].l;
      lambda_h = lambda_lh[i].h;
      gsl_root_fsolver_set(s_f, &F, lambda_l, lambda_h);

      do {
        iter++;
        status = gsl_root_fsolver_iterate(s_f);
        l = gsl_root_fsolver_root(s_f);
        lambda_l = gsl_root_fsolver_x_lower(s_f);
        lambda_h = gsl_root_fsolver_x_upper(s_f);
        status = gsl_root_test_interval(lambda_l, lambda_h, 0, 1e-1);
      } while (status == GSL_CONTINUE && iter < max_iter);

      iter = 0;

      gsl_root_fdfsolver_set(s_fdf, &FDF, l);

      do {
        iter++;
        status = gsl_root_fdfsolver_iterate(s_fdf);
        l_temp = l;
        l = gsl_root_fdfsolver_root(s_fdf);
        status = gsl_root_test_delta(l, l_temp, 0, 1e-5);
      } while (status == GSL_CONTINUE && iter < max_iter && l > l_min &&
               l < l_max);

      l = l_temp;
      if (l < l_min) {
        l = l_min;
      }
      if (l > l_max) {
        l = l_max;
      }
      if (func_name == 'R' || func_name == 'r') {
        logf_l = LogRL_f(l, params);
      } else {
        //writeln("here LogL_f is invoked!");
        logf_l = LogL_f(l, params);
      }

      if (i == 0) {
        logf = logf_l;
        lambda = l;
      } else if (logf < logf_l) {
        logf = logf_l;
        lambda = l;
      } else {
      }
    }

    gsl_root_fsolver_free(s_f);
    gsl_root_fdfsolver_free(s_fdf);

    if (func_name == 'R' || func_name == 'r') {
      logf_l = LogRL_f(l_min, params);
      logf_h = LogRL_f(l_max, params);
    } else {
      logf_l = LogL_f(l_min, params);
      logf_h = LogL_f(l_max, params);
    }

    if (logf_l > logf) {
      lambda = l_min;
      logf = logf_l;
    }
    if (logf_h > logf) {
      lambda = l_max;
      logf = logf_h;
    }
  }

  return;
}

// Calculate lambda in the null model.
void CalcLambda(char func_name, DMatrix eval,
                DMatrix UtW, DMatrix Uty,
                ref double l_min, ref double l_max, size_t n_region,
                ref  double lambda, ref double logl_H0) {
  writeln("in CalcLambda for null model");
  if (func_name != 'R' && func_name != 'L' && func_name != 'r' &&
      func_name != 'l') {
    writeln("func_name only takes 'R' or 'L': 'R' for
           log-restricted likelihood, 'L' for log-likelihood.");
    return;
  }

  size_t n_cvt = UtW.shape[1], ni_test = UtW.shape[0];
  size_t n_index = (n_cvt + 2 + 1) * (n_cvt + 2) / 2;

  DMatrix Uab = zeros_dmatrix(ni_test, n_index);

  DMatrix ab;
  ab.shape = [1, n_index];

  CalcUab(UtW, Uty, Uab);
  ab.elements = [6.901535246e-295,
  6.901535246e-295,
  4.67120702e-295,
  4.67120702e-295,
  4.671149335e-295,
  1.630416631e-307];

  Calcab(UtW, Uty, ab);

  loglikeparam param0 = loglikeparam(true, ni_test, n_cvt, eval, Uab, ab, 0);

  CalcLambda(func_name, cast(void *)&param0, l_min, l_max, n_region, lambda, logl_H0);

  return;
}

// ni_test is a LMM parameter
void CalcRLWald(size_t ni_test, double l, loglikeparam params, ref double beta,
                     ref double se, ref double p_wald) {
  size_t n_cvt = params.n_cvt;
  size_t n_index = (n_cvt + 2 + 1) * (n_cvt + 2) / 2;

  int df = to!int(ni_test) - to!int(n_cvt) - 1;

  DMatrix Pab;
  Pab.shape = [n_cvt + 2, n_index];
  DMatrix Hi_eval;
  Hi_eval.shape = [1, params.eval.elements.length];
  DMatrix v_temp;
  v_temp.shape = [1, params.eval.elements.length];

  v_temp.elements = params.eval.elements;
  v_temp = multiply_dmatrix_num(v_temp, l);
  if (params.e_mode == 0) {
    Hi_eval = set_ones_dmatrix(Hi_eval);
  } else {
    Hi_eval.elements = v_temp.elements.dup;
  }
  v_temp = add_dmatrix_num(v_temp, 1.0);
  Hi_eval = divide_dmatrix(Hi_eval, v_temp);

  CalcPab(n_cvt, params.e_mode, Hi_eval, params.Uab, params.ab, Pab);

  size_t index_yy = GetabIndex(n_cvt + 2, n_cvt + 2, n_cvt);
  size_t index_xx = GetabIndex(n_cvt + 1, n_cvt + 1, n_cvt);
  size_t index_xy = GetabIndex(n_cvt + 2, n_cvt + 1, n_cvt);
  double P_yy = accessor(Pab, n_cvt, index_yy);
  writeln("P_yy => ", P_yy);
  double P_xx = accessor(Pab, n_cvt, index_xx);
  writeln("P_xx => ", P_xx);
  double P_xy = accessor(Pab, n_cvt, index_xy);
  writeln("P_xy => ", P_xy);
  double Px_yy = accessor(Pab, n_cvt + 1, index_yy);
  writeln("Px_yy => ", Px_yy);

  beta = P_xy / P_xx;
  double tau = to!double(df) / Px_yy;
  se = sqrt(1.0 / (tau * P_xx));
  p_wald = gsl_cdf_fdist_Q((P_yy - Px_yy) * tau, 1.0, df);

  return;
}

void CalcRLScore(size_t ni_test, double l, loglikeparam params, double beta,
                      double se, double p_score) {
  size_t n_cvt = params.n_cvt;
  size_t n_index = (n_cvt + 2 + 1) * (n_cvt + 2) / 2;

  int df = to!int(ni_test) - to!int(n_cvt) - 1;

  DMatrix Pab;
  Pab.shape = [n_cvt + 2, n_index];
  DMatrix Hi_eval;
  Hi_eval.shape = [1, params.eval.elements.length];
  DMatrix v_temp;
  v_temp.shape = [1, params.eval.elements.length];

  v_temp.elements = params.eval.elements;
  v_temp = multiply_dmatrix_num(v_temp, l);

  if (params.e_mode == 0) {
    Hi_eval = set_ones_dmatrix(Hi_eval);
  } else {
    Hi_eval.elements = v_temp.elements.dup;
  }
  v_temp = add_dmatrix_num(v_temp, 1.0);
  Hi_eval = divide_dmatrix(Hi_eval, v_temp);

  CalcPab(n_cvt, params.e_mode, Hi_eval, params.Uab, params.ab, Pab);

  size_t index_yy = GetabIndex(n_cvt + 2, n_cvt + 2, n_cvt);
  size_t index_xx = GetabIndex(n_cvt + 1, n_cvt + 1, n_cvt);
  size_t index_xy = GetabIndex(n_cvt + 2, n_cvt + 1, n_cvt);
  double P_yy = accessor(Pab, n_cvt, index_yy);
  double P_xx = accessor(Pab, n_cvt, index_xx);
  double P_xy = accessor(Pab, n_cvt, index_xy);
  double Px_yy = accessor(Pab, n_cvt + 1, index_yy);

  beta = P_xy / P_xx;
  double tau = to!double(df) / Px_yy;
  se = sqrt(1.0 / (tau * P_xx));

  p_score =
      gsl_cdf_fdist_Q(to!double(ni_test) * P_xy * P_xy / (P_yy * P_xx), 1.0, df);

  return;
}

void CalcUab(DMatrix UtW, DMatrix Uty, ref DMatrix Uab) {
  size_t index_ab;
  size_t n_cvt = UtW.shape[1];

  DMatrix u_a;
  u_a.shape = [1, Uty.shape[1]];

  for (size_t a = 1; a <= n_cvt + 2; ++a) {
    if (a == n_cvt + 1) {
      continue;
    }

    if (a == n_cvt + 2) {
      u_a.elements = Uty.elements.dup;
    } else {
      DMatrix UtW_col = get_col(UtW, a - 1);
      u_a.elements = UtW_col.elements.dup;
    }
    for (size_t b = a; b >= 1; --b) {

      if (b == n_cvt + 1) {
        continue;
      }

      index_ab = GetabIndex(a, b, n_cvt);
      DMatrix Uab_col = get_col(Uab, index_ab);

      if (b == n_cvt + 2) {
        Uab_col.elements = Uty.elements.dup;
      } else {
        DMatrix UtW_col = get_col(UtW, b - 1);
        Uab_col.elements = UtW_col.elements.dup;
      }

      Uab_col = slow_multiply_dmatrix(Uab_col, u_a);
      Uab = set_col(Uab, index_ab, Uab_col);
    }
  }
  return;
}

void CalcUab(DMatrix UtW, DMatrix Uty, DMatrix Utx, ref DMatrix Uab) {
  size_t index_ab;
  size_t n_cvt = UtW.shape[1];

  for (size_t b = 1; b <= n_cvt + 2; ++b) {
    index_ab = GetabIndex(n_cvt + 1, b, n_cvt);
    DMatrix Uab_col = get_col(Uab, index_ab);

    if (b == n_cvt + 2) {
      Uab_col.elements = Uty.elements;
    } else if (b == n_cvt + 1) {
      Uab_col.elements = Utx.elements;
    } else {
      DMatrix UtW_col = get_col(UtW, b - 1);
      Uab_col.elements = UtW_col.elements.dup;
    }

    Uab_col = slow_multiply_dmatrix(Uab_col, Utx);
    Uab = set_col(Uab, index_ab, Uab_col);
  }

  return;
}

void Calcab(DMatrix W, DMatrix y, ref DMatrix ab) {
  size_t index_ab;
  size_t n_cvt = W.shape[1];

  double d;
  DMatrix v_a, v_b;
  v_a.shape = [1, y.shape[1]];
  v_b.shape = [1, y.shape[1]];

  for (size_t a = 1; a <= n_cvt + 2; ++a) {
    if (a == n_cvt + 1) {
      continue;
    }

    if (a == n_cvt + 2) {
      v_a.elements = y.elements.dup;
    } else {
      DMatrix W_col = get_col(W, a - 1);
      v_a.elements = W_col.elements.dup;
    }

    for (size_t b = a; b >= 1; --b) {
      if (b == n_cvt + 1) {
        continue;
      }

      index_ab = GetabIndex(a, b, n_cvt);

      if (b == n_cvt + 2) {
        v_b.elements = y.elements.dup;
      } else {
        DMatrix W_col = get_col(W, b - 1);
        v_b.elements = W_col.elements.dup;
      }

      d = matrix_mult(v_a.T, v_b).elements[0];
      ab.elements[index_ab] = d;
    }
  }

  return;
}

void Calcab(DMatrix W, DMatrix y, DMatrix x, ref DMatrix ab) {
  size_t index_ab;
  size_t n_cvt = W.shape[1];

  double d;
  DMatrix v_b;
  v_b.shape = [1, y.shape[1]];

  for (size_t b = 1; b <= n_cvt + 2; ++b) {
    index_ab = GetabIndex(n_cvt + 1, b, n_cvt);

    if (b == n_cvt + 2) {
      v_b.elements = y.elements.dup;
    } else if (b == n_cvt + 1) {
      v_b.elements = x.elements.dup;
    } else {
      DMatrix W_col = get_col(W, b - 1);
      v_b.elements = W_col.elements.dup;
    }

    d = matrix_mult(x.T, v_b).elements[0];
    ab.elements[index_ab] = d;
  }

  return;
}

// Obtain REML estimate for Vg and Ve using lambda_remle.
// Obtain beta and se(beta) for coefficients.
// ab is not used when e_mode==0.
void CalcLmmVgVeBeta(DMatrix eval, DMatrix UtW,
                     DMatrix Uty, ref double lambda, ref double vg,
                     ref double ve, ref DMatrix beta, ref DMatrix se_beta) {

  writeln("in CalcLmmVgVeBeta");
  size_t n_cvt = UtW.shape[1], ni_test = UtW.shape[0];
  size_t n_index = (n_cvt + 2 + 1) * (n_cvt + 2) / 2;

  DMatrix Uab;
  Uab.shape = [ni_test, n_index];

  DMatrix ab;
  ab.shape =[1, n_index];

  DMatrix Pab;
  Pab.shape = [n_cvt + 2, n_index];

  DMatrix Hi_eval;
  Hi_eval.shape =[1, eval.shape[0]];

  DMatrix v_temp;
  v_temp.shape =[1, eval.shape[0]];

  DMatrix HiW;
  HiW.shape = [eval.shape[1], UtW.shape[1]];

  DMatrix WHiW;
  WHiW.shape = [UtW.shape[1], UtW.shape[1]];

  DMatrix WHiy;
  WHiy.shape =[1, UtW.shape[1]];

  DMatrix Vbeta;
  Vbeta.shape = [UtW.shape[1], UtW.shape[1]];

  Uab = set_zeros_dmatrix(Uab);
  CalcUab(UtW, Uty, Uab);

  v_temp.elements = eval.elements.dup;
  v_temp = multiply_dmatrix_num(v_temp, lambda);
  Hi_eval = set_ones_dmatrix(Hi_eval);
  v_temp = add_dmatrix_num(v_temp, 1.0);
  Hi_eval = divide_dmatrix(Hi_eval, v_temp);

  // Calculate beta.
  HiW.elements = UtW.elements.dup;
  for (size_t i = 0; i < UtW.shape[1]; i++) {
    DMatrix HiW_col = get_col(HiW, i);
    HiW_col = slow_multiply_dmatrix(HiW_col, Hi_eval);
  }
  WHiW = matrix_mult(HiW, UtW);
  WHiy = matrix_mult(HiW, Uty);

  beta.elements = WHiy.elements.dup;

  auto m = to!int(WHiW.cols);
  assert(m>=1);
  auto n = to!int(WHiW.rows);
  int nrhs = n;

  auto ipiv = new int[min(m,n)+1];

  int lda = n;
  int ldb = m;

  enforce(LAPACKE_dgesv( 101, n, n, WHiW.elements.ptr, lda, ipiv.ptr,
                      beta.elements.ptr,  ldb ) == 0);
  //exit(0);

  Vbeta = inverse(WHiy);

  ab.elements = [6.901535246e-295,
  6.901535246e-295,
  4.67120702e-295,
  4.67120702e-295,
  4.671149335e-295,
  1.630416631e-307];
  Calcab(UtW, Uty, ab);
  CalcPab(n_cvt, 0, Hi_eval, Uab, ab, Pab);

  size_t index_yy = GetabIndex(n_cvt + 2, n_cvt + 2, n_cvt);
  double P_yy = accessor(Pab, n_cvt, index_yy);

  ve = P_yy / to!double(ni_test - n_cvt);
  vg = ve * lambda;

  // With ve, calculate se(beta).
  Vbeta = multiply_dmatrix_num(Vbeta, ve);

  // Obtain se_beta.
  for (size_t i = 0; i < Vbeta.shape[1]; i++) {
    se_beta.elements[i] = sqrt(accessor(Vbeta, i, i));
  }

  //gsl_permutation_free(pmt);
  writeln("out of CalcLmmVgVeBeta");


  return;
}

// Obtain REMLE estimate for PVE using lambda_remle.
void CalcPve(DMatrix eval, DMatrix UtW,
             DMatrix Uty, ref double lambda, ref double trace_G,
             ref double pve, ref double pve_se) {
  writeln("in CalcPve");

  size_t n_cvt = UtW.shape[1], ni_test = UtW.shape[0];
  size_t n_index = (n_cvt + 2 + 1) * (n_cvt + 2) / 2;

  DMatrix Uab;
  Uab.shape = [ni_test, n_index];
  DMatrix ab;
  ab.shape = [1, n_index];

  Uab = set_zeros_dmatrix(Uab);
  CalcUab(UtW, Uty, Uab);

  loglikeparam param0 = loglikeparam(true, ni_test, n_cvt, eval, Uab, ab, 0);
  //write constructor

  double se = sqrt(-1.0 / LogRL_dev2(lambda, &param0));

  pve = trace_G * lambda / (trace_G * lambda + 1.0);
  pve_se = trace_G / ((trace_G * lambda + 1.0) * (trace_G * lambda + 1.0)) * se;
  writeln("out of CalcPve");

  return;
}


struct GWAS_SNPs{
  bool size;
}

void AnalyzeBimbam (Param cPar, DMatrix U, DMatrix eval, DMatrix UtW, DMatrix Uty,
                        DMatrix W, DMatrix y, GWAS_SNPs gwasnps,
                        size_t n_cvt, size_t LMM_BATCH_SIZE = 100) {

  writeln("indicator_idv");
  DMatrix indicator_idv = read_matrix_from_file2("/home/prasun/dev/faster_lmm_d/data/gemma/indicator_idv.txt");
  writeln("indicator_snp");
  DMatrix indicator_snp = read_matrix_from_file2("/home/prasun/dev/faster_lmm_d/data/gemma/indicator_snp.txt");

  writeln(indicator_idv.shape);
  writeln(indicator_snp.shape);

  string filename = "/home/prasun/dev/faster_lmm_d/data/gemma/mouse_hs1940.geno.txt.gz";
  auto pipe = pipeShell("gunzip -c " ~ filename);
  File input = pipe.stdout;

  n_cvt = 1;

  SUMSTAT[] sumStat;

  double lambda_mle=0, lambda_remle=0, beta=0, se=0, p_wald=0;
  double p_lrt=0, p_score=0;
  int n_miss, c_phen;
  double geno, x_mean;

  double logl_H1=0.0;
  size_t ni_test = UtW.shape[0];
  size_t ni_total = 1940;
  size_t n_region = cPar.n_region;
  int a_mode = 1;
  double l_min = cPar.l_min;
  double l_mle_null = cPar.l_mle_null;
  double l_max = cPar.l_max;
  double logl_mle_H0 = cPar.logl_mle_H0;

  size_t n_index=(n_cvt+2+1)*(n_cvt+2)/2;

  writeln("ni_test =======> ", ni_test);
  writeln("ni_total =======> ", ni_total);
  writeln("n_region =======> ", n_region);
  writeln("l_mle_null =======> ", l_mle_null);
  writeln("l_remle_null =======> ", cPar.l_remle_null);
  writeln("l_max =======> ", l_max);
  writeln("l_min =======> ", l_min);
  writeln("logl_mle_H0 =======> ", logl_mle_H0);
  writeln("n_index =======> ", n_index);

  DMatrix x;
  x.shape = [U.shape[0],1];
  x = set_zeros_dmatrix(x);
  DMatrix x_miss;
  x_miss.shape = [1, U.shape[0]];
  DMatrix Utx;
  Utx.shape = [1, U.shape[1]];
  Utx = set_zeros_dmatrix(Utx);
  DMatrix ab;
  ab.shape = [1, n_index];

  // Create a large matrix.
  size_t msize=10000;
  DMatrix Xlarge;
  Xlarge.shape = [U.shape[0], msize];
  DMatrix UtXlarge;
  UtXlarge.shape = [U.shape[0], msize];
  Xlarge = set_zeros_dmatrix(Xlarge);
  UtXlarge = set_zeros_dmatrix(UtXlarge);


  //start reading genotypes and analyze
  size_t c=0, t_last=0;
  for (size_t t=0; t<indicator_snp.elements.length; ++t) {
    if (indicator_snp.elements[t]==0) {continue;}
    t_last++;
  }
  //writeln(indicator_snp);

  int t = 0;

  //Bar b = new Bar();
  //b.message = {return "Processing";};
  //b.max = indicator_snp.elements.length;
  
  DMatrix Uab;
  Uab.shape = [U.shape[1], n_index];

  Uab = set_zeros_dmatrix(Uab);
  CalcUab (UtW, Uty, Uab);

  writeln(Uab);
  DMatrix abc = DMatrix(Uab.shape.dup, Uab.elements.dup);

  //exit(0);
  foreach (line ; input.byLine) {
   
    if (indicator_snp.elements[t]==0) {
      t++;
      continue;
    }
//
    auto chr = to!string(line).split(",")[3..$];

    x_mean=0.0; c_phen=0; n_miss=0;
    x_miss = set_zeros_dmatrix(x_miss);
    for (size_t i=0; i<ni_total; ++i) {
      auto ch_ptr = to!string(chr[i].strip());
      if (indicator_idv.elements[i]==0) {continue;}

      if (ch_ptr == "NA") {
        x_miss.elements[c_phen] = 0.0;
        n_miss++;
      }
      else {
        geno=to!double(ch_ptr);

        x.elements[c_phen] = geno;
        x_miss.elements[c_phen] = 1.0;
        x_mean += geno;
      }
      c_phen++;
    }

    x_mean/= to!double(ni_test-n_miss);

    for (size_t i=0; i<ni_test; ++i) {
      if ( x_miss.elements[i] == 0) {
        x.elements[i] = x_mean;
      }
    }

    set_col2(Xlarge, c%msize, x);

    c++;

    if (c % msize==0 || c==t_last) {
      size_t l=0;
      if (c%msize==0) {l=msize;} else {l=c%msize;}

      DMatrix Xlarge_sub = get_sub_dmatrix(Xlarge, 0, 0, Xlarge.shape[0], l);
      //DMatrix UtXlarge_sub = get_sub_dmatrix(UtXlarge, 0, 0, UtXlarge.shape[0], l);

      DMatrix UtXlarge_sub = matrix_mult(U.T, Xlarge_sub);

      set_sub_dmatrix(UtXlarge, 0, 0, UtXlarge.shape[0], l, UtXlarge_sub);
      assert(UtXlarge_sub.shape == [U.shape[0], msize]);

      Xlarge = set_zeros_dmatrix(Xlarge);
      for (size_t i=0; i<l; i++) {

        DMatrix UtXlarge_col= get_col(UtXlarge, i);           //view
        Utx.elements = UtXlarge_col.elements.dup;

        CalcUab(UtW, Uty, Utx, Uab);


        //writeln(Uab);
        //writeln(Utx);

        //exit(0);

        writeln(Uab.shape);

        ab = set_zeros_dmatrix(ab);

        loglikeparam param1 = loglikeparam(false, ni_test, n_cvt, eval, Uab, ab, 0);

        // 3 is before 1.
        if (a_mode==3 || a_mode==4) {

          CalcRLScore (ni_test, l_mle_null, param1, beta, se, p_score);
        }

        if (a_mode==1 || a_mode==4) {
          CalcLambda ('R', cast(void *)&param1, l_min, l_max, n_region, lambda_remle, logl_H1);
          writeln("logl_H1  => ", logl_H1);
          writeln("lambda_remle => ", lambda_remle);
          CalcRLWald (ni_test, lambda_remle, param1, beta, se, p_wald);
          writeln("p_wald => ", p_wald);
        }

        if (a_mode==2 || a_mode==4) {
          CalcLambda ('L', cast(void *)&param1, l_min, l_max, n_region, lambda_mle, logl_H1);
          p_lrt=gsl_cdf_chisq_Q (2.0*(logl_H1 - logl_mle_H0), 1);
        }

        SUMSTAT SNPs = SUMSTAT(beta, se, lambda_remle, lambda_mle, p_wald, p_lrt, p_score);

        sumStat ~= SNPs;
        writeln(i);

      }
    }
    t++;
    //b.next();
  }
  //b.finish();
  writeln(sumStat);
  return;
}

unittest{
  size_t n_cvt = 2;
  size_t e_mode = 0;
  DMatrix Hi_eval = DMatrix([2,2],[1,2,3,4]);
  DMatrix Uab = DMatrix([2,2],[1,2,3,4]);
  DMatrix ab = DMatrix([2,2],[1,2,3,4]);
  DMatrix Pab;


  size_t index = GetabIndex(4, 9, n_cvt);
  assert(index == 0);

  index = GetabIndex(4, 9, 16);
  assert(index == 56);

  //CalcPab(n_cvt , e_mode,  Hi_eval, Uab,  ab, Pab);
  //writeln(Pab);
  //assert(Pab == 0);

  DMatrix PPab;
  DMatrix HiHi_eval;
  //CalcPPab(n_cvt, e_mode, HiHi_eval,  Uab, ab, Pab, PPab);
  ////assert();

  DMatrix HiHiHi_eval;
  DMatrix PPPab;
  //CalcPPPab( n_cvt, e_mode, HiHiHi_eval, Uab, ab, Pab, PPab, PPPab);
  ////assert();




  double lambda = 0.7;
  double logf;
  loglikeparam params;
  //CalcLambda(func_name, cast(void *)&params, l_min, l_max, n_region, lambda, logf);
  //assert();

  // Calculate lambda in the null model.

  DMatrix G = DMatrix([5, 5],[ 212,   7, 11, 12, 30,
                              11,  101, 34,  1, -7,
                              151,-101, 96,  1, 73,
                              87,  102, 64, 19, 67,
                              -21,  10, 334, 22, -2
                             ]);
  DMatrix W = ones_dmatrix(5,5);
  DMatrix y = DMatrix([5,1], [3, 14 ,-5, 18, 6]);
  auto kvakve = eigh(G);
  DMatrix U = kvakve.kva;
  DMatrix eval = kvakve.kve;
  DMatrix UtW = matrix_mult(U.T, W);
  DMatrix Uty = matrix_mult(U.T, y);
  double  logl_H0;

  Param cPar;

  char func_name = 'R';
  double l_min = 0.00001;
  double l_max = 10;
  size_t n_region = 10;

  ab = zeros_dmatrix(1,5);


  n_cvt = UtW.shape[1];
  size_t ni_test = UtW.shape[0];
  size_t n_index = (n_cvt + 2 + 1) * (n_cvt + 2) / 2;

  Uab = zeros_dmatrix(ni_test, n_index);

  loglikeparam param0 = loglikeparam(true, ni_test, n_cvt, eval, Uab, ab, 0);
  //CalcLambda(func_name, eval, UtW, Uty, l_min, l_max, n_region, lambda, logl_H0);
  //assert();

  double l = 6;
  double beta;
  double se;
  double p_wald;
  //CalcRLWald(ni_test, l, params, beta, se, p_wald);
  //assert();

  double p_score;
  //CalcRLScore(ni_test, l, params, beta, se, p_score);
  //assert();

  //CalcUab(UtW, Uty, Uab);
  //assert();

  //CalcUab(UtW, Uty, Utx, Uab);
  //assert();

  //Calcab(W, y, ab);
  //assert();

  DMatrix x;
  //Calcab(W, y, x, ab);
  //assert();

  double vg, ve;
  DMatrix se_beta;
  //CalcLmmVgVeBeta(eval, UtW, Uty, lambda, vg, ve, beta, se_beta);

  double trace_G, pve, pve_se;
  //CalcPve(eval,  UtW, Uty, lambda, trace_G, pve,  pve_se);
}

/*
   This code is part of faster_lmm_d and published under the GPLv3
   License (see LICENSE.txt)

   Copyright © 2017 - 2018 Prasun Anand & Pjotr Prins
*/

module faster_lmm_d.gemma_lmm;

import core.stdc.stdlib : exit;

import std.conv;
import std.exception;
import std.file;
import std.math;
import std.parallelism;
import std.algorithm: min, max, reduce;
alias mlog = std.math.log;
import std.process;
import std.range;
import std.stdio;
import std.typecons;
import std.experimental.logger;
import std.string;

import faster_lmm_d.dmatrix;
import faster_lmm_d.gemma_param;
import faster_lmm_d.helpers;
import faster_lmm_d.optmatrix;

import gsl.permutation;
import gsl.cdf;
import gsl.errno;
import gsl.math;
import gsl.min;
import gsl.roots;

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

DMatrix CalcPab(const size_t n_cvt, const size_t e_mode, const DMatrix Hi_eval,
              const DMatrix Uab, const DMatrix ab, const size_t[] shape) {
  size_t index_ab, index_aw, index_bw, index_ww;
  double p_ab, ps_ab, ps_aw, ps_bw, ps_ww;
  DMatrix Pab = zeros_dmatrix(shape[0], shape[1]);
  for (size_t p = 0; p <= n_cvt + 1; ++p) {
    for (size_t a = p + 1; a <= n_cvt + 2; ++a) {
      for (size_t b = a; b <= n_cvt + 2; ++b) {
        index_ab = GetabIndex(a, b, n_cvt);
        if (p == 0) {
          DMatrix Uab_row = get_row(Uab, index_ab);
          p_ab = vector_ddot(Hi_eval, Uab_row);

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
  return Pab;
}

DMatrix calc_Pab_batched(const size_t n_cvt, const DMatrix Hi_eval, const DMatrix Uab, const DMatrix ab,
                          const size_t[] shape, const double[] l) {
  size_t index_ab, index_aw, index_bw, index_ww;
  double p_ab, ps_ab, ps_aw, ps_bw, ps_ww;
  DMatrix Pab = zeros_dmatrix(Hi_eval.shape[0]*shape[0], shape[1]);

  DMatrix p_ab2 = cpu_mat_mult(Hi_eval, 0, Uab, 1);
  const size_t col_counter = Uab.shape[0]/Hi_eval.shape[0];
  size_t row_counter = shape[0];

  foreach(i; 0..Hi_eval.shape[0]){
    for (size_t p = 0; p <= n_cvt + 1; ++p) {
      for (size_t a = p + 1; a <= n_cvt + 2; ++a) {
        for (size_t b = a; b <= n_cvt + 2; ++b) {
          index_ab = GetabIndex(a, b, n_cvt);
          if (p == 0) {
            p_ab = p_ab2.accessor(i, i * col_counter + index_ab );
            Pab.elements[Pab.cols * (i * row_counter) + index_ab] = p_ab;
          } else {
            size_t row_no = i * row_counter + p - 1;
            index_aw = GetabIndex(a, p, n_cvt);
            index_bw = GetabIndex(b, p, n_cvt);
            index_ww = GetabIndex(p, p, n_cvt);

            ps_ab = accessor(Pab, row_no, index_ab);
            ps_aw = accessor(Pab, row_no, index_aw);
            ps_bw = accessor(Pab, row_no, index_bw);
            ps_ww = accessor(Pab, row_no, index_ww);

            p_ab = ps_ab - ps_aw * ps_bw / ps_ww;
            Pab.elements[(row_no + 1) * Pab.cols + index_ab] = p_ab;
          }
        }
      }
    }
  }
  return Pab;
}

DMatrix CalcPPab(const size_t n_cvt, const size_t e_mode,
              const DMatrix HiHi_eval, const DMatrix Uab,
              const DMatrix ab, const DMatrix Pab, const size_t[] shape) {
  size_t index_ab, index_aw, index_bw, index_ww;
  double p2_ab,  ps2_ab, ps_aw, ps_bw, ps_ww, ps2_aw, ps2_bw, ps2_ww;

  DMatrix PPab = zeros_dmatrix(shape[0], shape[1]);
  for (size_t p = 0; p <= n_cvt + 1; ++p) {
    for (size_t a = p + 1; a <= n_cvt + 2; ++a) {
      for (size_t b = a; b <= n_cvt + 2; ++b) {
        index_ab = GetabIndex(a, b, n_cvt);
        if (p == 0) {
          DMatrix Uab_row = get_row(Uab, index_ab);
          p2_ab = vector_ddot(HiHi_eval, Uab_row);  // check its shape is [1,1] else take transpose of HiHi_eval
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
  return PPab;
}

DMatrix CalcPPPab(const size_t n_cvt, const size_t e_mode,
               const DMatrix HiHiHi_eval, const DMatrix Uab,
               const DMatrix ab, const DMatrix Pab,
               const DMatrix PPab, const size_t[] shape) {
  size_t index_ab, index_aw, index_bw, index_ww;
  double p3_ab;
  double ps3_ab, ps_aw, ps_bw, ps_ww, ps2_aw, ps2_bw, ps2_ww, ps3_aw, ps3_bw, ps3_ww;
  DMatrix PPPab = zeros_dmatrix(shape[0], shape[1]);

  for (size_t p = 0; p <= n_cvt + 1; ++p) {
    for (size_t a = p + 1; a <= n_cvt + 2; ++a) {
      for (size_t b = a; b <= n_cvt + 2; ++b) {
        index_ab = GetabIndex(a, b, n_cvt);
        if (p == 0) {
          DMatrix Uab_row = get_row(Uab, index_ab);
          p3_ab = vector_ddot(HiHiHi_eval, Uab_row);
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
  return PPPab;
}

size_t GetabIndex( const size_t a, const size_t b, const size_t n_cvt)
in {
  size_t n = n_cvt + 2;
  assert( a <= n || b <= n || a > 0 || b > 0 , "error in GetabIndex.");
}
body {
  size_t n = n_cvt + 2;
  return ( b < a ?  ((2 * n - b + 2) * (b - 1) / 2 + a - b ): ((2 * n - a + 2) * (a - 1) / 2 + b - a) );
}

struct loglikeparam{
  const size_t n_cvt;
  const size_t ni_test;
  const size_t n_index;
  const bool calc_null;
  const int e_mode;
  const DMatrix eval;
  const DMatrix Uab;
  const DMatrix ab;

  this(const bool calc_null, const size_t ni_test, const size_t n_cvt, const DMatrix eval, const DMatrix Uab,  const DMatrix ab, const int e_mode) {
    this.n_cvt = n_cvt;
    this.ni_test = ni_test;
    this.calc_null = calc_null;
    this.e_mode = e_mode;
    this.eval = DMatrix(eval);
    this.Uab = DMatrix(Uab);
    this.ab = ab;
  }
}


double LogL_f(const double l, const void* params) {

  auto ptr = cast(loglikeparam *)params;
  loglikeparam p = *ptr;

  size_t n_cvt = p.n_cvt;
  size_t ni_test = p.ni_test;
  size_t n_index = (n_cvt + 2 + 1) * (n_cvt + 2) / 2;

  size_t nc_total = (p.calc_null == true ? n_cvt : n_cvt + 1);

  double f = 0.0;
  double logdet_h = 0.0;
  double d;
  size_t index_yy;

  DMatrix v_temp = multiply_dmatrix_num(p.eval.T, l);
  DMatrix Hi_eval = (p.e_mode == 0 ? divide_num_dmatrix(1, add_dmatrix_num(v_temp, 1.0)) : dup_dmatrix(v_temp));

  foreach (element; v_temp.elements)
    logdet_h += mlog(fabs(element) + 1);

  DMatrix Pab = CalcPab(n_cvt, p.e_mode, Hi_eval, p.Uab, p.ab, [n_cvt + 2, n_index]);

  double c = 0.5 * to!double(ni_test) * (mlog(to!double(ni_test)) - mlog(2 * M_PI) - 1.0);

  index_yy = GetabIndex(n_cvt + 2, n_cvt + 2, n_cvt);
  double P_yy = accessor(Pab, nc_total, index_yy);
  f = c - 0.5 * logdet_h - 0.5 * to!double(ni_test) * mlog(P_yy);

  return f;
}

double LogRL_f(const double l, const void* params) {

  auto ptr = cast(loglikeparam *)params;
  loglikeparam p = *ptr;

  size_t n_cvt = p.n_cvt;
  size_t ni_test = p.ni_test;
  size_t n_index = (n_cvt + 2 + 1) * (n_cvt + 2) / 2;

  size_t nc_total = (p.calc_null == true ? n_cvt : n_cvt + 1);
  double df = (p.calc_null == true ? to!double(ni_test) - to!double(n_cvt) : to!double(ni_test) - to!double(n_cvt) - 1.0);

  double f = 0.0, logdet_h = 0.0, logdet_hiw = 0.0;
  size_t index_ww;

  DMatrix v_temp = multiply_dmatrix_num(p.eval.T, l);
  DMatrix Hi_eval = (p.e_mode == 0 ? divide_num_dmatrix(1, add_dmatrix_num(v_temp, 1.0)) : dup_dmatrix(v_temp));

  foreach (element; v_temp.elements)
    logdet_h += mlog(fabs(element) + 1);

  DMatrix Pab = CalcPab(n_cvt, p.e_mode, Hi_eval, p.Uab, p.ab, [n_cvt + 2, n_index]);

  v_temp = set_ones_dmatrix(v_temp);
  DMatrix Iab = CalcPab(n_cvt, p.e_mode, v_temp, p.Uab, p.ab, [n_cvt + 2, n_index]);

  // Calculate |WHiW|-|WW|.
  logdet_hiw = 0.0;
  foreach (i; 0..nc_total) {
    index_ww = GetabIndex(i + 1, i + 1, n_cvt);
    logdet_hiw += mlog(accessor(Pab, i, index_ww)) - mlog(accessor(Iab, i, index_ww));
  }

  index_ww = GetabIndex(n_cvt + 2, n_cvt + 2, n_cvt);
  double P_yy = accessor(Pab, nc_total, index_ww);

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

  size_t nc_total = (p.calc_null == true ? n_cvt : n_cvt + 1);
  double df = (p.calc_null == true ? to!double(ni_test) - to!double(n_cvt) : to!double(ni_test) - to!double(n_cvt) - 1.0);

  double dev1 = 0.0, trace_Hi = 0.0;
  size_t index_ww;

  DMatrix v_temp = multiply_dmatrix_num(p.eval.T, l);
  DMatrix Hi_eval = (p.e_mode == 0 ? divide_num_dmatrix(1, add_dmatrix_num(v_temp, 1.0)) : dup_dmatrix(v_temp));
  DMatrix HiHi_eval = slow_multiply_dmatrix(Hi_eval, Hi_eval);

  v_temp = set_ones_dmatrix(v_temp);
  trace_Hi = vector_ddot(Hi_eval, v_temp);

  if (p.e_mode != 0) {
    trace_Hi = to!double(ni_test) - trace_Hi;
  }

  DMatrix Pab = CalcPab(n_cvt, p.e_mode, Hi_eval, p.Uab, p.ab, [n_cvt + 2, n_index]);
  DMatrix PPab = CalcPPab(n_cvt, p.e_mode, HiHi_eval, p.Uab, p.ab, Pab, [n_cvt + 2, n_index]);

  // Calculate tracePK and trace PKPK.
  double trace_P = trace_Hi;
  double ps_ww, ps2_ww;
  foreach (i; 0..nc_total) {
    index_ww = GetabIndex(i + 1, i + 1, n_cvt);
    ps_ww = accessor(Pab, i, index_ww);
    ps2_ww = accessor(PPab, i, index_ww);
    trace_P -= ps2_ww / ps_ww;
  }
  double trace_PK = (df - trace_P) / l;

  // Calculate yPKPy, yPKPKPy.
  index_ww = GetabIndex(n_cvt + 2, n_cvt + 2, n_cvt);
  double P_yy = accessor(Pab, nc_total, index_ww);
  double PP_yy = accessor(PPab, nc_total, index_ww);
  double yPKPy = (P_yy - PP_yy) / l;

  dev1 = -0.5 * trace_PK + 0.5 * df * yPKPy / P_yy;

  return dev1;

}

extern(C) double LogL_dev1(double l, void* params) {
  auto ptr = cast(loglikeparam *)params;
  loglikeparam p = *ptr;

  size_t n_cvt = p.n_cvt;
  size_t ni_test = p.ni_test;
  size_t n_index = (n_cvt + 2 + 1) * (n_cvt + 2) / 2;

  size_t nc_total = (p.calc_null == true ? n_cvt : n_cvt + 1);

  double dev1 = 0.0, trace_Hi = 0.0;
  size_t index_yy;

  DMatrix v_temp = multiply_dmatrix_num(p.eval.T, l);
  DMatrix Hi_eval = (p.e_mode == 0 ? divide_num_dmatrix(1, add_dmatrix_num(v_temp, 1.0)) : dup_dmatrix(v_temp));
  DMatrix HiHi_eval = slow_multiply_dmatrix(Hi_eval, Hi_eval);

  v_temp = set_ones_dmatrix(v_temp);
  trace_Hi = vector_ddot(Hi_eval, v_temp);

  if (p.e_mode != 0) {
    trace_Hi = to!double(ni_test) - trace_Hi;
  }

  DMatrix Pab = CalcPab(n_cvt, p.e_mode, Hi_eval, p.Uab, p.ab, [n_cvt + 2, n_index]);
  DMatrix PPab = CalcPPab(n_cvt, p.e_mode, HiHi_eval, p.Uab, p.ab, Pab, [n_cvt + 2, n_index]);

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

  size_t nc_total = (p.calc_null == true ? n_cvt : n_cvt + 1);
  double df = (p.calc_null == true ? to!double(ni_test) - to!double(n_cvt) : to!double(ni_test) - to!double(n_cvt) - 1.0);

  double dev2 = 0.0, trace_Hi = 0.0, trace_HiHi = 0.0;
  size_t index_ww;

  DMatrix v_temp = multiply_dmatrix_num(p.eval.T, l);
  DMatrix Hi_eval = (p.e_mode == 0 ? divide_num_dmatrix(1, add_dmatrix_num(v_temp, 1.0)) : dup_dmatrix(v_temp));
  DMatrix HiHi_eval = slow_multiply_dmatrix(Hi_eval, Hi_eval);
  DMatrix HiHiHi_eval = slow_multiply_dmatrix(HiHi_eval, Hi_eval);

  v_temp = set_ones_dmatrix(v_temp);
  trace_Hi = vector_ddot(Hi_eval, v_temp);
  trace_HiHi = vector_ddot(HiHi_eval, v_temp);

  if (p.e_mode != 0) {
    trace_Hi = to!double(ni_test) - trace_Hi;
    trace_HiHi = 2 * trace_Hi + trace_HiHi - to!double(ni_test);
  }

  DMatrix Pab = CalcPab(n_cvt, p.e_mode, Hi_eval, p.Uab, p.ab, [n_cvt + 2, n_index]);
  DMatrix PPab = CalcPPab(n_cvt, p.e_mode, HiHi_eval, p.Uab, p.ab, Pab, [n_cvt + 2, n_index]);
  DMatrix PPPab = CalcPPPab(n_cvt, p.e_mode, HiHiHi_eval, p.Uab, p.ab, Pab, PPab, [n_cvt + 2, n_index]);

  // Calculate tracePK and trace PKPK.
  double trace_P = trace_Hi, trace_PP = trace_HiHi;
  double ps_ww, ps2_ww, ps3_ww;
  foreach(i; 0..nc_total) {
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

  size_t nc_total = (p.calc_null == true ? n_cvt : n_cvt + 1);

  double dev2 = 0.0, trace_Hi = 0.0, trace_HiHi = 0.0;
  size_t index_yy;

  DMatrix v_temp = multiply_dmatrix_num(p.eval.T, l);
  DMatrix Hi_eval = (p.e_mode == 0 ? divide_num_dmatrix(1, add_dmatrix_num(v_temp, 1.0)) : dup_dmatrix(v_temp));
  DMatrix HiHi_eval = slow_multiply_dmatrix(Hi_eval, Hi_eval);
  DMatrix HiHiHi_eval = slow_multiply_dmatrix(HiHi_eval, Hi_eval);

  v_temp = set_ones_dmatrix(v_temp);

  trace_Hi = vector_ddot(Hi_eval, v_temp);

  trace_HiHi = vector_ddot(HiHi_eval, v_temp);
  if (p.e_mode != 0) {
    trace_Hi = to!double(ni_test) - trace_Hi;
    trace_HiHi = 2 * trace_Hi + trace_HiHi - to!double(ni_test);
  }

  DMatrix Pab = CalcPab(n_cvt, p.e_mode, Hi_eval, p.Uab, p.ab, [n_cvt + 2, n_index]);
  DMatrix PPab = CalcPPab(n_cvt, p.e_mode, HiHi_eval, p.Uab, p.ab, Pab, [n_cvt + 2, n_index]);
  DMatrix PPPab = CalcPPPab(n_cvt, p.e_mode, HiHiHi_eval, p.Uab, p.ab, Pab, PPab, [n_cvt + 2, n_index]);

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


  auto ptr = cast(loglikeparam *)params;
  loglikeparam p = *ptr;

  size_t n_cvt = p.n_cvt;
  size_t ni_test = p.ni_test;
  size_t n_index = (n_cvt + 2 + 1) * (n_cvt + 2) / 2;

  size_t nc_total = (p.calc_null == true ? n_cvt : n_cvt + 1);

  double trace_Hi = 0.0, trace_HiHi = 0.0;
  size_t index_yy;


  DMatrix v_temp = multiply_dmatrix_num(p.eval.T, l);
  DMatrix Hi_eval = (p.e_mode == 0 ? divide_num_dmatrix(1, add_dmatrix_num(v_temp, 1.0)) : dup_dmatrix(v_temp));
  DMatrix HiHi_eval = slow_multiply_dmatrix(Hi_eval, Hi_eval);
  DMatrix HiHiHi_eval = slow_multiply_dmatrix(HiHi_eval, Hi_eval);

  v_temp = set_ones_dmatrix(v_temp);
  trace_Hi = vector_ddot(Hi_eval, v_temp);
  trace_HiHi = vector_ddot(HiHi_eval, v_temp);

  if (p.e_mode != 0) {
    trace_Hi = to!double(ni_test) - trace_Hi;
    trace_HiHi = 2 * trace_Hi + trace_HiHi - to!double(ni_test);
  }

  DMatrix Pab = CalcPab(n_cvt, p.e_mode, Hi_eval, p.Uab, p.ab, [n_cvt + 2, n_index]);
  DMatrix PPab = CalcPPab(n_cvt, p.e_mode, HiHi_eval, p.Uab, p.ab, Pab, [n_cvt + 2, n_index]);
  DMatrix PPPab = CalcPPPab(n_cvt, p.e_mode, HiHiHi_eval, p.Uab, p.ab, Pab, PPab, [n_cvt + 2, n_index]);

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

  size_t nc_total = (p.calc_null == true ? n_cvt : n_cvt + 1);
  double df = (p.calc_null == true ? to!double(ni_test) - to!double(n_cvt) : to!double(ni_test) - to!double(n_cvt) - 1.0);

  double trace_Hi = 0.0, trace_HiHi = 0.0;
  size_t index_ww;

  DMatrix v_temp = multiply_dmatrix_num(p.eval.T, l);
  DMatrix Hi_eval = (p.e_mode == 0 ? divide_num_dmatrix(1, add_dmatrix_num(v_temp, 1.0)) : dup_dmatrix(v_temp));
  DMatrix HiHi_eval = slow_multiply_dmatrix(Hi_eval, Hi_eval);
  DMatrix HiHiHi_eval = slow_multiply_dmatrix(HiHi_eval, Hi_eval);

  v_temp = set_ones_dmatrix(v_temp);
  trace_Hi = vector_ddot(Hi_eval, v_temp);
  trace_HiHi = vector_ddot(HiHi_eval, v_temp);

  if (p.e_mode != 0) {
    trace_Hi = to!double(ni_test) - trace_Hi;
    trace_HiHi = 2 * trace_Hi + trace_HiHi - to!double(ni_test);
  }

  DMatrix Pab = CalcPab(n_cvt, p.e_mode, Hi_eval, p.Uab, p.ab, [n_cvt + 2, n_index]);
  DMatrix PPab = CalcPPab(n_cvt, p.e_mode, HiHi_eval, p.Uab, p.ab, Pab, [n_cvt + 2, n_index]);
  DMatrix PPPab = CalcPPPab(n_cvt, p.e_mode, HiHiHi_eval, p.Uab, p.ab, Pab, PPab, [n_cvt + 2, n_index]);

  // Calculate tracePK and trace PKPK.
  double trace_P = trace_Hi, trace_PP = trace_HiHi;

  double ps_ww, ps2_ww, ps3_ww;
  foreach(i; 0..nc_total) {
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

alias Tuple!(double, "l", double, "h") Lambda_tup;
alias Tuple!(double, "lambda", double, "logf") Lambda_result;

Lambda_result calc_lambda(const char func_name, void* params, const double l_min,
                const double l_max, const size_t n_region) {
  //writeln("in calc_lambda for NOT-NULL");
  double lambda, logf;

  if (func_name != 'R' && func_name != 'L' && func_name != 'r' &&
      func_name != 'l') {
    writeln("func_name only takes 'R' or 'L': 'R' for
            log-restricted likelihood, 'L' for log-likelihood.");
    exit(0);
  }

  Lambda_tup[] lambda_lh;

  // Evaluate first-order derivates in different intervals.
  double lambda_l, lambda_h;
  double lambda_interval = mlog(l_max / l_min) / to!double(n_region);
  double dev1_l, dev1_h, logf_l, logf_h;
  //writeln("lambda_interval = ", lambda_interval);
  foreach(i; 0..n_region) {
    lambda_l = l_min * exp(lambda_interval * i);
    lambda_h = l_min * exp(lambda_interval * (i + 1.0));

    if (func_name == 'R' || func_name == 'r') {
      dev1_l = LogRL_dev1(lambda_l, params);
      dev1_h = LogRL_dev1(lambda_h, params);
    } else {
      dev1_l = LogL_dev1(lambda_l, params);
      dev1_h = LogL_dev1(lambda_h, params);
    }
    if (dev1_l * dev1_h <= 0) {
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

    foreach(i; 0..1) {

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

  return Lambda_result(lambda, logf);
}

// Calculate lambda in the null model.
Lambda_result calc_lambda(const char func_name, const DMatrix eval, const DMatrix UtW, const DMatrix Uty,
                  const double l_min, const double l_max, const size_t n_region) {
  writeln("in calc_lambda for null model");
  if (func_name != 'R' && func_name != 'L' && func_name != 'r' &&
      func_name != 'l') {
    writeln("func_name only takes 'R' or 'L': 'R' for
           log-restricted likelihood, 'L' for log-likelihood.");
    exit(0);
  }

  size_t n_cvt = UtW.shape[1], ni_test = UtW.shape[0];
  size_t n_index = (n_cvt + 2 + 1) * (n_cvt + 2) / 2;
  DMatrix Uab = calc_Uab(UtW, Uty, ni_test, n_index);
  DMatrix ab = calc_ab(UtW, Uty, [1, n_index]);
  loglikeparam param0 = loglikeparam(true, ni_test, n_cvt, eval, Uab.T, ab, 0);
  return calc_lambda(func_name, cast(void *)&param0, l_min, l_max, n_region);
}

alias Tuple!(double, "beta", double, "se", double, "p_wald") Wald_score;

Wald_score calc_RL_Wald(const size_t ni_test, const double l, loglikeparam params) {
  size_t n_cvt = params.n_cvt;
  size_t n_index = (n_cvt + 2 + 1) * (n_cvt + 2) / 2;

  int df = to!int(ni_test) - to!int(n_cvt) - 1;

  DMatrix v_temp = multiply_dmatrix_num(params.eval.T, l);
  DMatrix Hi_eval = (params.e_mode == 0 ? divide_num_dmatrix(1, add_dmatrix_num(v_temp, 1.0)) : dup_dmatrix(v_temp));

  DMatrix Pab = CalcPab(n_cvt, params.e_mode, Hi_eval, params.Uab, params.ab, [n_cvt + 2, n_index]);

  size_t index_yy = GetabIndex(n_cvt + 2, n_cvt + 2, n_cvt);
  size_t index_xx = GetabIndex(n_cvt + 1, n_cvt + 1, n_cvt);
  size_t index_xy = GetabIndex(n_cvt + 2, n_cvt + 1, n_cvt);

  double P_yy = accessor(Pab, n_cvt, index_yy);
  double P_xx = accessor(Pab, n_cvt, index_xx);
  double P_xy = accessor(Pab, n_cvt, index_xy);
  double Px_yy = accessor(Pab, n_cvt + 1, index_yy);

  double beta = P_xy / P_xx;
  double tau = to!double(df) / Px_yy;
  double se = sqrt(1.0 / (tau * P_xx));
  double p_wald = gsl_cdf_fdist_Q((P_yy - Px_yy) * tau, 1.0, df);

  return Wald_score(beta, se, p_wald);
}

void calc_RL_Wald_batched(const size_t ni_test, const double[] l, loglikeparam params, const string[] indicators, File f) {
  const size_t n_cvt = params.n_cvt;
  const size_t n_index = (n_cvt + 2 + 1) * (n_cvt + 2) / 2;

  const int df = to!int(ni_test) - to!int(n_cvt) - 1;

  DMatrix Hi_eval= zeros_dmatrix(l.length, params.eval.elements.length);
  double[] v_temp_elements;

  foreach(i, snp; l){
    const DMatrix x =  divide_num_dmatrix(1, add_dmatrix_num(multiply_dmatrix_num(params.eval, snp), 1.0)) ;
    set_row2(Hi_eval, i, x);
  }

  const DMatrix Pab = calc_Pab_batched(n_cvt, Hi_eval, params.Uab, params.ab, [n_cvt + 2, n_index], l);

  SUMSTAT[] collection = new SUMSTAT[l.length];

  const size_t index_yy = GetabIndex(n_cvt + 2, n_cvt + 2, n_cvt);
  const size_t index_xx = GetabIndex(n_cvt + 1, n_cvt + 1, n_cvt);
  const size_t index_xy = GetabIndex(n_cvt + 2, n_cvt + 1, n_cvt);

  foreach(i, j ; l){
    const size_t row_no = i * (n_cvt + 2) + n_cvt;

    const double P_yy = accessor(Pab, row_no, index_yy);
    const double P_xx = accessor(Pab, row_no, index_xx);
    const double P_xy = accessor(Pab, row_no, index_xy);
    const double Px_yy = accessor(Pab, row_no + 1, index_yy);

    const double beta = P_xy / P_xx;
    const double tau = to!double(df) / Px_yy;
    const double se = sqrt(1.0 / (tau * P_xx));
    const double p_wald = gsl_cdf_fdist_Q((P_yy - Px_yy) * tau, 1.0, df);
    auto collect = SUMSTAT( beta, se, j, p_wald, indicators[i]);
    f.write(indicators[i], "\t", collect.beta, "\t", collect.se, "\t", j, "\t", collect.p_wald, "\n");
  }
}

alias Tuple!(double, "beta", double, "se", double, "p_score") RL_Score ;

RL_Score calc_RL_score(const size_t ni_test, const double l, loglikeparam params) {
  size_t n_cvt = params.n_cvt;
  size_t n_index = (n_cvt + 2 + 1) * (n_cvt + 2) / 2;

  int df = to!int(ni_test) - to!int(n_cvt) - 1;

  DMatrix v_temp = multiply_dmatrix_num(params.eval.T, l);
  DMatrix Hi_eval = (params.e_mode == 0 ? divide_num_dmatrix(1, add_dmatrix_num(v_temp, 1.0)) : dup_dmatrix(v_temp));

  DMatrix Pab = CalcPab(n_cvt, params.e_mode, Hi_eval, params.Uab, params.ab, [n_cvt + 2, n_index]);

  size_t index_yy = GetabIndex(n_cvt + 2, n_cvt + 2, n_cvt);
  size_t index_xx = GetabIndex(n_cvt + 1, n_cvt + 1, n_cvt);
  size_t index_xy = GetabIndex(n_cvt + 2, n_cvt + 1, n_cvt);
  double P_yy = accessor(Pab, n_cvt, index_yy);
  double P_xx = accessor(Pab, n_cvt, index_xx);
  double P_xy = accessor(Pab, n_cvt, index_xy);
  double Px_yy = accessor(Pab, n_cvt + 1, index_yy);

  double beta = P_xy / P_xx;
  double tau = to!double(df) / Px_yy;
  double se = sqrt(1.0 / (tau * P_xx));
  double p_score = gsl_cdf_fdist_Q(to!double(ni_test) * P_xy * P_xy / (P_yy * P_xx), 1.0, df);

  return RL_Score(beta, se, p_score);
}

DMatrix calc_Uab(const DMatrix UtW, const DMatrix Uty, const size_t ni_test, const size_t n_index) {
  size_t index_ab;
  size_t n_cvt = UtW.shape[1];

  DMatrix Uab = zeros_dmatrix(ni_test, n_index);

  DMatrix u_a;
  u_a.shape = [1, Uty.shape[1]];

  for (size_t a = 1; a <= n_cvt + 2; ++a) {
    if (a == n_cvt + 1) {
      continue;
    }

    if (a == n_cvt + 2) {
      u_a.elements = Uty.elements.dup_fast;
    } else {
      DMatrix UtW_col = get_col(UtW, a - 1);
      u_a.elements = UtW_col.elements.dup_fast;
    }
    for (size_t b = a; b >= 1; --b) {

      if (b == n_cvt + 1) {
        continue;
      }

      index_ab = GetabIndex(a, b, n_cvt);
      DMatrix Uab_col = get_col(Uab, index_ab);

      if (b == n_cvt + 2) {
        Uab_col.elements = Uty.elements.dup_fast;
      } else {
        DMatrix UtW_col = get_col(UtW, b - 1);
        Uab_col.elements = UtW_col.elements.dup_fast;
      }

      Uab_col = slow_multiply_dmatrix(Uab_col, u_a);
      Uab = set_col(Uab, index_ab, Uab_col);
    }
  }
  return Uab;
}

DMatrix calc_Uab(const DMatrix UtW, const DMatrix Uty, const DMatrix Utx, const DMatrix Uab_old) {
  size_t n_cvt = UtW.shape[1];
  DMatrix Uab = Uab_old.T;

  for (size_t b = 1; b <= n_cvt + 2; ++b) {
    size_t index_ab = GetabIndex(n_cvt + 1, b, n_cvt);
    DMatrix Uab_row = get_row(Uab, index_ab);

    if (b == n_cvt + 2) {
      Uab_row.elements = Uty.elements.dup_fast;
    } else if (b == n_cvt + 1) {
      Uab_row.elements = Utx.elements.dup_fast;
    } else {
      DMatrix UtW_col = get_col(UtW, b - 1);
      Uab_row.elements = UtW_col.elements.dup_fast;
    }

    Uab_row = slow_multiply_dmatrix(Uab_row, Utx);
    Uab = set_row(Uab, index_ab, Uab_row);
  }

  return Uab;
}

DMatrix calc_ab(const DMatrix W, const DMatrix y, const size_t[] shape) {
  size_t index_ab;
  size_t n_cvt = W.shape[1];

  double d;
  DMatrix v_a, v_b;
  v_a.shape = [1, y.shape[1]];
  v_b.shape = [1, y.shape[1]];

  DMatrix ab = zeros_dmatrix(shape[0], shape[1]);

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

  return ab;
}

DMatrix calc_ab(const DMatrix W, const DMatrix y, const DMatrix x, const size_t[] shape) {
  size_t index_ab;
  size_t n_cvt = W.shape[1];

  DMatrix ab = zeros_dmatrix(shape[0], shape[1]);

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

  return ab;
}

alias Tuple!(const double, "vg", const double, "ve", DMatrix, "beta", DMatrix, "se_beta") Mle_result;

// Obtain REML estimate for Vg and Ve using lambda_remle.
// Obtain beta and se(beta) for coefficients.
// ab is not used when e_mode==0.
Mle_result CalcLmmVgVeBeta(const DMatrix eval, const DMatrix UtW, const DMatrix Uty, const double lambda) {

  writeln("in CalcLmmVgVeBeta");
  double vg, ve;
  double[] se_beta;
  size_t n_cvt = UtW.shape[1], ni_test = UtW.shape[0];
  size_t n_index = (n_cvt + 2 + 1) * (n_cvt + 2) / 2;

  DMatrix Uab = calc_Uab(UtW, Uty, ni_test, n_index);

  DMatrix v_temp = multiply_dmatrix_num(eval.T, lambda);
  DMatrix Hi_eval = divide_num_dmatrix(1, add_dmatrix_num(v_temp, 1.0));

  // Calculate beta.
  DMatrix HiW =  UtW.T;
  foreach (i; 0..UtW.shape[1]) {
    DMatrix HiW_col = get_col(HiW, i);
    HiW_col = slow_multiply_dmatrix(HiW_col, Hi_eval);
    set_col2(HiW, i, HiW_col);
  }

  DMatrix WHiW = matrix_mult(HiW, UtW);
  DMatrix WHiy = matrix_mult(HiW, Uty);

  double[] beta = WHiy.elements;

  auto m = to!int(WHiW.cols);
  assert(m>=1);
  auto n = to!int(WHiW.rows);
  int nrhs = n;

  auto ipiv = new int[min(m,n)+1];

  int lda = n;
  int ldb = m;

  enforce(LAPACKE_dgesv( 101, n, n, WHiW.elements.ptr, lda, ipiv.ptr,
                      beta.ptr,  ldb ) == 0);

  DMatrix Vbeta = inverse(WHiW);

  DMatrix ab = calc_ab(UtW, Uty, [1, n_index]);
  DMatrix Pab = CalcPab(n_cvt, 0, Hi_eval, Uab.T, ab, [n_cvt + 2, n_index]);

  size_t index_yy = GetabIndex(n_cvt + 2, n_cvt + 2, n_cvt);
  double P_yy = accessor(Pab, n_cvt, index_yy);

  ve = P_yy / to!double(ni_test - n_cvt);
  vg = ve * lambda;

  Vbeta = multiply_dmatrix_num(Vbeta, ve);

  foreach ( i; 0..Vbeta.shape[1]) {
    se_beta ~= sqrt(accessor(Vbeta, i, i));
  }

  writeln("out of CalcLmmVgVeBeta");

  return Mle_result(vg, ve, DMatrix([1, beta.length], beta), DMatrix([1, se_beta.length], se_beta));
}

// Obtain REMLE estimate for PVE using lambda_remle.

alias Tuple!(const double, "pve", const double, "pve_se") Pve_result;

Pve_result calc_pve(const DMatrix eval, const DMatrix UtW,
             const DMatrix Uty, const double lambda, const double trace_G) {
  writeln("in calc_pve");

  size_t n_cvt = UtW.shape[1], ni_test = UtW.shape[0];
  size_t n_index = (n_cvt + 2 + 1) * (n_cvt + 2) / 2;

  DMatrix ab;
  ab.shape = [1, n_index];

  DMatrix Uab = calc_Uab(UtW, Uty, ni_test, n_index);

  loglikeparam param0 = loglikeparam(true, ni_test, n_cvt, eval, Uab.T, ab, 0);

  double se = sqrt(-1.0 / LogRL_dev2(lambda, &param0));

  double pve = trace_G * lambda / (trace_G * lambda + 1.0);
  double pve_se = trace_G / ((trace_G * lambda + 1.0) * (trace_G * lambda + 1.0)) * se;
  writeln("out of calc_pve");

  return Pve_result(pve, pve_se);
}


unittest{
  size_t n_cvt = 2;
  size_t e_mode = 0;
  DMatrix Hi_eval = DMatrix([2,2],[1,2,3,4]);
  DMatrix Uab = DMatrix([2,2],[1,2,3,4]);
  DMatrix ab = DMatrix([2,2],[1,2,3,4]);
  DMatrix Pab;

  size_t index = GetabIndex(4, 9, 16);
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
  //calc_lambda(func_name, cast(void *)&params, l_min, l_max, n_region, lambda, logf);
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
  //calc_lambda(func_name, eval, UtW, Uty, l_min, l_max, n_region, lambda, logl_H0);
  //assert();

  double l = 6;
  double beta;
  double se;
  double p_wald;
  //CalcRLWald(ni_test, l, params, beta, se, p_wald);
  //assert();

  double p_score;
  //calc_RL_score(ni_test, l, params, beta, se, p_score);
  //assert();

  //calc_Uab(UtW, Uty, Uab);
  //assert();

  //calc_Uab(UtW, Uty, Utx, Uab);
  //assert();

  //calc_ab(W, y, ab);
  //assert();

  DMatrix x;
  //calc_ab(W, y, x, ab);
  //assert();

  double vg, ve;
  DMatrix se_beta;
  //CalcLmmVgVeBeta(eval, UtW, Uty, lambda, vg, ve, beta, se_beta);

  double trace_G, pve, pve_se;
  //calc_pve(eval,  UtW, Uty, lambda, trace_G, pve,  pve_se);
}

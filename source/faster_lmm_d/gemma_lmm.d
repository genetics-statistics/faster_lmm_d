module faster_lmm_d.gemma_lmm;

import std.conv;
import std.exception;
import std.math;
alias mlog = std.math.log;
import std.stdio;
import std.typecons;
import std.experimental.logger;

import faster_lmm_d.dmatrix;
import faster_lmm_d.optmatrix;

import gsl.errno;
import gsl.math;
import gsl.min;

void CalcPab(const size_t n_cvt, const size_t e_mode, const DMatrix Hi_eval,
             const DMatrix Uab, const DMatrix ab, DMatrix Pab) {
  size_t index_ab, index_aw, index_bw, index_ww;
  double p_ab;
  double ps_ab, ps_aw, ps_bw, ps_ww;

  for (size_t p = 0; p <= n_cvt + 1; ++p) {
    for (size_t a = p + 1; a <= n_cvt + 2; ++a) {
      for (size_t b = a; b <= n_cvt + 2; ++b) {
        index_ab = GetabIndex(a, b, n_cvt);
        if (p == 0) {
          DMatrix Uab_col = get_col(Uab, index_ab);
          p_ab = matrix_mult(Hi_eval, Uab_col).elements[0];  // check its shape is [1,1] else take transpose of Hi_eval
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
              const DMatrix ab, const DMatrix Pab, DMatrix PPab) {
  size_t index_ab, index_aw, index_bw, index_ww;
  double p2_ab;
  double ps2_ab, ps_aw, ps_bw, ps_ww, ps2_aw, ps2_bw, ps2_ww;

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
}

double LogL_f(double l, loglikeparam p) {

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

  v_temp.elements = p.eval.elements;

  v_temp = multiply_dmatrix_num(v_temp, l);

  if (p.e_mode == 0) {
    //gsl_vector_set_all(Hi_eval, 1.0);
  } else {
    Hi_eval.elements = v_temp.elements;
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

double LogRL_f(double l, loglikeparam p) {
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
  v_temp.elements = p.eval.elements;

  v_temp = multiply_dmatrix_num(v_temp, l);
  if (p.e_mode == 0) {
    //gsl_vector_set_all(Hi_eval, 1.0);
  } else {
    Hi_eval.elements = v_temp.elements;
  }
  v_temp = add_dmatrix_num(v_temp, 1.0);
  Hi_eval = divide_dmatrix(Hi_eval, v_temp);

  for (size_t i = 0; i < p.eval.elements.length; ++i) {
    d = v_temp.elements[i];
    logdet_h += mlog(fabs(d));
  }

  CalcPab(n_cvt, p.e_mode, Hi_eval, p.Uab, p.ab, Pab);

  //gsl_vector_set_all(v_temp, 1.0);
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

  double c = 0.5 * df * (mlog(df) - mlog(2 * M_PI) - 1.0);
  f = c - 0.5 * logdet_h - 0.5 * logdet_hiw - 0.5 * df * mlog(P_yy);

  return f;
}

double LogRL_dev1(double l, loglikeparam p) {
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
  v_temp.elements = p.eval.elements;

  v_temp = multiply_dmatrix_num(v_temp, l);
  if (p.e_mode == 0) {
    //gsl_vector_set_all(Hi_eval, 1.0);
  } else {
    Hi_eval.elements = v_temp.elements;
  }

  v_temp = add_dmatrix_num(v_temp, 1.0);
  Hi_eval = divide_dmatrix(Hi_eval, v_temp);


  HiHi_eval.elements =  Hi_eval.elements.dup;
  HiHi_eval = slow_multiply_dmatrix(HiHi_eval, Hi_eval);

  //gsl_vector_set_all(v_temp, 1.0);
  trace_Hi = matrix_mult(Hi_eval, v_temp).elements[0];

  if (p.e_mode != 0) {
    trace_Hi = to!double(ni_test) - trace_Hi;
  }

  CalcPab(n_cvt, p.e_mode, Hi_eval, p.Uab, p.ab, Pab);
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

  // Calculate yPKPy, yPKPKPy.
  index_ww = GetabIndex(n_cvt + 2, n_cvt + 2, n_cvt);
  double P_yy = accessor(Pab, nc_total, index_ww);
  double PP_yy = accessor(PPab, nc_total, index_ww);
  double yPKPy = (P_yy - PP_yy) / l;

  dev1 = -0.5 * trace_PK + 0.5 * df * yPKPy / P_yy;

  return dev1;
}

double LogL_dev1(double l, loglikeparam p) {
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
    //gsl_vector_set_all(Hi_eval, 1.0);
  } else {
    Hi_eval.elements = v_temp.elements.dup;
  }
  v_temp = add_dmatrix_num(v_temp, 1.0);
  HiHi_eval = divide_dmatrix(Hi_eval, v_temp);

  HiHi_eval.elements = Hi_eval.elements.dup;
  HiHi_eval = slow_multiply_dmatrix(HiHi_eval, Hi_eval);

  //gsl_vector_set_all(v_temp, 1.0);
  trace_Hi = matrix_mult(Hi_eval, v_temp).elements[0];

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

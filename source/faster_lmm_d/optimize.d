module faster_lmm_d.optimize;
import gsl.errno;
import gsl.math;
import gsl.min;
import std.stdio;
import std.math;

//returns
//xmin : ndarray
//Optimum point.
//fval : float
//Optimum value.
//iter : int
//Number of iterations.
//funcalls : int
//Number of objective function evaluations made

double brent(string func, string args, double[] brack, double tol=1.48e-08, int full_output=0, int maxiter=500){
  return 0;
}

void _minimize_scalar_brent(string func, double[] brack, double xtol=1.48e-8, int maxiter=500){
}


extern (C) double fn1 (double x, void* params)
{
  return cos(x) + 1.0;
}

void checkGSL()
{
  writeln("In checkGSL");
  int status;
  int iter = 0, max_iter = 100;
  const(gsl_min_fminimizer_type) *T;
  gsl_min_fminimizer *s;
  double m = 2.0, m_expected = M_PI;
  double a = 0.0, b = 6.0;
  gsl_function F;
  F.function_ = &fn1;
  //F.params = 0;
  T = gsl_min_fminimizer_brent;
  s = gsl_min_fminimizer_alloc (T);
  gsl_min_fminimizer_set (s, &F, m, a, b);

  writeln("using ",*gsl_min_fminimizer_name(s)," method");

  writeln("iter\t", "lower\t", "upper\t", "min\t",
          "err\t\t", "err(est)\t");

  writeln(iter, "\t",a, "\t", b, "\t",
          m, "\t", m - m_expected, "\t", b - a);

  do
    {
      iter++;
      status = gsl_min_fminimizer_iterate (s);

      m = gsl_min_fminimizer_x_minimum (s);
      a = gsl_min_fminimizer_x_lower (s);
      b = gsl_min_fminimizer_x_upper (s);

      status
        = gsl_min_test_interval (a, b, 0.001, 0.0);

      if (status == GSL_SUCCESS)
        writeln("Converged:\n");

      writeln(iter, "\t",a, "\t", b, "\t",
          m, "\t", m - m_expected, "\t", b - a);
    }
  while (status == GSL_CONTINUE && iter < max_iter);

  gsl_min_fminimizer_free (s);

  //return status;
}

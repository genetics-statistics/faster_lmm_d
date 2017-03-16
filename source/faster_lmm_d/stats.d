import dstats;

double _sf(double x){
  return _norm_sf(x);
}

double _norm_sf(double x){
  return normalCDF(-x);
}


double _logsf(double x){
  return _norm_logsf(x);
}

double _norm_logsf(double x){
  return normalCDF(-x);
}



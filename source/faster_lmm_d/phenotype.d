/*
   This code is part of faster_lmm_d and published under the GPLv3
   License (see LICENSE.txt)

   Copyright © 2017 Prasun Anand & Pjotr Prins
*/

module faster_lmm_d.phenotype;

import std.experimental.logger;
import std.typecons;

import faster_lmm_d.helpers;

alias Tuple!(double[], "Y", bool[], "keep", int, "n") phenoStruct;

phenoStruct remove_missing( int n, double[] y){

  //Remove missing data. Returns new n,y,keep

  trace("In remove missing new");
  bool[] v = isnan(y);
  bool[] keep = negateBool(v);
  double[] Y = getNumArray(y,keep);
  n = cast(int)Y.length;
  return phenoStruct(Y, keep, n);

}

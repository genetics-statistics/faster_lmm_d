/*
   This code is part of faster_lmm_d and published under the GPLv3
   License (see LICENSE.txt)

   Copyright © 2017 - 2018 Prasun Anand & Pjotr Prins
*/

module faster_lmm_d.phenotype;

import std.conv;
import std.experimental.logger;
import std.typecons;

import faster_lmm_d.helpers;

alias Tuple!(immutable double[], "Y", const bool[], "keep", immutable ulong, "n") PhenoStruct;

PhenoStruct remove_missing( const ulong n, const double[] y){
  //Remove missing data. Returns new n,y,keep

  trace("In remove missing new");
  bool[] v = is_nan(y);
  bool[] keep = negate_bool(v);
  double[] Y = get_num_array(y,keep);
  return PhenoStruct(cast(immutable)Y, keep, to!int(Y.length));
}

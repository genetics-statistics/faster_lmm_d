module simplelmm.dmatrix;

struct dmatrix{
  int[] shape;
  double[] elements;

  this(int[] s, double[] e){
    shape = s;
    elements = e;
  }
}

dmatrix newDmatrix(dmatrix inDmat, int start, int end){
  return inDmat;
}
module simplelmm.dmatrix;

struct dmatrix{
  int[] shape;
	double[] elements;

  this(int[] s, double[] e){
    shape = s;
    elements = e;
  }
}
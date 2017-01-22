module simplelmm.dmatrix;
import std.math;

struct dmatrix{
  int[] shape;
  double[] elements;
  bool init = false;

  this(int[] s, double[] e){
    shape = s;
    elements = e;
    init = true;
  }

  double acc(int row, int col){
    return this.elements[row*this.shape[0]+col];
  }
}

dmatrix newDmatrix(dmatrix inDmat, int start, int end){
  return inDmat;
}
dmatrix logDmatrix(dmatrix inDmat){
  double[] elements;
  for(int i = 0; i < inDmat.shape[0]*inDmat.shape[1]; i++){
    elements ~= log(inDmat.elements[i]);
  }
  return dmatrix(inDmat.shape, elements);
}

dmatrix addDmatrix(dmatrix lha, dmatrix rha){
  assert(lha.shape[0] == rha.shape[0]);
  assert(lha.shape[1] == rha.shape[1]);
  double[] elements;
  for(int i = 0; i < lha.shape[0]*lha.shape[1]; i++){
    elements ~= lha.elements[i] + rha.elements[i];
  }
  return dmatrix(lha.shape, elements);
}

dmatrix subDmatrix(dmatrix lha, dmatrix rha){
  assert(lha.shape[0] == rha.shape[0]);
  assert(lha.shape[1] == rha.shape[1]);
  double[] elements;
  for(int i = 0; i < lha.shape[0]*lha.shape[1]; i++){
    elements ~= lha.elements[i] - rha.elements[i];
  }
  return dmatrix(lha.shape, elements);
}

dmatrix multiplyDmatrix(dmatrix lha, dmatrix rha){
  assert(lha.shape[0] == rha.shape[0]);
  assert(lha.shape[1] == rha.shape[1]);
  double[] elements;
  for(int i = 0; i < lha.shape[0]*lha.shape[1]; i++){
    elements ~= lha.elements[i] * rha.elements[i];
  }
  return dmatrix(lha.shape, elements);
}

dmatrix subDmatrixNum(dmatrix input, double num){
  double[] elements;
  for(int i = 0; i < input.shape[0]*input.shape[1]; i++){
    elements ~= input.elements[i] - num;
  }
  return dmatrix(input.shape, elements);
}

dmatrix multiplyDmatrixNum(dmatrix input, double num){
  double[] elements;
  for(int i = 0; i < input.shape[0]*input.shape[1]; i++){
    elements ~= input.elements[i] * num;
  }
  return dmatrix(input.shape, elements);
}

dmatrix addDMatrixNum(dmatrix input, double num){
  double[] elements;
  for(int i = 0; i < input.shape[0]*input.shape[1]; i++){
    elements ~= input.elements[i] + num;
  }
  return dmatrix(input.shape, elements);
}

dmatrix divideDmatrixNum(dmatrix input, double factor){
  double[] elements;
  for(int i = 0; i < input.shape[0]*input.shape[1]; i++){
    elements ~= input.elements[i]/factor;
  }
  return dmatrix(input.shape, elements);
}

dmatrix zerosMatrix(int rows, int cols){
  double[] elements;
  for(int i = 0; i < rows*cols; i++){
    elements ~= 0;
  }
  return dmatrix([rows, cols], elements);
}

dmatrix onesMatrix(int rows, int cols){
  double[] elements;
  for(int i = 0; i < rows*cols; i++){
    elements ~= 1;
  }
  return dmatrix([rows, cols], elements);
}

double sumArray(double[] arr){
  return 1;  
}

dmatrix horizontallystack(dmatrix a, dmatrix b){
  int n = a.shape[0];
  double[] arr;
  for(int i = 0; i < n; i++){
    arr ~= a.elements[(a.shape[1]*i)..(a.shape[1]*(i+1))];
    arr ~= b.elements[(b.shape[1]*i)..(b.shape[1]*(i+1))];
  }
  return dmatrix([a.shape[0], a.shape[1]+b.shape[1]], arr);
}

bool[] compareGt(dmatrix lha, double val){
  bool[] result;
  foreach(element; lha.elements){
    if(element > val){
      result ~= true;
    }
    else{
      result ~= false;
    }
  }
  return result;
}
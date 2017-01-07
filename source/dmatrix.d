module simplelmm.dmatrix;

struct dmatrix{
  int[] shape;
  double[] elements;
  bool init = false;

  this(int[] s, double[] e){
    shape = s;
    elements = e;
    init = true;
  }
}

dmatrix newDmatrix(dmatrix inDmat, int start, int end){
  return inDmat;
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

double gethstack(dmatrix a, dmatrix b){
  return 1;
}

double sumArray(double[] arr){
  return 1;  
}
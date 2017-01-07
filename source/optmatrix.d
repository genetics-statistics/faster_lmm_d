module simplelmm.optmatrix;
import simplelmm.dmatrix;
import simplelmm.helpers;
import std.stdio;
import cblas;
import lapack;

dmatrix matrixMult(ref dmatrix lha, ref dmatrix rha){
	writeln("In matMul");
  auto C = new double[lha.shape[0]*rha.shape[1]];
  gemm(Order.RowMajor, Transpose.NoTrans, Transpose.NoTrans, lha.shape[0], rha.shape[1], lha.shape[1], /*no scaling*/
    1,lha.elements.ptr, lha.shape[1], rha.elements.ptr, rha.shape[1], /*no addition*/0, C.ptr, rha.shape[1]);
  int[] resshape = [lha.shape[0],rha.shape[1]];
  return dmatrix(resshape, C);
}

dmatrix matrixMultT(ref dmatrix lha, ref dmatrix rha){
  writeln("In matMulTranspose");
  double[] A = [1, 0, 0,
                  0, 1, 1];
  double[] B = [1, 0,
                  0, 1,
                  2, 2];
  auto C = new double[lha.shape[0]*rha.shape[0]];
  gemm(Order.RowMajor, Transpose.NoTrans, Transpose.NoTrans, lha.shape[0], rha.shape[0], lha.shape[1], /*no scaling*/
    1,lha.elements.ptr, lha.shape[1], rha.elements.ptr, rha.shape[0], /*no addition*/0, C.ptr, rha.shape[0]);
  int[] resshape = [lha.shape[0],rha.shape[0]];
  return dmatrix(resshape, C);
}

dmatrix matrixTranspose(dmatrix input){
  writeln("In matrixTranspose");
  auto matrix = new double[input.shape[0]*input.shape[1]];
  double[] output = new double[input.shape[0]*input.shape[1]];
  int index = 0;
  for(int i=0; i< input.shape[1]; i++){
    for(int j=0; j< input.shape[0]; j++){
      output[index] = input.elements[j*input.shape[1]+i];
      index++;
    }
  }
  int[] resshape = [input.shape[1],input.shape[0]];
  return dmatrix(resshape,output);

}

void prettyPrint(dmatrix input){
  writeln("[");
  if(input.shape[0]>6){
    for(int i=0; i < 3; i++){
      writeln(input.elements[(input.shape[1]*i)..(input.shape[1]*(i+1))]);
    }
    writeln("...");
    for(int i=input.shape[0]-3; i < input.shape[0]; i++){
      writeln(input.elements[(input.shape[1]*i)..(input.shape[1]*(i+1))]);
    }
  }
  else{
    for(int i=0; i < input.shape[0]; i++){
      writeln(input.elements[(input.shape[1]*i)..(input.shape[1]*(i+1))]);
    }
  }
  
  writeln("]");
}

dmatrix sliceDmatrix(dmatrix input, int[] along){
  writeln("In sliceDmatrix");
  writeln(along.length);
  writeln(along);
  double[] output;
  foreach(int rowIndex; along){
    for(int i=rowIndex*input.shape[1]; i < (rowIndex+1)*input.shape[1]; i++){
      output ~= input.elements[i];
    }
  }
  return dmatrix([cast(int)along.length,input.shape[1]],output);
}

dmatrix sliceDmatrixKeep(dmatrix input, bool[] along){
  writeln("In sliceDmatrix");
  assert(along.length == input.shape[0]);
  double[] output;
  int rowIndex = 0;
  int shape0 = 0;
  foreach(bool toKeep; along){
    if(toKeep){
      for(int i=rowIndex*input.shape[1]; i < (rowIndex+1)*input.shape[1]; i++){
        output ~= input.elements[i];
      }
      shape0++;
    }
    rowIndex++;
    
  }
  return dmatrix([shape0,input.shape[1]],output);
}

void normalize_along_row(ref dmatrix G, dmatrix input){
  double[] largeArr;
  double[] arr;
  for(int i = 0; i < input.shape[0]; i++){
    arr = input.elements[(input.shape[1]*i)..(input.shape[1]*(i+1))];
    bool[] missing = isnan(arr);
    bool[] valuesArr = negateBool(missing);
    double[] values = getNumArray(arr,valuesArr);
    double mean = globalMean(values);
    double variation = getVariation(values, mean);
    double stddev = std.math.sqrt(variation);
    replaceNaN(arr, valuesArr, mean);
    if(stddev == 0){
      foreach(ref elem; arr){
        elem -= mean;      
      }
    }else{
      foreach(ref elem; arr){
        elem = (elem - mean) / stddev;      
      }
    }
    largeArr ~= arr;
  }
  G = dmatrix(input.shape, largeArr);
}

//dmatrix eigh(dmatrix input){

//}

double det(dmatrix input){

}

dmatrix inverse(dmatrix input){

    //int *IPIV = new int[N+1];
  double[] elements= input.elements.dup;
  //writeln(input.elements);
  int LWORK = input.shape[0]*input.shape[0];      
  double[] WORK = new double[input.shape[0]*input.shape[0]];
  auto ipiv = new int[input.shape[0]+1];
  //writeln("In matrix inverse");
  auto result = new double[input.shape[0]*input.shape[1]];
  int info;
  int output = LAPACKE_dgetrf(101, input.shape[0],input.shape[0],elements.ptr,input.shape[0],ipiv.ptr);
  int[] resshape = [input.shape[0],input.shape[0]];
  //writeln("After getrf");
  //writeln(output);
  writeln(ipiv);
  LAPACKE_dgetri(101, input.shape[0],elements.ptr, input.shape[0], ipiv.ptr);
  //writeln("After getri");
  //writeln(input.elements);
  return dmatrix(input.shape, elements);
} 
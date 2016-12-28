module simplelmm.optmatrix;
import simplelmm.dmatrix;
import std.stdio;
import cblas;

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
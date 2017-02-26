module faster_lmm_d.optmatrix;
import faster_lmm_d.dmatrix;
import faster_lmm_d.helpers;
import std.stdio;
import cblas;
import lapack;

dmatrix matrixMult(dmatrix lha, dmatrix rha){
  double[] C = new double[lha.shape[0]*rha.shape[1]];
  gemm(Order.RowMajor, Transpose.NoTrans, Transpose.NoTrans, lha.shape[0], rha.shape[1], lha.shape[1], /*no scaling*/
    1,lha.elements.ptr, lha.shape[1], rha.elements.ptr, rha.shape[1], /*no addition*/0, C.ptr, rha.shape[1]);
  int[] resshape = [lha.shape[0],rha.shape[1]];
  return dmatrix(resshape, C);
}

dmatrix matrixMultT(dmatrix lha, dmatrix rha){
  double[] C = new double[lha.shape[0]*rha.shape[0]];
  gemm(Order.RowMajor, Transpose.NoTrans, Transpose.NoTrans, lha.shape[0], rha.shape[0], lha.shape[1], /*no scaling*/
    1,lha.elements.ptr, lha.shape[1], rha.elements.ptr, rha.shape[0], /*no addition*/0, C.ptr, rha.shape[0]);
  int[] resshape = [lha.shape[0],rha.shape[0]];
  return dmatrix(resshape, C);
}

dmatrix matrixTranspose(dmatrix input){
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

void pPrint(dmatrix input){
  writeln("[");
  for(int i=0; i < input.shape[0]; i++){
    writeln(input.elements[(input.shape[1]*i)..(input.shape[1]*(i+1))]);
  }
  writeln("]");
}

void pPrint2(dmatrix input){
  writeln("[");
  for(int i=0; i < input.shape[0]; i++){
    writeln(input.elements[(input.shape[1]*i)..(input.shape[1]*i+3)],"...", input.elements[(input.shape[1]*(i+1)-3)..(input.shape[1]*(i+1))]);
  }
  writeln("]");
}

void pPrint3(dmatrix input){
  writeln("[");
  if(input.shape[0]>6){
    for(int i=0; i < 3; i++){
      writeln(input.elements[(input.shape[1]*i)..(input.shape[1]*i+3)],"...", input.elements[(input.shape[1]*(i+1)-3)..(input.shape[1]*(i+1))]);
    }
    writeln("...");
    for(int i=input.shape[0]-3; i < input.shape[0]; i++){
      writeln(input.elements[(input.shape[1]*i)..(input.shape[1]*i+3)],"...", input.elements[(input.shape[1]*(i+1)-3)..(input.shape[1]*(i+1))]);
    }
  }
  
  writeln("]");
}


dmatrix sliceDmatrix(dmatrix input, int[] along){
  writeln("In sliceDmatrix");
  //writeln(along.length);
  //writeln(along);
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

dmatrix normalize_along_row(dmatrix input){
  double[] largeArr;
  double[] arr;
  writeln(input.shape);
  for(int i = 0; i < input.shape[0]; i++){
    arr = input.elements[(input.shape[1]*i)..(input.shape[1]*(i+1))].dup;
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
  return dmatrix(input.shape, largeArr);
}

dmatrix removeCols(dmatrix input, bool[] keep){
  int colLength = sum(keep);
  double[] arr = new double[input.shape[0]*colLength];
  int index = 0;
  for(int i= 0; i < input.shape[0]; i++){
    for(int j = i*input.shape[1], count = 0; j < (i+1)*input.shape[1]; j++){
      if(keep[count] == true){
        arr[index] = input.elements[j];
        index++;
      }
      count++;
    }
  }
  int[] shape = [input.shape[0], sum(keep)];
  return dmatrix(shape, arr);
}


void eigh(dmatrix input,ref dmatrix kva, ref dmatrix kve){

  double[] z = new double[input.shape[0] * input.shape[1]]; //eigenvalues
  double[] w = new double[input.shape[0]];  // eigenvectors
  double[] elements = input.elements.dup;

  double wi;
  int n = input.shape[0];
  double vu, vl;
  //vl = 0;
  //vu = 1.0;
  int[] m = new int[input.shape[0]];
  int[] isuppz = new int[2*input.shape[0]];
  int il = 1;
  int iu = input.shape[1];
  int ldz = n;
  double abstol = -1; //default value for abstol
  
  LAPACKE_dsyevr(101, 'V', 'A', 'L', n, input.elements.ptr, n, vl, vu, il, iu, abstol, m.ptr, w.ptr, z.ptr, ldz, isuppz.ptr);
  
  kve = dmatrix([input.shape[0],1], w);
  kva = dmatrix(input.shape, z);
}



double det(dmatrix input){
  double[] narr = input.elements.dup;
  int[] pivot = getrf(narr, input.shape.dup);

  int num_perm = 0;
  int j = 0;
  foreach(swap; pivot){
    if(swap-1 != j){num_perm += 1;}
    j++;
  }
  double prod;
  if(num_perm % 2 == 1){
    prod = 1;
  }else{
   prod = -1; //# odd permutations => negative
  }
  int min = input.shape[0];
  if(input.shape[0]> input.shape[1]){min = input.shape[1];}
  for(int i =0;i< min; i++){
    prod *= narr[input.shape[0]*i + i];
  }
  return prod;
}

int[] getrf(double[] arr, int[] shape){
  auto ipiv = new int[shape[0]+1];
  LAPACKE_dgetrf(101, shape[0],shape[0],arr.ptr,shape[0],ipiv.ptr);
  return ipiv;
}

dmatrix inverse(dmatrix input){
  double[] elements= input.elements.dup;
  int LWORK = input.shape[0]*input.shape[0];      
  double[] WORK = new double[input.shape[0]*input.shape[0]];
  auto ipiv = new int[input.shape[0]+1];
  auto result = new double[input.shape[0]*input.shape[1]];
  int info;
  int output = LAPACKE_dgetrf(101, input.shape[0],input.shape[0],elements.ptr,input.shape[0],ipiv.ptr);
  int[] resshape = [input.shape[0],input.shape[0]];
  LAPACKE_dgetri(101, input.shape[0],elements.ptr, input.shape[0], ipiv.ptr);
  return dmatrix(input.shape, elements);
}

unittest{
  dmatrix d1 = dmatrix([3,4],[2,4,5,6, 7,8,9,10, 2,-1,-4,3]);
  dmatrix d2 = dmatrix([4,2],[2,7,8,9, -5,2,-1,-4]);
  dmatrix d3 = dmatrix([3,2], [5,36,23, 99,13,-15]);
  assert(matrixMult(d1,d2) == d3);
  
  dmatrix d4 = dmatrix([2,2], [2, -1, -4, 3]);
  dmatrix d5 = dmatrix([2,2], [1.5, 0.5, 2, 1]);
  assert(inverse(d4) == d5);
  
  dmatrix d6 = dmatrix([2,2],[2, -4, -1, 3]);
  assert(matrixTranspose(d4) == d6);
  
  dmatrix d7 = dmatrix([4,2],[-3,13,7, -5, -12, 26, 2, -8]);
  assert(matrixMultT(d2, d6) == d7);
  
  assert(det(d4) == 2);
  
  dmatrix eigenvalue, dvl, dvr;
  dmatrix mat = dmatrix([3,3], [2,0,0, 0,3,2, 0,1,2]);
  eigh(mat, eigenvalue, dvl, dvr);
  // dmatrix ev = dmatrix([3,1], []);
  // dmatrix vl = dmatrix([3,3], []);
  //dmatrix vr = dmatrix([3,1], []);
  // assert(eqeq(ev, eigehnvalue));
  // assert(eqeq(vl, dvl));
  // assert(eqeq(vr, dvr));
}


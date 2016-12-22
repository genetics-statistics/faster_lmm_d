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

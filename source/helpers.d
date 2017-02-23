module simplelmm.helpers;
import std.math;
import std.stdio;

double modDiff(double x, double y){
  double rem = y - x;
  if(rem<0){return -rem;}
  return rem;
}

bool[] isnan(double[] vector){
  bool[] result;
  foreach(element; vector){
    result ~= isNaN(element);
  }
  return result;
}

bool[] negateBool(bool[] vector){
  bool[] result;
  foreach(element; vector){
    result ~= true - element;
  }
  return result;
}

double sum(double[] vector){
  double result = 0;
  foreach(element;vector){result+=element;}
  return result;
}

int sum(bool[] vector){
  int result = 0;
  foreach(element;vector){
    if(element == true){
      result +=1;
    }
  }
  return result;
}

double[] array(int n){
  return [];
}

int range(int n){
  return 0;
}

double globalMean(double[] input){
  return sum(input)/input.length;
}

double getVariation(double[] vector, double mean){
  double result = 0;
  foreach(element;vector){result+= pow(element-mean,2);}
  return result/vector.length;
}

double[] getNumArray(double[] arr,bool[] valuesArr){
  double[] result = new double[sum(valuesArr)];
  for(int k = 0, index = 0 ; k < arr.length; k++){
    if(valuesArr[k] == true){
      result[index] = arr[k];
      index++;
    }
  }
  return result;
}

void replaceNaN(ref double[] arr, bool[] valuesArr, double mean){
  int index = 0;
  foreach(ref element; valuesArr){
    if(element == true){
      index++;
    }else{
      arr[index] = mean;
      index++;
    }
  }
}

double[] rangeArray(int count){
  double[] arr;
  for(int i = 0; i < count; i++){
    arr ~= i;
  }
  return arr;
}

unittest{
  double[] arr = [4,3,4,5];
  bool[] arr2 = [true, false, true, true];

  assert(sum(arr) == 16);
  assert(sum(arr2) == 3);
  assert(globalMean(arr) == 4); 
}

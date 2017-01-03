module simplelmm.helpers;
import std.math;


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
  double[] result;
  int index = 0;
  foreach(element; arr){
    if(valuesArr[index] == true){
      result ~= element;
      index++;
    }
  }
  return result;
}

void replaceNaN(ref double[] arr, bool[] valuesArr, double mean){
  int index = 0;
  foreach(element; valuesArr){
    if(element == true){
      index++;
    }else{
      arr[index] = mean;
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
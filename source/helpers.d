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
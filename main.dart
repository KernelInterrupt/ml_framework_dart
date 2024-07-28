import 'dart:async';
import 'autograd.dart';
import 'tensor.dart';
Node func(Node x,Node y){

  return x.pow(Node(createTensor([2.0,2.0])))+y.pow(Node(createTensor([2.0,2.0])));
}
void main(){

var f=[1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0,10.0,11.0,12.0];
var h=Tensor(f, [2,3,2]);
var x=[1.0,2.0,3.0,4.0];
var g=Tensor(x, [2,1,2]);
var list=broadcast(h, g);
print(h);
print(g);
print(list[0]);
print(list[1]);
}
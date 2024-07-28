import 'dart:async';
import 'autograd.dart';
import 'tensor.dart';
Node func(Node x,Node y){

  return x.pow(Node(createTensor([2.0,2.0])))+y.pow(Node(createTensor([2.0,2.0])));
}
void main(){

var f=[[1.0,2.0,3.0],[4.0,5.0,6.0]];
var h=createTensor(f);
print(h);
print(h.strides);
print(f);
print(h[1][0]);
print(h.repeat(4,axis: 1));
}
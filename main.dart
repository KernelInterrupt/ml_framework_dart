import 'dart:async';
import 'autograd.dart';
import 'tensor.dart';
Node func(Node x,Node y){

  return x.pow(Node(createTensor([2.0,2.0])))+y.pow(Node(createTensor([2.0,2.0])));
}
void main(){

Node a=Node(createTensor([2.0,2.0]));
Node b=Node(createTensor([2.0,2.0]));
Node z=func(a,b);
z.backward(createTensor([1.0,1.0]));
var h=Tensor([1.0,2.0,3.0,4.0,5.0,6.0], [2,3]);
print(h.strides);
}
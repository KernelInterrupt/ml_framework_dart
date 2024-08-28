import 'dart:async';
import 'autograd.dart';
import 'tensor.dart';
Node func(Node x,Node y){

  return x-y;
}
void main(){
var x=Node(Tensor([2.0],[1]));
var y=Node(Tensor([1.0,2.0,3.0],[3]));
var z=func(x, y);
z.backward();
print(x.grad);
print(y.grad);
print(z.grad);
}
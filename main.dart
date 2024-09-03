import 'dart:async';
import 'autograd.dart';
import 'tensor.dart';
Node func(Node x,Node y){

  return x.matmul(y);
}
void main(){
var y=Node(Tensor([1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0,10.0,11.0,12.0],[2,3,2]));
var x=Node(Tensor([1.0,2.0,3.0],[3]));
var z=func(x, y);
z.backward();
print(z.tensor);
print(z.grad);
print(x.grad);
print(y.grad);
}
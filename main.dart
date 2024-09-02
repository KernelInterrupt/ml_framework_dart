import 'dart:async';
import 'autograd.dart';
import 'tensor.dart';
Node func(Node x,Node y){

  return y.matmul(x);
}
void main(){
var x=Node(Tensor([1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0,10.0,11.0,12.0],[2,3,2]));
var y=Node(Tensor([1.0,2.0,3.0],[3]));
var z=func(x, y);
var a=Tensor([1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0,10.0,11.0,12.0,13.0,14.0,15.0,16.0,17.0,18.0],[2,3,3]);
print(a);
print(a.permute([2,1,0]));
}
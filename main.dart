
import 'autograd.dart';
import 'tensor.dart';
import 'nn.dart' as nn;
Node func(Node x,Node y){

  return x.matmul(y);
}
void main(){
  var x=Node(Tensor([1.0,2.0,3.0,4.0,5.0],[5]));
  var y=Node(Tensor(List.generate(5, (index) => (index*5+3).toDouble()),[5]));
var z=nn.Linear(5,5);
var output=z.forward(x);
var predictY=y-output;
predictY.backward();
Tensor lr1=full(0.001, z.weights.grad.shape);
Tensor lr2=full(0.001, z.bias.grad.shape);
for(int j=0;j<=10;j++){
for(int i=0;i<=20;i++){
z.weights.tensor=z.weights.tensor-lr1*z.weights.grad;
z.bias.tensor=z.bias.tensor-lr2*z.bias.grad;
x=Node(Tensor(List.generate(5, (index) => (index+5*i).toDouble()),[5]));
y=Node(x.tensor*full(5, [5])+full(3,[5]));
print(x.tensor);
print(y.tensor);
output=z.forward(x);
print(output.tensor);
predictY=y-output;
predictY.backward();
print("--------------");




}
}
print(z.weights.tensor);
print(z.bias.tensor);
print(output.tensor);
print(y.tensor);
}
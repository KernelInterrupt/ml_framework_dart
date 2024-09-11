
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
var loss=nn.MSELoss(y, output);
loss.backward();
Tensor lr1=full(0.001, z.weights.grad.shape);
Tensor lr2=full(0.001, z.bias.grad.shape);
for(int j=0;j<=300;j++){
for(int i=0;i<=20;i++){
z.weights.tensor=z.weights.tensor-lr1*z.weights.grad;
z.bias.tensor=z.bias.tensor-lr2*z.bias.grad;
x=Node(Tensor(List.generate(5, (index) => (index+i).toDouble()),[5]));
y=Node(x.tensor*5+3);

output=z.forward(x);
print("-------------");
print(x.tensor);
print(y.tensor);
print(z.weights.grad);
print(z.bias.grad);
print(z.weights.tensor);
print(z.bias.tensor);
print(output.tensor);
print(loss.tensor);
print("--------------");

loss=nn.MSELoss(y, output);
loss.backward();





}
}
x=Node(Tensor([1.0,2.0,3.0,4.0,5.0],[5]));
output=z.forward(x);
print(output.tensor);
}
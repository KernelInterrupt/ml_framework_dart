import 'dart:math' as math;
import 'tensor.dart';
class Node{

Tensor tensor;
String? op;
List<Node>? parents;
Tensor grad=Tensor([0.0],[1]);
Node(this.tensor,{this.op,this.parents});
    
    Node operator +(Node other){

      return add(other);
    }
    Node operator -(Node other){

      return sub(other);
    }
    Node operator *(Node other){
      return mul(other);
     
    }
    Node operator /(Node other){
      return div(other);
    }



Node add(Node other){
if(this.tensor.shape.equal(other.tensor.shape)){
  return Node(this.tensor+other.tensor,op: 'add',parents: [this,other]);
}
else{
  List<int> targetShape=calculateBroadcastedShape(this.tensor, other.tensor);
  Node? broadcastedThis;
  Node? broadcastedOther;
  if(this.tensor.shape.equal(targetShape)){broadcastedThis=this;}else{broadcastedThis=this.broadcast_to(targetShape);}
  if(other.tensor.shape.equal(targetShape)){broadcastedOther=other;}else{broadcastedOther=other.broadcast_to(targetShape);}
  return Node(broadcastedThis.tensor-broadcastedOther.tensor,op: 'sub',parents: [broadcastedThis,broadcastedOther]);
}
}
Node sub(Node other){
if(this.tensor.shape.equal(other.tensor.shape)){
  return Node(this.tensor-other.tensor,op: 'sub',parents: [this,other]);
}
else{
List<int> targetShape=calculateBroadcastedShape(this.tensor, other.tensor);
  Node? broadcastedThis;
  Node? broadcastedOther;
  if(this.tensor.shape.equal(targetShape)){broadcastedThis=this;}else{broadcastedThis=this.broadcast_to(targetShape);}
  if(other.tensor.shape.equal(targetShape)){broadcastedOther=other;}else{broadcastedOther=other.broadcast_to(targetShape);}
  return Node(broadcastedThis.tensor-broadcastedOther.tensor,op: 'sub',parents: [broadcastedThis,broadcastedOther]);
}

}
Node mul(Node other){
   if(this.tensor.shape.equal(other.tensor.shape)){
  return Node(this.tensor*other.tensor,op: 'mul',parents: [this,other]);
}
else{
 List<int> targetShape=calculateBroadcastedShape(this.tensor, other.tensor);
  Node? broadcastedThis;
  Node? broadcastedOther;
  if(this.tensor.shape!=targetShape){broadcastedThis=this.broadcast_to(targetShape);}else{broadcastedThis=this;}
  if(other.tensor.shape!=targetShape){broadcastedOther=other.broadcast_to(targetShape);}else{broadcastedOther=other;}
  return Node(broadcastedThis.tensor*broadcastedOther.tensor,op: 'mul',parents: [broadcastedThis,broadcastedOther]);
}
}
Node div(Node other){
  if(this.tensor.shape.equal(other.tensor.shape)){
  return Node(this.tensor/other.tensor,op: 'div',parents: [this,other]);
}
else{
  List<int> targetShape=calculateBroadcastedShape(this.tensor, other.tensor);
  Node? broadcastedThis;
  Node? broadcastedOther;
  if(this.tensor.shape!=targetShape){broadcastedThis=this.broadcast_to(targetShape);}else{broadcastedThis=this;}
  if(other.tensor.shape!=targetShape){broadcastedOther=other.broadcast_to(targetShape);}else{broadcastedOther=other;}

  return Node(broadcastedThis.tensor/broadcastedOther.tensor,op: 'div',parents: [broadcastedThis,broadcastedOther]);
}
}

Node sin()
{
  return Node(this.tensor.sin(),op: 'sin',parents: [this]);
}

Node cos()
{
  return Node(this.tensor.cos(),op: 'cos',parents: [this]);
}


Node asin()
{
  return Node(this.tensor.asin(),op: 'asin',parents: [this]);
}

Node acos()
{
  return Node(this.tensor.acos(),op: 'acos',parents: [this]);
}


Node exp()
{
   return Node(this.tensor.exp(),op: 'exp',parents: [this]);
}

Node log()
{
  return Node(this.tensor.log(),op: 'log',parents: [this]);
}
Node pow(Node other){
   return Node(this.tensor.pow(other.tensor),op: 'pow',parents: [this,other]);
}
Node broadcast_to(List<int> broadcastedShape){
return Node(this.tensor.broadcastTo(broadcastedShape),op:'broadcast',parents: [this]);

}
//Node sqrt(){
  //return Node(this.tensor.sqrt(),op:'sqrt',parents:[this]);
//}




void backward({Tensor? gradient})
{if(gradient == null){
  gradient=full(1.0,this.tensor.shape);
}
this.grad=gradient;

if(op=='add')
{
parents![0].backward(gradient: gradient);
parents![1].backward(gradient: gradient);

}
else if(op=='sub'){
parents![0].backward(gradient:gradient);
parents![1].backward(gradient:gradient.neg());
}
else if(op=='mul')
{
parents![0].backward(gradient:gradient*parents![1].tensor);
parents![1].backward(gradient:gradient*parents![0].tensor);
}
else if(op=='div')
{
parents![0].backward(gradient:gradient/parents![1].tensor);
parents![1].backward(gradient:(gradient*parents![0].tensor/parents![1].tensor.pow(2)).neg());

}
else if(op=='pow'){
  parents![0].backward(gradient:gradient*(parents![1].tensor)*this.tensor/parents![0].tensor);//this.value/parents![0].value==math.pow(parents![0].value,parents![1].value-1)
  parents![1].backward(gradient:gradient*this.tensor*parents![0].tensor.log());
}
else if(op=='sin'){
  parents![0].backward(gradient:gradient*parents![0].tensor.cos());
}
else if(op=='cos'){
  parents![0].backward(gradient:gradient*parents![0].tensor.sin().neg());

}
else if(op=='asin'){
  parents![0].backward(gradient:gradient/((interOp(1)-parents![0].tensor).sqrt()));
}
else if(op=='acos'){
  parents![0].backward(gradient:(gradient.neg())/((interOp(1)-parents![0].tensor).sqrt()));
}
else if(op=='log'){
  parents![0].backward(gradient:gradient/parents![0].tensor);
}
else if(op=='exp'){
parents![0].backward(gradient:gradient*this.tensor);//this.value==math.exp(parents![0].value)

}
else if(op=='broadcast'){
  parents![0].backward(gradient:gradient.sum(this.tensor.broadcastedAxes!));
}
}

}




extension interOp on num{
 Tensor operator-(Tensor other){
    return full(0.0,other.shape)-other;
  }
}

Node sin(Node a){
  return a.sin();
}

Node cos(Node a){
  return a.cos();
}

Node exp(Node a)
{
  return a.exp();
}
Node log(Node a){
  return a.log();
}

Node asin(Node a){
  return a.asin();
}

Node acos(Node a){
  return a.acos();
}

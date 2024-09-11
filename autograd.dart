
import 'dart:math' as math;
import 'tensor.dart';
class Node{

Tensor tensor;
String? op;
List<Node>? parents;
int? axis;
List<int>? transposedAxes;
Tensor grad=Tensor([0.0],[1]);
Node(this.tensor,{this.op,this.parents,this.axis,this.transposedAxes});
    
    Node operator +(dynamic other){

      return add(other);
    }
    Node operator -(dynamic other){

      return sub(other);
    }
    Node operator *(dynamic other){
      return mul(other);
     
    }
    Node operator /(dynamic other){
      return div(other);
    }



Node add(dynamic other){
  if(other is Node){
if(this.tensor.shape.equal(other.tensor.shape)){
  return Node(this.tensor+other.tensor,op: 'add',parents: [this,other]);
}
else{
  List<int> targetShape=calculateBroadcastedShape(this.tensor, other.tensor);
  Node? broadcastedThis;
  Node? broadcastedOther;
  if(this.tensor.shape.equal(targetShape)){broadcastedThis=this;}else{broadcastedThis=this.broadcast_to(targetShape);}
  if(other.tensor.shape.equal(targetShape)){broadcastedOther=other;}else{broadcastedOther=other.broadcast_to(targetShape);}
  return Node(broadcastedThis.tensor+broadcastedOther.tensor,op: 'add',parents: [broadcastedThis,broadcastedOther]);
}
  }
  else if(other is num){
    Node generatedOther=Node(full(other.toDouble(), this.tensor.shape));
    return this.add(generatedOther);
  }
  else{throw Exception("wrong data type");}
}
Node sub(dynamic other){
if(other is Node){
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
  else if(other is num){
    Node generatedOther=Node(full(other.toDouble(), this.tensor.shape));
    return this.sub(generatedOther);
  }
  else{throw Exception("wrong data type");}

}
Node mul(dynamic other){
   if(other is Node){
if(this.tensor.shape.equal(other.tensor.shape)){
  return Node(this.tensor*other.tensor,op: 'mul',parents: [this,other]);
}
else{
  List<int> targetShape=calculateBroadcastedShape(this.tensor, other.tensor);
  Node? broadcastedThis;
  Node? broadcastedOther;
  if(this.tensor.shape.equal(targetShape)){broadcastedThis=this;}else{broadcastedThis=this.broadcast_to(targetShape);}
  if(other.tensor.shape.equal(targetShape)){broadcastedOther=other;}else{broadcastedOther=other.broadcast_to(targetShape);}
  return Node(broadcastedThis.tensor*broadcastedOther.tensor,op: 'mul',parents: [broadcastedThis,broadcastedOther]);
}
  }
  else if(other is num){
    Node generatedOther=Node(full(other.toDouble(), this.tensor.shape));
    return this.mul(generatedOther);
  }
  else{throw Exception("wrong data type");}
}
Node div(dynamic other){
  
  if(other is Node){
if(this.tensor.shape.equal(other.tensor.shape)){
  return Node(this.tensor/other.tensor,op: 'div',parents: [this,other]);
}
else{
  List<int> targetShape=calculateBroadcastedShape(this.tensor, other.tensor);
  Node? broadcastedThis;
  Node? broadcastedOther;
  if(this.tensor.shape.equal(targetShape)){broadcastedThis=this;}else{broadcastedThis=this.broadcast_to(targetShape);}
  if(other.tensor.shape.equal(targetShape)){broadcastedOther=other;}else{broadcastedOther=other.broadcast_to(targetShape);}
  return Node(broadcastedThis.tensor/broadcastedOther.tensor,op: 'div',parents: [broadcastedThis,broadcastedOther]);
}
  }
  else if(other is num){
    Node generatedOther=Node(full(other.toDouble(), this.tensor.shape));
    return this.div(generatedOther);
  }
  else{throw Exception("wrong data type");}
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
Node pow(dynamic other){

  if(other is Node){
   return Node(this.tensor.pow(other.tensor),op: 'pow',parents: [this,other]);
  }
  else if(other is num)
  {
    Node generatedOther=Node(full(other.toDouble(), this.tensor.shape));
    return this.pow(generatedOther);
  }
  else{throw Exception("wrong data type");}
}
Node broadcast_to(List<int> broadcastedShape){
return Node(this.tensor.broadcastTo(broadcastedShape),op:'broadcast',parents: [this]);

}
Node sigmoid(){

  return Node(this.tensor.sigmoid(),op:'sigmoid',parents:[this]);
}
//Node sqrt(){
  //return Node(this.tensor.sqrt(),op:'sqrt',parents:[this]);
//}
Node matmul(Node other){
  Tensor a=this.tensor.clone();
  Tensor b=other.tensor.clone();
  Node thisTmp=this;
  Node otherTmp=other;
  String isUnsqueezed="None";
if (a.shape.length == 0 || b.shape.length == 0) {
      throw Exception("wrong shape");
    } else if (a.shape.length == 1 && b.shape.length == 1) {
      if (a.shape[0] == b.shape[0]) {
        
      } else {
        throw Exception("wrong shape");
      }
    }
    else if (a.shape.length == 1) {
      if (a.shape[0] != b.shape[b.shape.length - 2]) {
        throw Exception("wrong shape");
      } else {
        thisTmp=this.unsqueeze(0);
       isUnsqueezed="this";
        
      }
    } else if(b.shape.length==1){
      if (a.shape[a.shape.length - 1] != b.shape[0]) {
        throw Exception("wrong shape");
      } else {
        otherTmp=other.unsqueeze(1);
        isUnsqueezed="other";
      }
    }
   
List<List<int>> targetShape=calculateMatmulBroadcastedShape(thisTmp.tensor, otherTmp.tensor);
  Node? broadcastedThis;
  Node? broadcastedOther;
  if(thisTmp.tensor.shape.equal(targetShape[0])){broadcastedThis=thisTmp;}else{broadcastedThis=thisTmp.broadcast_to(targetShape[0]);}
  if(otherTmp.tensor.shape.equal(targetShape[1])){broadcastedOther=otherTmp;}else{broadcastedOther=otherTmp.broadcast_to(targetShape[1]);}
Tensor result=broadcastedThis.tensor.matmul(broadcastedOther.tensor);
if(isUnsqueezed=="this")
{Node resultNode=Node(result,op: 'matmul',parents: [broadcastedThis,broadcastedOther]);
  return resultNode.squeeze(axis: result.shape.length-2);
}
else if(isUnsqueezed=="other"){
  Node resultNode=Node(result,op: 'matmul',parents: [broadcastedThis,broadcastedOther]);
  return resultNode.squeeze(axis: result.shape.length-1);
}
else if(isUnsqueezed=="None"){
  Node resultNode=Node(result,op: 'matmul',parents: [broadcastedThis,broadcastedOther]);
  return resultNode;
}
else{throw Exception("wrong result shape");}
}
Node unsqueeze(int axis){
  return Node(this.tensor.unsqueeze(axis),op:'unsqueeze',parents: [this],axis:axis);
}
Node squeeze({int? axis }){
  return Node(this.tensor.squeeze(axis: axis),op: 'squeeze',parents: [this],axis:axis);
}
Node relu(){
  return Node(this.tensor.relu(),op:'relu',parents: [this]);
}
Node transpose(int axis1,int axis2){
  return Node(this.tensor.transpose(axis1, axis2),op:'transpose',parents: [this],transposedAxes:[axis1,axis2]);
}
Node sum({List<int> axes=const []}){
  return Node(this.tensor.sum(axes: axes),op: 'sum',parents: [this]);

}





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
  parents![0].backward(gradient:gradient.sum(axes:this.tensor.broadcastedAxes!));
}
else if(op=='matmul'){
void calculateLeftGrad(){
Tensor a=gradient!.clone();
  Tensor b=parents![1].tensor.clone();
 
  String isUnsqueezed="None";
if (a.shape.length == 0 || b.shape.length == 0) {
      throw Exception("wrong shape");
    } else if (a.shape.length == 1 && b.shape.length == 1) {
      if (a.shape[0] == b.shape[0]) {
        
      } else {
        throw Exception("wrong shape");
      }
    }
    else if (a.shape.length == 1) {
      if (a.shape[0] != b.shape[b.shape.length - 2]) {
        throw Exception("wrong shape");
      } else {
        a=a.unsqueeze(0);
       isUnsqueezed="this";
        
      }
    } else if(b.shape.length==1){
      if (a.shape[a.shape.length - 1] != b.shape[0]) {
        throw Exception("wrong shape");
      } else {
        b=b.unsqueeze(0);
        isUnsqueezed="other";
      }
    }
    
      b=b.transpose(b.shape.length-2, b.shape.length-1);
    
    
List<List<int>> targetShape=calculateMatmulBroadcastedShape(a, b);
  Tensor? broadcastedA;
  Tensor? broadcastedB;
  if(a.shape!=targetShape){broadcastedA=a.broadcastTo(targetShape[0]);}else{broadcastedA=a;}
  if(b.shape!=targetShape){broadcastedB=b.broadcastTo(targetShape[1]);}else{broadcastedB=b;}
  
Tensor Weightresult=broadcastedA.matmul(broadcastedB);
if(isUnsqueezed=="this")
{
  Weightresult=Weightresult.squeeze(axis: Weightresult.shape.length-2);
  parents![0].backward(gradient: Weightresult);
}
else if(isUnsqueezed=="other"){
   Weightresult=Weightresult.squeeze(axis: Weightresult.shape.length-1);
  parents![0].backward(gradient: Weightresult);
}
else if(isUnsqueezed=="None"){
 parents![0].backward(gradient: Weightresult);
}
else{throw Exception("wrong result shape");}
}

void calculateRightGrad(){



Tensor b=gradient!.clone();
  Tensor a=parents![0].tensor.clone();
 
  String isUnsqueezed="None";
if (a.shape.length == 0 || b.shape.length == 0) {
      throw Exception("wrong shape");
    } else if (a.shape.length == 1 && b.shape.length == 1) {
      if (a.shape[0] == b.shape[0]) {
        
      } else {
        throw Exception("wrong shape");
      }
    }
     else if (a.shape.length == 1) {
      if (a.shape[0] != b.shape[b.shape.length - 2]) {
        throw Exception("wrong shape");
      } else {
        a=a.unsqueeze(0);
       isUnsqueezed="this";
        
      }
    } else if(b.shape.length==1){
      if (a.shape[a.shape.length - 1] != b.shape[0]) {
        throw Exception("wrong shape");
      } else {
        b=b.unsqueeze(1);
        isUnsqueezed="other";
      }
    }
    
      a=a.transpose(a.shape.length-2, a.shape.length-1);
    
    
List<List<int>> targetShape=calculateMatmulBroadcastedShape(a, b);
  Tensor? broadcastedA;
  Tensor? broadcastedB;
  if(a.shape!=targetShape){broadcastedA=a.broadcastTo(targetShape[0]);}else{broadcastedA=a;}
  if(b.shape!=targetShape){broadcastedB=b.broadcastTo(targetShape[1]);}else{broadcastedB=b;}
Tensor Xresult=broadcastedA.matmul(broadcastedB);
if(isUnsqueezed=="this")
{
  Xresult=Xresult.squeeze(axis: Xresult.shape.length-1);
  parents![1].backward(gradient: Xresult);
}
else if(isUnsqueezed=="other"){
   Xresult=Xresult.squeeze(axis: Xresult.shape.length-2);
  parents![1].backward(gradient: Xresult);
}
else if(isUnsqueezed=="None"){
 parents![1].backward(gradient: Xresult);
}
else{throw Exception("wrong result shape");}

}

calculateLeftGrad();
calculateRightGrad();


}
else if(op=='unsqueeze')
{
parents![0].backward(gradient: gradient.squeeze(axis:this.axis));

}
else if(op=='squeeze'){
parents![0].backward(gradient: gradient.unsqueeze(this.axis!));

}

else if(op=='sigmoid')
{
parents![0].backward(gradient: gradient*(full(1.0,gradient.shape)-gradient));

}
else if (op=='transpose')
{
  parents![0].backward(gradient: gradient.transpose(transposedAxes![1], transposedAxes![0]));
}
else if(op=='sum'){
 parents![0].backward(gradient: gradient.broadcastTo(parents![0].tensor.shape));

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




import 'dart:math' as math;
class Tensor{
List<double> data;
List<int> shape;
late List<int> strides;
List<int>? broadcastedAxes;
Tensor(this.data,this.shape){
  int _shapeProduct =
        shape.fold(1, (previousValue, element) => previousValue * element);
        if(_shapeProduct==data.length){
          strides=_computeStrides(shape);
        }
        else{
          throw Exception("wrong data");
        }
        


}

List<int> _computeStrides(List<int> shape) {
    List<int> strides = List.filled(shape.length, 1);
    for (int i = shape.length - 2; i >= 0; i--) {
      strides[i] = strides[i + 1] * shape[i + 1];
    }
    return strides;
  }
  int get size => data.length;
  Tensor operator +(Tensor other){

      return add(other);
    }
    Tensor operator -(Tensor other){

      return sub(other);
    }
    Tensor operator *(Tensor other){
      return mul(other);
     
    }
    Tensor operator /(Tensor other){
      return div(other);
    }
    Tensor operator [](int num){
      return index(num);
    }

Tensor index(int num){
  int startPos=num*strides[0];
  int endPos=startPos+strides[0];
  if(startPos>=endPos||startPos>=this.data.length||endPos>this.data.length)
  {throw Exception("out of range.More detail:start position is:$startPos and end position is:$endPos");}
  else{if(this.shape.length!=1){
    int shapeStartPos=1;
    int shapeEndPos=this.shape.length;
    return Tensor(this.data.sublist(startPos,endPos),this.shape.sublist(shapeStartPos,shapeEndPos));
  }
  else{
  
  return Tensor(this.data.sublist(startPos,endPos),[1]);}
  
  }
}

Tensor add(dynamic other)
{if(other is Tensor){
  if(this.shape.equal(other.shape)){
List<double> resultData = List.generate(size, (i) => data[i] + other.data[i]);
return Tensor(resultData,this.shape);
  }
  else{throw Exception("wrong shape");}
}
else if(other is num)
{
  return this.add(full(other.toDouble(),this.shape));
}
else{throw Exception("wrong type");}
}
Tensor sub(dynamic other)
{
  if(other is Tensor){
  if(this.shape.equal(other.shape)){
List<double> resultData = List.generate(size, (i) => data[i] - other.data[i]);
return Tensor(resultData,this.shape);
  }
  else{throw Exception("wrong shape");}
}
else if(other is num)
{
  return this.sub(full(other.toDouble(),this.shape));
}
else{throw Exception("wrong type");}
}
Tensor mul(dynamic other)
{
  if(other is Tensor){
  if(this.shape.equal(other.shape)){
List<double> resultData = List.generate(size, (i) => data[i] * other.data[i]);
return Tensor(resultData,this.shape);
  }
  else{throw Exception("wrong shape");}
}
else if(other is num)
{
  return this.mul(full(other.toDouble(),this.shape));
}
else{throw Exception("wrong type");}
}
Tensor div(dynamic other)
{
 if(other is Tensor){
  if(this.shape.equal(other.shape)){
List<double> resultData = List.generate(size, (i) => data[i] / other.data[i]);
return Tensor(resultData,this.shape);
  }
  else{throw Exception("wrong shape");}
}
else if(other is num)
{
  return this.div(full(other.toDouble(),this.shape));
}
else{throw Exception("wrong type");}
}
Tensor sin(){
List<double> resultData = List.generate(size, (i) => math.sin(data[i]) );
return Tensor(resultData,this.shape);
}
Tensor cos(){
List<double> resultData = List.generate(size, (i) => math.cos(data[i]) );
return Tensor(resultData,this.shape);
}
Tensor asin(){
List<double> resultData = List.generate(size, (i) => math.asin(data[i]) );
return Tensor(resultData,this.shape);
}
Tensor acos(){
List<double> resultData = List.generate(size, (i) => math.acos(data[i]) );
return Tensor(resultData,this.shape);
}
Tensor exp(){
List<double> resultData = List.generate(size, (i) => math.exp(data[i]) );
return Tensor(resultData,this.shape);
}
Tensor log(){
List<double> resultData = List.generate(size, (i) => math.log(data[i]) );
return Tensor(resultData,this.shape);
}

Tensor sqrt(){
List<double> resultData = List.generate(size, (i) => math.sqrt(data[i]) );
return Tensor(resultData,this.shape);
}
Tensor pow(dynamic other){
  if(other is Tensor){
if(this.shape.equal(other.shape)){
List<double> resultData = List.generate(size, (i) => math.pow(data[i] , other.data[i]).toDouble());
return Tensor(resultData,this.shape);
  }

  else{throw Exception("wrong shape");}
  }
  else if(other is num)
  {
    return this.pow(full(other.toDouble(), this.shape));
  }
  else throw Exception("wrong type");
}


Tensor neg(){
  List<double> resultData = List.generate(size, (i) => -data[i] );
return Tensor(resultData,this.shape);
}


bool broadcastable(List<int> broadcastedShape){
int minLength=this.shape.length<=broadcastedShape.length?this.shape.length:broadcastedShape.length;

for(int i=minLength-1;i>=0;i--){
  if(this.shape[i]==broadcastedShape[i]||(this.shape[i]==1||broadcastedShape[i]==1)){}
  else{return false;}

}
return true;

}



Tensor repeat(int repeatNum,{int axis=0}){

  if(axis==0){
    List<double> result=[];
    if(axis==this.shape.length-1){
      
      for(int i=0;i<this.shape[this.shape.length-1];i++)
      {
        for(int j=0;j<repeatNum;j++)
        {
          result.add(this.data[i]);
        }
      }
      return Tensor(result, [result.length]);
    }
    else{
      for(int i=0;i<this.shape[0];i++){
      for(int j=0;j<repeatNum;j++)
        {
          result.addAll(this[i].data);
        }
    }
   
    int shapeStartPos=1;
    int shapeEndPos=this.shape.length;
    List<int> outputShape=this.shape.sublist(shapeStartPos,shapeEndPos);
    outputShape.insert(0,this.shape[0]*repeatNum);
    return Tensor(result,outputShape);
  
 
    }
  }
  else{
Tensor resultTensor=this[0].repeat(repeatNum,axis:axis-1).unsqueeze(0);
    for(int i=1;i<this.shape[0];i++){
      Tensor other=this[i].repeat(repeatNum,axis:axis-1).unsqueeze(0);
      resultTensor=resultTensor.append(other);
    }
return resultTensor;
  }
}

Tensor append(Tensor other)
{
List<double> result=this.data;
result.addAll(other.data);
List<int> outputShape=this.shape;
if(this.shape.sublist(1,this.shape.length).equal(other.shape.sublist(1,this.shape.length))){
outputShape[0]=outputShape[0]+other.shape[0];
return Tensor(result, outputShape);}
else{throw Exception("wrong shape");}


}

Tensor unsqueeze(int axis){

  List<int> outputShape=this.shape;
  outputShape.insert(axis,1);
  return Tensor(this.data,outputShape);
}


Tensor broadcastTo(List<int> broadcastedShape){
if(this.broadcastable(broadcastedShape)){
if(this.shape.length<broadcastedShape.length){throw Exception("can't broadcast to destined shape");}
else{
  
  Tensor thisTmp=this;
  List<int> bcAxes=[];
//this.shape.length>=broadcastedShape.length
 
  for(int i=this.shape.length-1;i>=0;i--){
    if((this.shape[i]!=broadcastedShape[i])&&(this.shape[i]!=1&&broadcastedShape[i]!=1)){
      throw Exception("can't broadcast to destined shape");
    }
    else if(this.shape[i]==broadcastedShape[i]){}
  else if(this.shape[i]==1){
thisTmp=this.repeat(broadcastedShape[i],axis: i);
bcAxes.add(i);
  }
  
  else{throw Exception("Unknown error");}
  }
thisTmp.broadcastedAxes=bcAxes;
  return thisTmp;
}
}
else{throw Exception("can't broadcast to destined shape");}

}



@override
String toString(){

  List<dynamic> output=this.toList();
  return output.toString();
}

    List<dynamic> toList() {
    
    
    List<int> tensorShape = this.shape;
   
     
      final List<double> flatList = this.data;

      

      List<dynamic> buildList(int dimension, int offset) {
        if (dimension == tensorShape.length - 1) {
          return flatList.sublist(offset, offset + tensorShape[dimension]);
        }

        List<dynamic> result = [];
        for (int i = 0; i < tensorShape[dimension]; i++) {
          var sublist =
              buildList(dimension + 1, offset + i * strides[dimension]);
          
            result.add(sublist);
          
        }
        return result;
      }

      return buildList(0, 0);
}
}
Tensor createTensor(dynamic list) {
  List<num> flatList = [];
  List<int> sizes = [];
  List<bool> isFirstElementAtDepth = [];

  void flatten(dynamic element, int depth) {
    if (element is List) {
      if (isFirstElementAtDepth.length <= depth ||
          isFirstElementAtDepth[depth]) {
        // 如果是首次进入此深度的列表
        if (sizes.length <= depth) {
          sizes.add(element.length); // 添加新尺寸
        } else {
          sizes[depth] = element.length; // 更新此深度的尺寸
        }
        // 扩展或更新首元素标记列表
        if (isFirstElementAtDepth.length <= depth) {
          isFirstElementAtDepth.add(false); // 添加新的深度标记
        } else {
          isFirstElementAtDepth[depth] = false; // 更新此深度的首元素标记
        }
      }

      for (var subElement in element) {
        flatten(subElement, depth + 1);
      }

      // 退出当前列表深度时，重置此深度的首元素标记
      if (isFirstElementAtDepth.length > depth) {
        isFirstElementAtDepth[depth] = true;
      }
    } else if (element is num) {
      flatList.add(element);
    }
  }

  flatten(list, 0);
Tensor outputTensor =
      Tensor(flatList.cast<double>(), sizes);
      return outputTensor;
}


Tensor full(double value,List<int> shape){
int shapeProduct =
        shape.fold(1, (previousValue, element) => previousValue * element);
        List<double> resultList = List.generate(shapeProduct, (index) => value);
        return Tensor(resultList, shape);

}

extension on List<int> {
  bool equal(List<int> other) {
    if (identical(this, other)) return true;
    
      if (length != other.length) return false;
      for (int i = 0; i < length; i++) {
        if (this[i] != other[i]) return false;
      }
      return true;
    
   
  }
}

List<Tensor> broadcast(Tensor a,Tensor b){
if(a.broadcastable(b.shape)){
if(a.shape.length<b.shape.length){return broadcast(b,a);}
else{
  
  Tensor aTmp=a;
  Tensor bTmp=b;
//a.shape.length>=b.shape.length
  for(int i=0;i<a.shape.length-b.shape.length;i++){
    bTmp=bTmp.unsqueeze(0);
  }
  for(int i=a.shape.length-1;i>=0;i--){
    if((a.shape[i]!=b.shape[i])&&(a.shape[i]!=1&&b.shape[i]!=1)){
      throw Exception("can't broadcast to destined shape");
    }
    else if(a.shape[i]==b.shape[i]){}
  else if(a.shape[i]==1){
aTmp=aTmp.repeat(b.shape[i],axis: i);
  }
  else if(b.shape[i]==1){
bTmp=bTmp.repeat(a.shape[i],axis: i);
  }
  else{throw Exception("Unknown error");}
  }

  return [aTmp,bTmp];
}
}
else{throw Exception("can't broadcast to destined shape");}

}

List<int> calculateBroadcastedShape(Tensor a,Tensor b){
if(a.broadcastable(b.shape)){
if(a.shape.length<b.shape.length){return calculateBroadcastedShape(b,a);}
else{
  
  List<int> broadcastedShape=a.shape;
//a.shape.length>=b.shape.length
  
  for(int i=a.shape.length-1;i>=0;i--){
    if((a.shape[i]!=b.shape[i])&&(a.shape[i]!=1&&b.shape[i]!=1)){
      throw Exception("can't broadcast to destined shape");
    }
    else if(a.shape[i]==b.shape[i]){}
  else if(a.shape[i]==1){
broadcastedShape[i]=b.shape[i];
  }
  else if(b.shape[i]==1){
broadcastedShape[i]=a.shape[i];
  }
  else{throw Exception("Unknown error");}
  }

  return broadcastedShape;
}
}
else{throw Exception("can't broadcast to destined shape");}

}
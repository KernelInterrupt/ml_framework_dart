import 'dart:math' as math;

class Tensor {
  List<double> data;
  List<int> shape;
  late List<int> strides;
  List<int>? broadcastedAxes;
  Tensor(this.data, this.shape) {
    int _shapeProduct =
        shape.fold(1, (previousValue, element) => previousValue * element);
    if (_shapeProduct == data.length) {
      strides = _computeStrides(shape);
    } else {
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
  Tensor operator +(Tensor other) {
    return add(other);
  }

  Tensor operator -(Tensor other) {
    return sub(other);
  }

  Tensor operator *(Tensor other) {
    return mul(other);
  }

  Tensor operator /(Tensor other) {
    return div(other);
  }

  dynamic operator [](int num) {
    return index(num);
  }

  Tensor index(int num) {
    int startPos = num * strides[0];
    int endPos = startPos + strides[0];
    if (startPos >= endPos ||
        startPos >= this.data.length ||
        endPos > this.data.length) {
      throw Exception(
          "out of range.More detail:start position is:$startPos and end position is:$endPos");
    } else {
      if (this.shape.length != 1) {
        int shapeStartPos = 1;
        int shapeEndPos = this.shape.length;
        return Tensor(this.data.sublist(startPos, endPos),
            this.shape.sublist(shapeStartPos, shapeEndPos));
      } else {
        return Tensor([this.data[startPos]], []);
      }
    }
  }

  Tensor add(dynamic other) {
    if (other is Tensor) {
      if (this.shape.equal(other.shape)) {
        List<double> resultData =
            List.generate(size, (i) => data[i] + other.data[i]);
        return Tensor(resultData, this.shape);
      } else {
        throw Exception("wrong shape");
      }
    } else if (other is num) {
      return this.add(full(other.toDouble(), this.shape));
    } else {
      throw Exception("wrong type");
    }
  }

  Tensor sub(dynamic other) {
    if (other is Tensor) {
      if (this.shape.equal(other.shape)) {
        List<double> resultData =
            List.generate(size, (i) => data[i] - other.data[i]);
        return Tensor(resultData, this.shape);
      } else {
        throw Exception("wrong shape");
      }
    } else if (other is num) {
      return this.sub(full(other.toDouble(), this.shape));
    } else {
      throw Exception("wrong type");
    }
  }

  Tensor mul(dynamic other) {
    if (other is Tensor) {
      if (this.shape.equal(other.shape)) {
        List<double> resultData =
            List.generate(size, (i) => data[i] * other.data[i]);
        return Tensor(resultData, this.shape);
      } else {
        throw Exception("wrong shape");
      }
    } else if (other is num) {
      return this.mul(full(other.toDouble(), this.shape));
    } else {
      throw Exception("wrong type");
    }
  }

  Tensor div(dynamic other) {
    if (other is Tensor) {
      if (this.shape.equal(other.shape)) {
        List<double> resultData =
            List.generate(size, (i) => data[i] / other.data[i]);
        return Tensor(resultData, this.shape);
      } else {
        throw Exception("wrong shape");
      }
    } else if (other is num) {
      return this.div(full(other.toDouble(), this.shape));
    } else {
      throw Exception("wrong type");
    }
  }

  Tensor sin() {
    List<double> resultData = List.generate(size, (i) => math.sin(data[i]));
    return Tensor(resultData, this.shape);
  }

  Tensor cos() {
    List<double> resultData = List.generate(size, (i) => math.cos(data[i]));
    return Tensor(resultData, this.shape);
  }

  Tensor asin() {
    List<double> resultData = List.generate(size, (i) => math.asin(data[i]));
    return Tensor(resultData, this.shape);
  }

  Tensor acos() {
    List<double> resultData = List.generate(size, (i) => math.acos(data[i]));
    return Tensor(resultData, this.shape);
  }

  Tensor exp() {
    List<double> resultData = List.generate(size, (i) => math.exp(data[i]));
    return Tensor(resultData, this.shape);
  }

  Tensor log() {
    List<double> resultData = List.generate(size, (i) => math.log(data[i]));
    return Tensor(resultData, this.shape);
  }

  Tensor sqrt() {
    List<double> resultData = List.generate(size, (i) => math.sqrt(data[i]));
    return Tensor(resultData, this.shape);
  }

  Tensor pow(dynamic other) {
    if (other is Tensor) {
      if (this.shape.equal(other.shape)) {
        List<double> resultData = List.generate(
            size, (i) => math.pow(data[i], other.data[i]).toDouble());
        return Tensor(resultData, this.shape);
      } else {
        throw Exception("wrong shape");
      }
    } else if (other is num) {
      return this.pow(full(other.toDouble(), this.shape));
    } else
      throw Exception("wrong type");
  }

  Tensor neg() {
    List<double> resultData = List.generate(size, (i) => -data[i]);
    return Tensor(resultData, this.shape);
  }

  Tensor matmul(Tensor other) {
    if(this.shape.length>2&&other.shape.length>2)//大于2维，默认已经经历过广播操作
    {
if(this.shape.sublist(0,this.shape.length-2).equal(other.shape.sublist(0,other.shape.length-2))==0){
  throw Exception("wrong shape");
}
else{
  Tensor aTmp=this.clone();
  Tensor bTmp=other.clone();
  Tensor result=(aTmp[0].matmul(bTmp[0])).unsqueeze(0);
for(int i=1;i<aTmp.shape[0];i++)
{
result=result.append(aTmp[i].matmul(bTmp[i]).unsqueeze(0));
}
return result;
}

    }
    if (this.shape.length == 2 && other.shape.length == 2) {
      if (this.shape[1] == other.shape[0]) {
        List<double> resultData =
            List.filled(this.shape[0] * other.shape[1], 0);
        for (int i = 0; i < this.shape[0]; i++) {
          for (int j = 0; j < other.shape[1]; j++) {
            for (int k = 0; k < this.shape[1]; k++) {
              resultData[i * other.shape[1] + j] +=
                  this.data[i * this.shape[1] + k] *
                      other.data[j + k * other.shape[1]];
            }
          }
        }
        return Tensor(resultData, [this.shape[0], other.shape[1]]);
      } else {
        throw Exception("wrong shape");
      }
    } else if (this.shape.length == 1 && other.shape.length == 1) {
      if (this.shape[0] == other.shape[0]) {
        double resultData = 0;
        for (int i = 0; i < this.shape[0]; i++) {
          resultData += this.data[i] * other.data[i];
        }
        return Tensor([resultData], []);
      } else {
        throw Exception("wrong shape");
      }
    } else if (this.shape.length == 1 && other.shape.length == 2) {
      if (this.shape[0] == other.shape[0]) {
        List<double> resultData = List.filled(other.shape[1], 0);
        for (int i = 0; i < other.shape[1]; i++) {
          for (int j = 0; j < other.shape[0]; j++) {
            resultData[i] += this.data[j] * other.data[j * other.shape[1] + i];
          }
        }
        return Tensor(resultData, [other.shape[1]]);
      } else {
        throw Exception("wrong shape");
      }
    } else {
      throw Exception("wrong shape");
    }
  }


Tensor sigmoid(){

  return full(1, this.shape)/full(1, this.shape)+this.neg().exp();
}

Tensor relu(){
  return max(full(0.0,this.shape));
}

Tensor max(Tensor other){
if(this.data.length==other.data.length)
{
return Tensor(List.generate(this.data.length, (i) => this.data[i] > other.data[i] ? this.data[i] : other.data[i]),this.shape);
}
else{throw Exception("wrong shape");}
}


  bool broadcastable(List<int> broadcastedShape) {
    int minLength = this.shape.length <= broadcastedShape.length
        ? this.shape.length
        : broadcastedShape.length;

    for (int i = minLength - 1; i >= 0; i--) {
      if (this.shape[i] == broadcastedShape[i] ||
          (this.shape[i] == 1 || broadcastedShape[i] == 1)) {
      } else {
        return false;
      }
    }
    return true;
  }

  bool matmul_broadcastable(List<int> broadcastedShape) {
    if (this.shape.length == 0 && broadcastedShape.length == 0) {
      return false;
    } else if (this.shape.length == 1 && broadcastedShape.length == 1) {
      if (this.shape[0] == broadcastedShape[0]) {
        return true;
      } else {
        return false;
      }
    } else if (this.shape.length >= 2 && broadcastedShape.length >= 2) {
      if (this.shape[this.shape.length - 1] !=
          broadcastedShape[broadcastedShape.length - 2]) {
        return false;
      }
      int minLength = this.shape.length <= broadcastedShape.length
          ? this.shape.length
          : broadcastedShape.length;

      for (int i = minLength - 1 - 2; i >= 0; i--) {
        if (this.shape[i] == broadcastedShape[i] ||
            (this.shape[i] == 1 || broadcastedShape[i] == 1)) {
        } else {
          return false;
        }
      }
      return true;
    } else if (this.shape.length == 1) {
      if (this.shape[0] != broadcastedShape[broadcastedShape.length - 2]) {
        return false;
      } else {
        return true;
      }
    } else {
      if (this.shape[this.shape.length - 1] != broadcastedShape[0]) {
        return false;
      } else {
        return true;
      }
    }
  }

  Tensor repeat(int repeatNum, {int axis = 0}) {
    if (axis == 0) {
      List<double> result = [];
      if (axis == this.shape.length - 1) {
        for (int i = 0; i < this.shape[this.shape.length - 1]; i++) {
          for (int j = 0; j < repeatNum; j++) {
            result.add(this.data[i]);
          }
        }
        return Tensor(result, [result.length]);
      } else {
        for (int i = 0; i < this.shape[0]; i++) {
          for (int j = 0; j < repeatNum; j++) {
            result.addAll(this[i].data);
          }
        }

        int shapeStartPos = 1;
        int shapeEndPos = this.shape.length;
        List<int> outputShape = this.shape.sublist(shapeStartPos, shapeEndPos);
        outputShape.insert(0, this.shape[0] * repeatNum);
        return Tensor(result, outputShape);
      }
    } else {
      Tensor resultTensor =
          this[0].repeat(repeatNum, axis: axis - 1).unsqueeze(0);
      for (int i = 1; i < this.shape[0]; i++) {
        Tensor other = this[i].repeat(repeatNum, axis: axis - 1).unsqueeze(0);
        resultTensor = resultTensor.append(other);
      }
      return resultTensor;
    }
  }

  Tensor append(Tensor other) {
    List<double> result = List.from(this.data);
    result.addAll(other.data);
    List<int> outputShape = List.from(this.shape);
    if (this.shape.length >= 1) {
      if (this
          .shape
          .sublist(1, this.shape.length)
          .equal(other.shape.sublist(1, this.shape.length))) {
        outputShape[0] = outputShape[0] + other.shape[0];
        return Tensor(result, outputShape);
      } else {
        throw Exception("wrong shape");
      }
    } else if (this.shape.length == 0) {
      return Tensor(result, [2]);
    } else {
      throw Exception("wrong shape");
    }
  }

  Tensor unsqueeze(int axis) {
    List<int> outputShape = List.from(this.shape);
    outputShape.insert(axis, 1);
    return Tensor(this.data, outputShape);
  }

  Tensor sumSingleAxis(int axis, {bool KeepDim = false}) {
    if (KeepDim == false) {
      if (axis == 0) {
        Tensor outputTensor = this[0].clone();
        for (int i = 1; i < this.shape[0]; i++) {
          outputTensor = outputTensor + this[i];
        }
        return outputTensor;
      } else if (axis >= 1) {
        Tensor outputTensor = this[0].clone().sumSingleAxis(axis - 1);

        outputTensor = outputTensor.unsqueeze(0);
        for (int i = 1; i < this.shape[0]; i++) {
          outputTensor =
              outputTensor.append(this[i].sumSingleAxis(axis - 1).unsqueeze(0));
        }
        return outputTensor;
      } else {
        throw Exception("axis out of bound");
      }
    } else if (KeepDim == true) {
      if (axis == 0) {
        Tensor outputTensor = this[0].clone();
        for (int i = 1; i < this.shape[0]; i++) {
          outputTensor = outputTensor + this[i];
        }
        return outputTensor.unsqueeze(0);
      } else if (axis >= 1) {
        Tensor outputTensor =
            this[0].clone().sumSingleAxis(axis - 1, KeepDim: KeepDim).unsqueeze(0);
        for (int i = 1; i < this.shape[0]; i++) {
          outputTensor = outputTensor.append(
              this[i].clone().sumSingleAxis(axis - 1, KeepDim: KeepDim).unsqueeze(0));
        }
        return outputTensor;
      } else {
        throw Exception("axis out of bound");
      }
    } else {
      throw Exception("unknown error");
    }
  }

  Tensor sum(List<int> axes, {bool KeepDim = false}) {
    Tensor tmpTensor = this.clone().sumSingleAxis(axes[0], KeepDim: KeepDim);
    Tensor? outputTensor;
    if (KeepDim == false) {
      int count = 1;
      for (int i = 1; i < axes.length - 1; i++) {
        tmpTensor = tmpTensor.sumSingleAxis(axes[i] - count, KeepDim: KeepDim);
        count++;
      }
      if (axes.length > 1) {
        outputTensor = tmpTensor.sumSingleAxis(axes[axes.length - 1] - count,
            KeepDim: KeepDim);
      } else {
        outputTensor = tmpTensor;
      }
      return outputTensor;
    } else if (KeepDim == true) {
      for (int i = 1; i < axes.length; i++) {
        tmpTensor = tmpTensor.sumSingleAxis(axes[i], KeepDim: KeepDim);
      }
      return tmpTensor;
    } else {
      throw Exception("wrong parameters");
    }
  }

  Tensor broadcastTo(List<int> broadcastedShape) {
    if (this.broadcastable(broadcastedShape)) {
      Tensor thisTmp = this;
      List<int> bcAxes = [];
      if (this.shape.length <= broadcastedShape.length) {
        int count = 0;
        for (int i = 0; i < broadcastedShape.length - this.shape.length; i++) {
          thisTmp = thisTmp.unsqueeze(0);
          count++;
        }
        for (int i = 0; i < count; i++) {
          bcAxes.add(i);
        }
      } else if (this.shape.length > broadcastedShape.length) {
        throw Exception("can't broadcast to destined shape");
      } else {
        throw Exception("can't broadcast to destined shape");
      }

//this.shape.length>=broadcastedShape.length

      for (int i = thisTmp.shape.length - 1; i >= 0; i--) {
        if ((thisTmp.shape[i] != broadcastedShape[i]) &&
            (thisTmp.shape[i] != 1 && broadcastedShape[i] != 1)) {
          throw Exception("can't broadcast to destined shape");
        } else if (thisTmp.shape[i] == broadcastedShape[i]) {
        } else if (thisTmp.shape[i] == 1) {
          thisTmp = thisTmp.repeat(broadcastedShape[i], axis: i);
          bcAxes.addUniqueElement(i);
        } else {
          throw Exception("Unknown error");
        }
      }
      thisTmp.broadcastedAxes = bcAxes;
      return thisTmp;
    } else {
      throw Exception("can't broadcast to destined shape");
    }
  }

Tensor clone(){

  List<double> resultData=List.from(this.data);
  List<int> resultShape=List.from(this.shape);
  return Tensor(resultData,resultShape);
}



Tensor squeeze( {int? axis}) {
  List<int> newShape = [];
  if (axis != null) {
    // Squeeze a specific axis
    for (int i = 0; i < this.shape.length; i++) {
      if (i == axis && this.shape[i] != 1) {
        throw Exception("Cannot squeeze non-unit axis");
      }
      if (i != axis) {
        newShape.add(this.shape[i]);
      }
    }
  } else {
    // Squeeze all unit axes
    for (int dim in this.shape) {
      if (dim != 1) {
        newShape.add(dim);
      }
    }
  }
  return Tensor(this.data, newShape);
}

Tensor transpose(int axis1,int axis2){
List<int> Shape=List.generate(this.shape.length, (index) => index);
Shape[axis1]=axis2;
Shape[axis2]=axis1;

return permute(Shape);



}

 Tensor permute(List<int> perm) {
    // Step 1: 创建新形状
    List<int> newShape = List.generate(perm.length, (i) => shape[perm[i]]);
    List<int> newStrides=_computeStrides(newShape);
   

    // Step 3: 创建新数据列表，填充数据
    List<double> newData = List.filled(data.length, 0.0);
    int rank = shape.length;

    // 遍历新数据列表
   for (int i = 0; i < newData.length; i++) {
    int newIndex = 0;
    int remainingIndex = i;

    // 根据 perm 的顺序计算原始数据的索引
    for (int j = 0; j < rank; j++) {
        
        int currentIndex = remainingIndex ~/strides[j];
        remainingIndex %= strides[j];

        // 将这个索引映射到原始张量的顺序
        newIndex += currentIndex * newStrides[perm[j]];
        
       
    }
   
    // 将原始张量中的值赋值到新的数据列表
    newData[newIndex] = data[i];
}
    return Tensor(newData, newShape);
  }






  @override
  String toString() {
    dynamic output = this.toList();
    return output.toString();
  }

  dynamic toList() {
    List<int> tensorShape = this.shape;

    final List<double> flatList = this.data;
    if (tensorShape.length == 0) {
      return this.data[0];
    }
    List<dynamic> buildList(int dimension, int offset) {
      if (dimension == tensorShape.length - 1) {
        return flatList.sublist(offset, offset + tensorShape[dimension]);
      }

      List<dynamic> result = [];
      for (int i = 0; i < tensorShape[dimension]; i++) {
        var sublist = buildList(dimension + 1, offset + i * strides[dimension]);

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
  Tensor outputTensor = Tensor(flatList.cast<double>(), sizes);
  return outputTensor;
}

Tensor full(double value, List<int> shape) {
  int shapeProduct =
      shape.fold(1, (previousValue, element) => previousValue * element);
  List<double> resultList = List.generate(shapeProduct, (index) => value);
  return Tensor(resultList, shape);
}

extension Equal on List<int> {
  bool equal(List<int> other) {
    if (identical(this, other)) return true;

    if (length != other.length) return false;
    for (int i = 0; i < length; i++) {
      if (this[i] != other[i]) return false;
    }
    return true;
  }
}

List<Tensor> broadcast(Tensor a, Tensor b) {
  if (a.broadcastable(b.shape)) {
    if (a.shape.length < b.shape.length) {
      return broadcast(b, a);
    } else {
      Tensor aTmp = a.clone();
      Tensor bTmp = b.clone();
//a.shape.length>=b.shape.length
      for (int i = 0; i < a.shape.length - b.shape.length; i++) {
        bTmp = bTmp.unsqueeze(0);
      }
      for (int i = a.shape.length - 1; i >= 0; i--) {
        if ((a.shape[i] != b.shape[i]) &&
            (a.shape[i] != 1 && b.shape[i] != 1)) {
          throw Exception("can't broadcast to destined shape");
        } else if (a.shape[i] == b.shape[i]) {
        } else if (a.shape[i] == 1) {
          aTmp = aTmp.repeat(b.shape[i], axis: i);
        } else if (b.shape[i] == 1) {
          bTmp = bTmp.repeat(a.shape[i], axis: i);
        } else {
          throw Exception("Unknown error");
        }
      }

      return [aTmp, bTmp];
    }
  } else {
    throw Exception("can't broadcast to destined shape");
  }
}

List<int> calculateBroadcastedShape(Tensor a, Tensor b) {
  if (a.broadcastable(b.shape)) {
    if (a.shape.length < b.shape.length) {
      return calculateBroadcastedShape(b, a);
    } else {
      List<int> broadcastedShape = List.from(a.shape);
      Tensor? aTmp;
      Tensor? bTmp;
//a.shape.length>=b.shape.length
      if (a.shape.length > 0) {
        aTmp = a.clone();
      } else if (a.shape.length == 0) {
        aTmp = a.clone().unsqueeze(0);
      } else {
        throw Exception("unknown error");
      }
      if (b.shape.length > 0) {
        bTmp = b.clone();
      } else if (b.shape.length == 0) {
        bTmp = b.clone().unsqueeze(0);
      } else {
        throw Exception("unknown error");
      }
       int offset = broadcastedShape.length - bTmp.shape.length;

  for (int i = broadcastedShape.length - 1; i >= 0; i--) {
    int bIndex = i - offset;

    if (bIndex >= 0 && aTmp.shape[i] != bTmp.shape[bIndex] && aTmp.shape[i] != 1 && bTmp.shape[bIndex] != 1) {
      throw Exception("Can't broadcast to destined shape");
    }

    if (bIndex >= 0) {
      broadcastedShape[i] = math.max(aTmp.shape[i], bTmp.shape[bIndex]);
    }
  }

      return broadcastedShape;
    }
  } else {
    throw Exception("can't broadcast to destined shape");
  }
}




List<List<int>> calculateMatmulBroadcastedShape(Tensor a, Tensor b){
if (a.shape.length == 1 && b.shape.length == 1) {
      if (a.shape[0] == b.shape[0]) {
        return[List.from(a.shape),List.from(b.shape)];
      } else {
        throw Exception("wrong shape");
      }
}

    else if (a.shape.length >= 2 && b.shape.length >= 2) {
      Tensor aTmp=a.clone();
      Tensor bTmp=b.clone();
      if (a.shape.length < b.shape.length) {
      for (int i = 0; i < b.shape.length - a.shape.length; i++) {
        aTmp = aTmp.unsqueeze(0);
      }
    } 
      List<int> broadcastedShape = List.from(aTmp.shape);
      
    
      if (aTmp.shape[aTmp.shape.length - 1] !=
          bTmp.shape[bTmp.shape.length - 2]) {
        throw Exception("wrong shape");
      }
    
int offset = broadcastedShape.length - bTmp.shape.length;
      for (int i = broadcastedShape.length - 1 - 2; i >= 0; i--) {
        int bIndex = i - offset;

    if (bIndex >= 0 && aTmp.shape[i] != bTmp.shape[bIndex] && aTmp.shape[i] != 1 && bTmp.shape[bIndex] != 1) {
      throw Exception("Can't broadcast to destined shape");
    }

    if (bIndex >= 0) {
      broadcastedShape[i] = math.max(aTmp.shape[i], bTmp.shape[bIndex]);
    }
      }
      List<int> aShape=List.from(broadcastedShape);
      broadcastedShape.removeLast();
      broadcastedShape.removeLast();
      List<int> bShape=List.from(broadcastedShape);
      bShape.addAll([bTmp.shape[bTmp.shape.length-2],bTmp.shape[bTmp.shape.length-1]]);
      return [aShape,bShape];
      
      
    
    
    } 
    else{throw Exception("wrong shape");}



}

Tensor max(Tensor a,Tensor b){
return a.max(b);

}




extension on List<int> {
  void addUniqueElement(int element) {
    int index = this.indexOf(element);
    if (index != -1) {
    } else {
      // 如果元素不存在，添加到列表中
      this.add(element);
    }
  }
}


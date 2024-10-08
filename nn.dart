import 'dart:async';
import 'autograd.dart';
import 'tensor.dart';
import 'dart:math' as math;

class Sequential {
  List<dynamic> layers;

  Sequential(this.layers);

  // 前向传播，依次执行所有层
  Node forward(Node input) {
    Node output = input;
    for (var layer in layers) {
      output = layer.forward(output);
    }
    return output;
  }
}

class Linear {
  int inputSize;
  int outputSize;
  late Node weights;
  late Node bias;

  Linear(this.inputSize, this.outputSize) {
    var rand = math.Random();
    List<double> weightData = List.generate(outputSize * inputSize, (_) => rand.nextDouble());
    
    // 权重的形状是 [outputSize, inputSize]
    weights = Node(Tensor(weightData, [outputSize, inputSize]));
    
    // 初始化偏置，形状为 [outputSize]
    List<double> biasData = List.generate(outputSize, (_) => rand.nextDouble()+0.001);//避免log函数报错
    bias = Node(Tensor(biasData, [outputSize]));
  }

  // 前向传播
 Node forward(Node input) {
    if (input.tensor.shape[0] != inputSize) {
      throw Exception('Input size must match Linear layer input size');
    }
    
    // 计算: output = weights * input + bias
    Node output=input.matmul(weights.transpose(weights.tensor.shape.length-2,weights.tensor.shape.length-1))+bias;
    
    return output;
  }
 
}
Node MSELoss(Node target,Node prediction)
{
return ((target-prediction).pow(2).sum())/target.tensor.data.length;

}

class ReLU{

Node forward(Node input){

return input.relu();

}
}

class Sigmoid{

Node forward(Node input){

return input.sigmoid();

}

}
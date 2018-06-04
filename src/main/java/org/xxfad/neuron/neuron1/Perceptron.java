package org.xxfad.neuron.neuron1;

import org.neuroph.core.Layer;
import org.neuroph.core.NeuralNetwork;
import org.neuroph.nnet.comp.layer.InputLayer;
import org.neuroph.nnet.comp.neuron.ThresholdNeuron;
import org.neuroph.nnet.learning.BinaryDeltaRule;
import org.neuroph.nnet.learning.PerceptronLearning;
import org.neuroph.util.*;

/**
 * 简单感知机
 *
 * @author xxfad 2018/5/26
 */
public class Perceptron extends NeuralNetwork<PerceptronLearning> {


    private static final long serialVersionUID = 8957359532187906229L;

    public Perceptron(int inputNeuronsCount, int outputNeuronsCount) {
        this.createNetwork(inputNeuronsCount, outputNeuronsCount, TransferFunctionType.STEP);
    }

    public Perceptron(int inputNeuronsCount, int outputNeuronsCount,
                      TransferFunctionType transferFunctionType) {
        this.createNetwork(inputNeuronsCount, outputNeuronsCount, transferFunctionType);
    }

    private void createNetwork(int inputNeuronsCount, int outputNeuronsCount,
                               TransferFunctionType transferFunctionType) {
        // 设置神经网络类型，为感知器
        this.setNetworkType(NeuralNetworkType.PERCEPTRON);

           // 创建输入刺激
        Layer inputLayer = new InputLayer(inputNeuronsCount);
        this.addLayer(inputLayer);

        NeuronProperties outputNeuronProperties = new NeuronProperties() {{
            setProperty("neuronType", ThresholdNeuron.class);
            setProperty("thresh", Math.abs(Math.random()));
            setProperty("transferFunction", transferFunctionType);
            // 为 TransferFunctionType.LINEAR 传输函数设置斜率属性
            setProperty("transferFunction.slope", 1d);
        }};

        // create 一个神经元的输出
        Layer outpuLayer = LayerFactory.createLayer(outputNeuronsCount,
                outputNeuronProperties);
        this.addLayer(outpuLayer);

        // 在输入和输出层中建建立安全链接
        ConnectionFactory.fullConnect(inputLayer, outpuLayer);

        // 为神经网络设置默认输入输出
        NeuralNetworkFactory.setDefaultIO(this);
        this.setLearningRule(new BinaryDeltaRule());

    }


}

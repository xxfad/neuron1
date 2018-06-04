package org.xxfad.neuron.neuron1;

import org.neuroph.core.Layer;
import org.neuroph.core.NeuralNetwork;
import org.neuroph.nnet.comp.layer.InputLayer;
import org.neuroph.nnet.comp.neuron.ThresholdNeuron;
import org.neuroph.nnet.learning.BinaryDeltaRule;
import org.neuroph.nnet.learning.PerceptronLearning;
import org.neuroph.util.*;

/**
 * �򵥸�֪��
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
        // �������������ͣ�Ϊ��֪��
        this.setNetworkType(NeuralNetworkType.PERCEPTRON);

           // ��������̼�
        Layer inputLayer = new InputLayer(inputNeuronsCount);
        this.addLayer(inputLayer);

        NeuronProperties outputNeuronProperties = new NeuronProperties() {{
            setProperty("neuronType", ThresholdNeuron.class);
            setProperty("thresh", Math.abs(Math.random()));
            setProperty("transferFunction", transferFunctionType);
            // Ϊ TransferFunctionType.LINEAR ���亯������б������
            setProperty("transferFunction.slope", 1d);
        }};

        // create һ����Ԫ�����
        Layer outpuLayer = LayerFactory.createLayer(outputNeuronsCount,
                outputNeuronProperties);
        this.addLayer(outpuLayer);

        // �������������н�������ȫ����
        ConnectionFactory.fullConnect(inputLayer, outpuLayer);

        // Ϊ����������Ĭ���������
        NeuralNetworkFactory.setDefaultIO(this);
        this.setLearningRule(new BinaryDeltaRule());

    }


}

package org.xxfad.neuron.neuron1;

import org.neuroph.core.NeuralNetwork;
import org.neuroph.core.data.DataSet;
import org.neuroph.core.data.DataSetRow;
import org.neuroph.core.events.LearningEvent;
import org.neuroph.core.events.LearningEventListener;
import org.neuroph.core.learning.LearningRule;
import org.neuroph.core.learning.SupervisedLearning;
import org.testng.annotations.Test;

import java.util.Arrays;

/**
 * @author xxfad 2018/5/26
 */
public class PerceptronTest implements LearningEventListener {

    @Test
    public void logicAnd() {

        DataSet trainingSet = new DataSet(2, 1) {{
            add(new DataSetRow(new double[]{0, 0}, new double[]{0}));
            add(new DataSetRow(new double[]{0, 1}, new double[]{0}));
            add(new DataSetRow(new double[]{1, 0}, new double[]{0}));
            add(new DataSetRow(new double[]{1, 1}, new double[]{1}));
        }};

        NeuralNetwork percerptron = new Perceptron(2, 1);
        LearningRule lr = percerptron.getLearningRule();
        lr.addListener(this);
        System.out.println("learning");
        lr.learn(trainingSet);
        System.out.println("learning end");

        System.out.println("testing");
        testNeuralNetwork(percerptron, trainingSet);
        System.out.println("testing end");
    }

    @Test
    public void logicOr() {

        DataSet trainingSet = new DataSet(2, 1) {{
            add(new DataSetRow(new double[]{0, 0}, new double[]{0}));
            add(new DataSetRow(new double[]{0, 1}, new double[]{1}));
            add(new DataSetRow(new double[]{1, 0}, new double[]{1}));
            add(new DataSetRow(new double[]{1, 1}, new double[]{1}));
        }};

        NeuralNetwork percerptron = new Perceptron(2, 1);
        LearningRule lr = percerptron.getLearningRule();
        lr.addListener(this);
        System.out.println("learning");
        lr.learn(trainingSet);
        System.out.println("learning end");

        System.out.println("testing");
        testNeuralNetwork(percerptron, trainingSet);
        System.out.println("testing end");

    }

    /**
     * Prints network output for the each element from the specified training set.
     *
     * @param neuralNet neural network
     * @param testSet   test set
     */
    public static void testNeuralNetwork(NeuralNetwork neuralNet, DataSet testSet) {

        for (DataSetRow trainingElement : testSet.getRows()) {
            neuralNet.setInput(trainingElement.getInput());
            neuralNet.calculate();
            double[] networkOutput = neuralNet.getOutput();

            System.out.print("Input: " + Arrays.toString(trainingElement.getInput()));
            System.out.println(" Output: " + Arrays.toString(networkOutput));
        }
    }

    @Override
    public void handleLearningEvent(LearningEvent event) {
        SupervisedLearning bp = (SupervisedLearning) event.getSource();
        if (event.getEventType() != LearningEvent.Type.LEARNING_STOPPED) {
            System.out.println(bp.getCurrentIteration() + ". iteration : " + bp.getTotalNetworkError());
        }
    }

}
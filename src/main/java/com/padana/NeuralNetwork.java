package com.padana;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import static org.nd4j.linalg.ops.transforms.Transforms.sigmoid;

public class NeuralNetwork {
    private int inodes;
    private int hnodes;
    private int onodes;
    private double lr;

    public INDArray wih;
    public INDArray who;

    public NeuralNetwork(int inputNodes, int hiddenNodes, int outputNodes, double learningRate) {
        this.inodes = inputNodes;
        this.hnodes = hiddenNodes;
        this.onodes = outputNodes;
        this.lr = learningRate;

        this.wih = Nd4j.rand(hiddenNodes, inputNodes).sub(0.5);
        this.who = Nd4j.rand(outputNodes, hiddenNodes).sub(0.5);
        //after that we should normalize the weights: They sample the weights
        //from a normal probability distribution centred around zero and with a standard deviation that is
        //related to the number of incoming links into a node, 1/√(number of incoming links)
    }

    public void train(INDArray inputList, INDArray targetList) {
        // calculate signals into hidden layer
        INDArray hiddenInput = wih.mmul(inputList);
        // calculate the signals emerging from hidden layer
        INDArray hiddenOutputs = activationFunction(hiddenInput);
        // calculate signals into final output layer
        INDArray finalInputs = who.mmul(hiddenOutputs);
        // calculate the signals emerging from final output layer
        INDArray finalOutputs = activationFunction(finalInputs);

        // error is the (target ­ actual)
        // generate the error matrix
        INDArray outputError = targetList.sub(finalOutputs);

        INDArray transposeWho = who.transpose();
        // the hidden error is transpose who multiply by output error
        INDArray hiddenError = transposeWho.mmul(outputError);

        // start update the weights for the links between the hidden and output layers
        this.who = this.who.add((outputError.mul(finalOutputs).mul(finalOutputs.muli(-1).addi(1.0))).muli(lr).mmul(hiddenOutputs.transpose()));
        // end update the weights for the links between the hidden and output layers

        // start update the weights for the links between the input and hidden layers
        this.wih = this.wih.add((hiddenError.mul(hiddenOutputs).mul(hiddenOutputs.muli(-1).addi(1.0))).muli(lr).mmul(inputList.transpose()));
        // end update the weights for the links between the input and hidden layers

    }

    public INDArray query(INDArray inputs) {
        // calculate signals into hidden layer
        INDArray hiddenInput = wih.mmul(inputs);
        // calculate the signals emerging from hidden layer
        INDArray hiddenOutputs = activationFunction(hiddenInput);
        // calculate signals into final output layer
        INDArray finalInputs = who.mmul(hiddenOutputs);
        // calculate the signals emerging from final output layer
        INDArray finalOutputs = activationFunction(finalInputs);

        return finalOutputs;
    }

    // activate function applies sigmoid function to array
    private INDArray activationFunction(INDArray activateArray) {
        return sigmoid(activateArray);
    }
}

package com.padana;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;
import java.util.stream.Collectors;

public class Main {
    public static void main(String[] args) {
        int inputNodes = 784;
        int hiddenNodes = 200;
        int outputNodes = 10;
        double learningRate = 0.02;
        int epochs = 2;


        NeuralNetwork neuralNetwork = new NeuralNetwork(inputNodes, hiddenNodes, outputNodes, learningRate);
        try {
            for (int i = 0; i < epochs; ++i) {
                train(neuralNetwork);
            }
            testNetwork(neuralNetwork);
        } catch (Exception e) {
            e.printStackTrace();
        }

      /*  try {
            File bufferedImage = new File("/home/dragos.pantiru/Downloads/3.jpg");
            BufferedImage image = ImageIO.read(bufferedImage);
            byte[] pixels = ((DataBufferByte) image.getRaster().getDataBuffer()).getData();
            inputNodes = pixels.length;
            double[] inputs = new double[pixels.length];
            for (int i = 1; i < pixels.length; ++i) {
                //Dividing the raw inputs which are in the range 0­255 by 255 will bring them into the range 0­1.
                //We then need to multiply by 0.99 to bring them into the range 0.0 ­ 0.99. We then add 0.01 to
                //shift them up to the desired range 0.01 to 1.00
                double tempVal = Double.parseDouble(String.valueOf(pixels[i - 1]));
                double finalVal = (tempVal / 255.0 * 0.9) + 0.01;
                inputs[i] = finalVal;
            }
            INDArray inputData = Nd4j.create(inputs, new int[]{pixels.length, 1});
            INDArray result = neuralNetwork.query(inputData);
            double numberIndex = Nd4j.argMax(result, 0).getDouble(0);
            System.out.println(result);
            System.out.println(numberIndex);

        } catch (IOException e) {
            e.printStackTrace();
        }*/

        /*double[] inputArray = {1.0, 0.5, -1.5};
        double[] flat = ArrayUtil.flattenDoubleArray(inputArray);
        INDArray inArr = Nd4j.create(flat, new int[]{3, 1}, 'c'); // create a 3x1 matrix from array

        double[] targetArray = {0.6, 0.45, 0.40};
        double[] flatTarget = ArrayUtil.flattenDoubleArray(inputArray);
        INDArray tarArr = Nd4j.create(flat, new int[]{3, 1}, 'c'); // create a 3x1 matrix from array*/

      /*  System.out.println(neuralNetwork.query(inArr));
        for (int i = 0; i < 100; ++i) {
            neuralNetwork.train(inArr, tarArr);
        }
        System.out.println(neuralNetwork.query(inArr));*/


    }

    private static void train(NeuralNetwork neuralNetwork) throws Exception {
        try {
            List<String> dataList = Files.lines(Paths.get("/home/dragos.pantiru/gitlab/neuralAiJava/src/main/resources/minst/mnist_train_100.csv")).parallel().collect(Collectors.toList());

            for (String data : dataList) {
                String[] dataStringArray = data.split(",");
                double[] inputs = new double[784];
                for (int i = 1; i < 784; ++i) {
                    //Dividing the raw inputs which are in the range 0­255 by 255 will bring them into the range 0­1.
                    //We then need to multiply by 0.99 to bring them into the range 0.0 ­ 0.99. We then add 0.01 to
                    //shift them up to the desired range 0.01 to 1.00
                    double tempVal = Double.parseDouble(dataStringArray[i - 1]);
                    double finalVal = (tempVal / 255.0 * 0.9) + 0.01;
                    inputs[i] = finalVal;
                }
                INDArray inputData = Nd4j.create(inputs, new int[]{784, 1});
                INDArray targetTrainArray = Nd4j.zeros(10, 1).addi(0.01).putScalar(Integer.parseInt(dataStringArray[0]), 0.99);
                // targetTrainArray.putScalar(Integer.parseInt(dataStringArray[0]), 0.99);

                neuralNetwork.train(inputData, targetTrainArray);
            }

        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    private static void testNetwork(NeuralNetwork neuralNetwork) {
        List<Double> scoreCard = new ArrayList<>();
        Double scoreCardSum = 0d;
        try {
            List<String> dataList = Files.lines(Paths.get("/home/dragos.pantiru/gitlab/neuralAiJava/src/main/resources/minst/mnist_test_10.csv")).parallel().collect(Collectors.toList());

            for (String data : dataList) {
                String[] dataStringArray = data.split(",");
                double[] inputs = new double[784];
                for (int i = 1; i < 784; ++i) {
                    //Dividing the raw inputs which are in the range 0­255 by 255 will bring them into the range 0­1.
                    //We then need to multiply by 0.99 to bring them into the range 0.0 ­ 0.99. We then add 0.01 to
                    //shift them up to the desired range 0.01 to 1.00
                    double tempVal = Double.parseDouble(dataStringArray[i - 1]);
                    double finalVal = (tempVal / 255.0 * 0.9) + 0.01;
                    inputs[i] = finalVal;
                }
                INDArray inputData = Nd4j.create(inputs, new int[]{784, 1});
                INDArray result = neuralNetwork.query(inputData);
                double numberIndex = Nd4j.argMax(result, 0).getDouble(0);

                scoreCard.add(numberIndex);
                if (Float.parseFloat(dataStringArray[0]) == numberIndex) {
                    scoreCardSum += 1d;
                }

                System.out.println(dataStringArray[0] + "---" + result);
            }

            System.out.println("Performance: " + scoreCardSum / scoreCard.size());

        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}

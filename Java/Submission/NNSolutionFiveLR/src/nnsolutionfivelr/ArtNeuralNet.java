/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package nnsolutionfivelr;

import static java.lang.Math.sqrt;
import java.util.ArrayList;
import java.util.Scanner;

/**
 *
 * @author ubuntu-admin
 */
public class ArtNeuralNet {

    public Double m_batch_error = 0.0;
    public Double m_batch_valid_error = 0.0;

    public int misPredictCount = 0;
    public static Double m_fraction_test_samples = 0.8;
    private Double m_error;
    private ArrayList<ArrayList<ArtNeuron>> m_layers = new ArrayList<ArrayList<ArtNeuron>>();

    public ArtNeuralNet(ArrayList<Integer> arch) {

        System.out.println(" Created New Neural Net");

        int num_layers = arch.size();

        for (int l_num = 0; l_num < num_layers; l_num++) {
            // create a new nerons layer
            ArrayList<ArtNeuron> newLayer = new ArrayList<ArtNeuron>();

            //if op layer then there are no further connections.
            int numOp = (l_num == arch.size() - 1) ? 0 : arch.get(l_num + 1);

            int num_neuron = 0;
            boolean isBiasNeuron = false;
            //create all neurons including bias
            for (num_neuron = 0; num_neuron <= arch.get(l_num); num_neuron++) {
                //last neuron created is bias neuron
                isBiasNeuron = num_neuron == arch.get(l_num);
                ArtNeuron newNeuron = new ArtNeuron(numOp, num_neuron, isBiasNeuron);
                newLayer.add(newNeuron);
//                System.out.println("");
            }

            //force bias neuron output to 1
            newLayer.get(num_neuron - 1).setOutVal(1.0);
            m_layers.add(newLayer);

        }

        // Print the numbers of neurons in each layer as a required output for the assignment
//        for (int l_num = 0; l_num < num_layers; l_num++) {
//            System.out.print("  " + arch.get(l_num));
//        }
//        System.out.println("");
//        for (int l_num = 1; l_num < num_layers; l_num++) {
//            ArrayList<Neuron> prevLayer = m_layers.get(l_num - 1);
//            for (int num_neuron = 0; num_neuron < arch.get(l_num); num_neuron++) {
//
//                for (int pre_neuron = 0; pre_neuron <= prevLayer.size() - 1; pre_neuron++) {
//                    System.out.print("  " + prevLayer.get(pre_neuron).getWeights().get(num_neuron).weight);
//                }
//
//                System.out.println("");
//            }
//            System.out.println("");
//        }
        if (NNSolutionFiveLR.isWeightInput) {
            //Accept weights for individual neurons in the specific layers
            for (int l_num = 1; l_num < num_layers; l_num++) {
                ArrayList<ArtNeuron> prevLayer = m_layers.get(l_num - 1);
                int num_neuron = 0;
                int pre_neuron = 0;

                for (num_neuron = 0; num_neuron < arch.get(l_num); num_neuron++) {

                    System.out.println("Enter weights of layer " + l_num + " Neuron no " + num_neuron);
                    Scanner scanner = new Scanner(System.in);
                    String str_weights = scanner.nextLine();

                    String[] weight_array = str_weights.split(",", -1);

                    if (NNSolutionFiveLR.debugMode) {
                        System.out.println("Your weights for layer are " + str_weights);
                    }
                    for (pre_neuron = 0; pre_neuron <= prevLayer.size() - 2; pre_neuron++) {
                        //                    System.out.print(prevLayer.get(pre_neuron).getWeights().get(num_neuron).weight + ",");
                        Double weight_val = Double.parseDouble(weight_array[pre_neuron]);
                        prevLayer.get(pre_neuron).getWeights().get(num_neuron).weight = weight_val;
                    }
                }
            }

            // Print the weight values to verify all neuron weights are initialized as specified by the user
            if (NNSolutionFiveLR.debugMode) {

                for (int l_num = 1; l_num < num_layers; l_num++) {
                    ArrayList<ArtNeuron> prevLayer = m_layers.get(l_num - 1);
                    int num_neuron = 0;
                    int pre_neuron = 0;

                    for (num_neuron = 0; num_neuron < arch.get(l_num); num_neuron++) {

                        for (pre_neuron = 0; pre_neuron <= prevLayer.size() - 2; pre_neuron++) {
                            System.out.print(prevLayer.get(pre_neuron).getWeights().get(num_neuron).weight + ",");
                        }
                        System.out.print(prevLayer.get(pre_neuron).getWeights().get(num_neuron).weight);

                        System.out.println("");
                    }
                }

            }
        }

    }

    public void forwardPass(ArrayList<Double> inputVal) {
        // validate number of inputs == number of neurons in input layer

        //load inputs
        for (int i = 0; i < inputVal.size(); i++) {
            m_layers.get(0).get(i).setOutVal(inputVal.get(i));
        }
        //forward feed
        for (int layer_num = 1; layer_num < m_layers.size(); layer_num++) {
            ArrayList<ArtNeuron> prevLayer = m_layers.get(layer_num - 1);
            ArrayList<ArtNeuron> currentLayer = m_layers.get(layer_num);

            boolean isFinalLayer = ((m_layers.size() - 1) == layer_num);
            // access all but bias neurons
            for (int n = 0; n < m_layers.get(layer_num).size() - 1; n++) {
                currentLayer.get(n).feedForward(prevLayer, isFinalLayer);
            }
        }

    }

    
    public void calculateValidationError(ArrayList<Double> outputVal) {
                // calculate error of network
        ArrayList<ArtNeuron> outLayer = m_layers.get(m_layers.size() - 1);
        double valid_error = 0.0;
                // loop all neurons except the bias neuron
        for (int n = 0; n < outLayer.size() - 1; n++) {
            Double delta = outputVal.get(n) - outLayer.get(n).getOutVal();

//            System.out.println(" Expected Out = " + outputVal.get(n) + " calculated val = " + outLayer.get(n).getOutVal());
            valid_error += delta * delta;
        }

        valid_error /= outLayer.size() - 1; // get average seuared error
        valid_error = sqrt(valid_error);    //RMS
        
        m_batch_valid_error += valid_error;
    }
    public void backwordPass(ArrayList<Double> outputVal) {
        // calculate error of network
        ArrayList<ArtNeuron> outLayer = m_layers.get(m_layers.size() - 1);
        m_error = 0.0;

        // loop all neurons except the bias neuron
        for (int n = 0; n < outLayer.size() - 1; n++) {
            Double delta = outputVal.get(n) - outLayer.get(n).getOutVal();

//            System.out.println(" Expected Out = " + outputVal.get(n) + " calculated val = " + outLayer.get(n).getOutVal());
            m_error += delta * delta;
        }

        m_error /= outLayer.size() - 1; // get average seuared error
        m_error = sqrt(m_error);    //RMS
        
        m_batch_error += m_error;

//        System.out.println(" RMS Error = " + m_error);
        // calculate gradient of output layer
        // loop all neurons except the bias neuron
        for (int n = 0; n < outLayer.size() - 1; n++) {
            outLayer.get(n).finalLayerGradientCalculation(outputVal.get(n));
        }

        // calculate gradient of hidden layers
        for (int lyr_num = m_layers.size() - 2; lyr_num > 0; --lyr_num) {
            ArrayList<ArtNeuron> hiddenLayer = m_layers.get(lyr_num);
            ArrayList<ArtNeuron> nextLayer = m_layers.get(lyr_num + 1);

            // loop all neurons except the bias neuron
            for (int n = 0; n < hiddenLayer.size() - 1; n++) {
                hiddenLayer.get(n).hiddenLayerGradientCalculation(nextLayer);
            }
        }

        //Print the partial derivatives of the neurons from last to first layer.
//        for (int lyr_num = m_layers.size() - 1; lyr_num > 0; --lyr_num) {
//            ArrayList<ArtNeuron> currentLayer = m_layers.get(lyr_num);
//
////            if (NNSolutionFiveLR.debugMode) {
//                System.out.println(" Printing gradients for layer = " + lyr_num);
////            }
//
//            int n = 0;
//            // loop all neurons except the bias neuron
//            for (n = 0; n < currentLayer.size() - 1; n++) {
//                System.out.print(currentLayer.get(n).getGradient() + ",");
//            }
//            System.out.println(currentLayer.get(n).getGradient());
//
//        }
        // update weights according to the gradient
        for (int lyr_num = m_layers.size() - 1; lyr_num > 0; --lyr_num) {
            ArrayList<ArtNeuron> currentLayer = m_layers.get(lyr_num);
            ArrayList<ArtNeuron> prevLayer = m_layers.get(lyr_num - 1);

            // loop all neurons except the bias neuron
            for (int n = 0; n < currentLayer.size() - 1; n++) {
                currentLayer.get(n).weightsUpdate(prevLayer);
            }
        }

    }

    public void getResult(ArrayList<Double> result) {
        result.clear();
        int numLayers = m_layers.size() - 1;
        for (int n = 0; n < m_layers.get(numLayers).size() - 1; n++) {
            result.add(m_layers.get(numLayers).get(n).getOutVal());
        }
    }

    public static void showVector(ArrayList<Double> inputVal) {
        for (int i = 0; i < inputVal.size(); i++) {
            System.out.print("," + inputVal.get(i));

        }
        System.out.println("");
    }

    public Double getRMSerr() {
        return m_error;
    }

    public void printRequiredResults() {
        System.out.println(" RMS batch Error = " + m_batch_error);

        if (NNSolutionFiveLR.debugMode) {
            System.out.println(" Displaying architecture : ");

            //print architecture
            for (int l_num = 0; l_num < m_layers.size(); l_num++) {
                System.out.print(m_layers.get(l_num).size() - 1 + ",");
            }
            System.out.println("");

            // Print the weight values to verify all neuron weights are initialized as specified by the user
            System.out.println(" Displaying weights : ");
            for (int l_num = 1; l_num < m_layers.size() - 1; l_num++) {
                ArrayList<ArtNeuron> prevLayer = m_layers.get(l_num - 1);
                int num_neuron = 0;
                int pre_neuron = 0;

                for (num_neuron = 0; num_neuron < m_layers.get(l_num).size() - 1; num_neuron++) {

                    for (pre_neuron = 0; pre_neuron <= prevLayer.size() - 2; pre_neuron++) {
                        System.out.print(prevLayer.get(pre_neuron).getWeights().get(num_neuron).weight + ",");
                    }
                    System.out.print(prevLayer.get(pre_neuron).getWeights().get(num_neuron).weight);

                    System.out.println("");
                }
            }
        }

    }

    public void trainNetwork(ArrayList<Double> inputValues, ArrayList<Double> targetValues) {

        //Forward Pass
        forwardPass(inputValues);

        //Results verification
        ArrayList<Double> outputValues = new ArrayList<Double>();
        getResult(outputValues);

        //Backward Pass  
        backwordPass(targetValues);
//        System.out.println("");

        if (NNSolutionFiveLR.debugMode) {
            System.out.print(" Input :: ");
            showVector(inputValues);

            System.out.print(" Result :: ");
            showVector(outputValues);

            System.out.print(" Expected :: ");
            showVector(targetValues);
        }

    }

    public void verifyNetwork(ArrayList<Double> inputValues, ArrayList<Double> targetValues) {

        //Forward Pass
        forwardPass(inputValues);

        //Results verification
        ArrayList<Double> outputValues = new ArrayList<Double>();
        getResult(outputValues);

        int bin_out = outputValues.get(0) > 0.9 ? 1 : 0;

        if (bin_out != targetValues.get(0)) {
            misPredictCount++;
        }
        
        calculateValidationError(targetValues);

        if (NNSolutionFiveLR.debugMode) {
            System.out.print(" Input :: ");
            showVector(inputValues);

            System.out.print(" Result :: ");
            showVector(outputValues);

            System.out.print(" Expected :: ");
            showVector(targetValues);
        }

    }
}

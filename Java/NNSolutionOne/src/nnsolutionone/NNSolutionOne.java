/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package nnsolutionone;

import static java.lang.Math.sqrt;
import static java.lang.Math.tanh;
import java.util.ArrayList;
import java.util.Random;

class Connection {

    public Double weight;
    public Double deltaWeight;

    public Connection(boolean isBiasNeuron) {
        weight = isBiasNeuron ? 0.0 : randomWeight();
    }

    public Double randomWeight() {
        Random rand = new Random();
        double standardDeviation = 0.1;
        double requiredMean = 0;
        Double num = rand.nextGaussian() * standardDeviation + requiredMean;

        return num;
    }

};

class Neuron {

    public Neuron(int numOp, int myIndex, boolean isBiasNeuron) {
//        System.out.println(" created Neuron ");

        // Initialize all connections with random values
        for (int num_connect = 0; num_connect < numOp; num_connect++) {
            Connection newConnection = new Connection(isBiasNeuron);
            m_weights.add(newConnection);
        }
        my_Index = myIndex;
    }

    static double activationFun(double input) {
        return Math.max(0.0, input);
//        return tanh(input);
    }

    static double activationFunDerivative(double input) {
        //when x<0 so the derivative of f(x) with respect to x gives result f'(x)=0.
        //In the second case, it's clear to compute f'(x)=1    
        return (input <= 0.0 ? 0.0 : 1.0); 
//        return (1.0 - (input * input));
    }

    public void feedForward(ArrayList<Neuron> prevLayer, boolean isFinalLayer) {
        double sum = 0.0;

        // Including the boas neuron in the previous layer
        for (int n = 0; n < prevLayer.size(); n++) {
            double val = prevLayer.get(n).getOutVal();
            sum += (val * prevLayer.get(n).m_weights.get(my_Index).weight);
        }
        // If it is final layer then apply unit linearity otherwise for all other layers apply activation function
        m_outVal = isFinalLayer ? sum : activationFun(sum);
    }

    public void setOutVal(double outVal) {
        m_outVal = outVal;
    }

    public double getOutVal() {
        return m_outVal;
    }

    public void finalLayerGradientCalculation(double outVal) {
        double delta = outVal - m_outVal;
        m_gradient = delta * activationFunDerivative(m_outVal);

    }

    public void hiddenLayerGradientCalculation(ArrayList<Neuron> nextLayer) {
        double dow = delta_mul_weight_sum(nextLayer);
        m_gradient = dow * activationFunDerivative(m_outVal);
    }

    private double delta_mul_weight_sum(ArrayList<Neuron> nextLayer) {
        double sum = 0.0;
        // sum our contributions of the errors at the nodes we feed
        for (int num_neuron = 0; num_neuron < nextLayer.size() - 1; num_neuron++) {
            sum += m_weights.get(num_neuron).weight * nextLayer.get(num_neuron).m_gradient;
        }

        return sum;
    }

    public void weightsUpdate(ArrayList<Neuron> prevLayer) {
        // all neurons including the bias
        for (int num_neuron = 0; num_neuron < prevLayer.size(); num_neuron++) {
            Neuron currNeuron = prevLayer.get(num_neuron);
//        double oldDeltaWeight = currNeuron.m_weights.get(my_Index).deltaWeight;

            double newDeltaWeight
                    = // Individual input, magnified by the gradient and train rate
                    eta
                    * currNeuron.getOutVal()
                    * m_gradient;
            // Also add a momentum = a fraction of the previous delta weight
//                + alpha
//                * oldDeltaWeight;

            currNeuron.m_weights.get(my_Index).deltaWeight = newDeltaWeight;
            currNeuron.m_weights.get(my_Index).weight += newDeltaWeight;

        }
    }

    public ArrayList<Connection> getWeights() {
        return m_weights;
    }
    private static double eta = 0.15;
    private double m_outVal = new Double(0);
    private ArrayList<Connection> m_weights = new ArrayList<Connection>();
    private int my_Index;
    private double m_gradient;

}

class NeuralNet {

    public NeuralNet(ArrayList<Integer> arch) {

        int num_layers = arch.size();

        for (int l_num = 0; l_num < num_layers; l_num++) {
            // create a new nerons layer
            ArrayList<Neuron> newLayer = new ArrayList<Neuron>();

            //if op layer then there are no further connections.
            int numOp = (l_num == arch.size() - 1) ? 0 : arch.get(l_num + 1);

            int num_neuron = 0;
            boolean isBiasNeuron = false;
            //create all neurons including bias
            for (num_neuron = 0; num_neuron <= arch.get(l_num); num_neuron++) {
                //last neuron created is bias neuron
                isBiasNeuron = num_neuron == arch.get(l_num);
                Neuron newNeuron = new Neuron(numOp, num_neuron, isBiasNeuron);
                newLayer.add(newNeuron);
                System.out.println("");
            }

            //force bias neuron output to 1
            newLayer.get(num_neuron - 1).setOutVal(1.0);
            m_layers.add(newLayer);

        }

        // Print the numbers of neurons in each layer as a required output for the assignment
        for (int l_num = 0; l_num < num_layers; l_num++) {
            System.out.print("  " + arch.get(l_num));
        }
        System.out.println("");
        for (int l_num = 1; l_num < num_layers; l_num++) {
            ArrayList<Neuron> prevLayer = m_layers.get(l_num - 1);
            for (int num_neuron = 0; num_neuron < arch.get(l_num); num_neuron++) {

                for (int pre_neuron = 0; pre_neuron <= prevLayer.size() - 1; pre_neuron++) {
                    System.out.print("  " + prevLayer.get(pre_neuron).getWeights().get(num_neuron).weight);
                }

                System.out.println("");
            }
            System.out.println("");
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
            ArrayList<Neuron> prevLayer = m_layers.get(layer_num - 1);
            ArrayList<Neuron> currentLayer = m_layers.get(layer_num);
            
            boolean isFinalLayer = ((m_layers.size() - 1 ) == layer_num);
            // access all but bias neurons
            for (int n = 0; n < m_layers.get(layer_num).size() - 1; n++) {
                currentLayer.get(n).feedForward(prevLayer, isFinalLayer);
            }
        }

    }

    public void backwordPass(ArrayList<Double> outputVal) {
        // calculate error of network
        ArrayList<Neuron> outLayer = m_layers.get(m_layers.size() - 1);
        m_error = 0.0;

        // loop all neurons except the bias neuron
        for (int n = 0; n < outLayer.size() - 1; n++) {
            Double delta = outputVal.get(n) - outLayer.get(n).getOutVal();
            m_error += delta * delta;
        }

        m_error /= outLayer.size() - 1; // get average seuared error
        m_error = sqrt(m_error);    //RMS

        // calculate gradient of output layer
        // loop all neurons except the bias neuron
        for (int n = 0; n < outLayer.size() - 1; n++) {
            outLayer.get(n).finalLayerGradientCalculation(outputVal.get(n));
        }

        // calculate gradient of hidden layers
        for (int lyr_num = m_layers.size() - 2; lyr_num > 0; --lyr_num) {
            ArrayList<Neuron> hiddenLayer = m_layers.get(lyr_num);
            ArrayList<Neuron> nextLayer = m_layers.get(lyr_num + 1);

            // loop all neurons except the bias neuron
            for (int n = 0; n < hiddenLayer.size() - 1; n++) {
                hiddenLayer.get(n).hiddenLayerGradientCalculation(nextLayer);
            }
        }

        // update weights according to the gradient
        for (int lyr_num = m_layers.size() - 1; lyr_num > 0; --lyr_num) {
            ArrayList<Neuron> currentLayer = m_layers.get(lyr_num);
            ArrayList<Neuron> prevLayer = m_layers.get(lyr_num - 1);

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

    private Double m_error;
    private ArrayList<ArrayList<Neuron>> m_layers = new ArrayList<ArrayList<Neuron>>();
};

/**
 *
 * @author ubuntu-admin
 */
public class NNSolutionOne {

    public static void showVector(ArrayList<Double> inputVal) {
        for (int i = 0; i < inputVal.size(); i++) {
            System.out.print(" " + inputVal.get(i));

        }
        System.out.println("");
    }

    /**
     * @param args the command line arguments
     */
    public static void main(String[] args) {
        // TODO code application logic here

        ArrayList<Integer> architecture = new ArrayList<Integer>();
        architecture.add(2);
        architecture.add(4);
        architecture.add(1);

        NeuralNet myNet = new NeuralNet(architecture);

        
        for (int epoc = 1; epoc < 10000; epoc++) {

            System.out.println(" Pass " + epoc);

            // 1,1 = 1
            ArrayList<Double> inputVal = new ArrayList<Double>();
            inputVal.add(1.0);
            inputVal.add(1.0);
            System.out.print(" Input :: ");

            showVector(inputVal);

            myNet.forwardPass(inputVal);

            ArrayList<Double> resultVal = new ArrayList<Double>();
            myNet.getResult(resultVal);
            System.out.print(" Result :: ");

            showVector(resultVal);

            ArrayList<Double> targetVal = new ArrayList<Double>();
            targetVal.add(1.0);
            System.out.print(" Expected :: ");
            showVector(targetVal);

            myNet.backwordPass(targetVal);
            System.out.println("");

            // 0,0 = 0
            inputVal.clear();
            inputVal.add(0.0);
            inputVal.add(0.0);
            System.out.print(" Input :: ");
            showVector(inputVal);

            myNet.forwardPass(inputVal);

            resultVal.clear();
            myNet.getResult(resultVal);
            System.out.print(" Result :: ");
            showVector(resultVal);

            targetVal.clear();
            targetVal.add(0.0);
            System.out.print(" Expected :: ");
            showVector(targetVal);

            myNet.backwordPass(targetVal);

            System.out.println("");

            // 1,0 = 0
            inputVal.clear();
            inputVal.add(1.0);
            inputVal.add(0.0);
            System.out.print(" Input :: ");
            showVector(inputVal);

            myNet.forwardPass(inputVal);

            resultVal.clear();
            myNet.getResult(resultVal);
            System.out.print(" Result :: ");
            showVector(resultVal);

            targetVal.clear();
            targetVal.add(0.0);
            System.out.print(" Expected :: ");
            showVector(targetVal);

            myNet.backwordPass(targetVal);

            System.out.println("");

            // 0,1 = 0
            inputVal.clear();
            inputVal.add(0.0);
            inputVal.add(1.0);
            System.out.print(" Input :: ");
            showVector(inputVal);

            myNet.forwardPass(inputVal);

            resultVal.clear();
            myNet.getResult(resultVal);
            System.out.print(" Result :: ");
            showVector(resultVal);

            targetVal.clear();
            targetVal.add(0.0);
            System.out.print(" Expected :: ");
            showVector(targetVal);

            myNet.backwordPass(targetVal);

        }
    }

}

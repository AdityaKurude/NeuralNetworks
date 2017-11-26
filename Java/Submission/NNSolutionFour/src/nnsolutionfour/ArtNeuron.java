/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package nnsolutionfour;

import java.util.ArrayList;

/**
 *
 * @author ubuntu-admin
 */


public class ArtNeuron {
    
    public ArtNeuron(int numOp, int myIndex, boolean isBiasNeuron) {
//        System.out.println(" created Neuron ");

        // Initialize all connections with random values
        for (int num_connect = 0; num_connect < numOp; num_connect++) {
            NeuronConnect newNeuronConnect = new NeuronConnect(isBiasNeuron);
            m_weights.add(newNeuronConnect);
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

    public void feedForward(ArrayList<ArtNeuron> prevLayer, boolean isFinalLayer) {
        double sum = 0.0;

        // Including the boas neuron in the previous layer
        for (int n = 0; n < prevLayer.size(); n++) {
            double val = prevLayer.get(n).getOutVal();
            sum += (val * prevLayer.get(n).m_weights.get(my_Index).weight);
        }
        // If it is final layer then apply unit linearity otherwise for all other layers apply activation function
        m_outVal = isFinalLayer ? sum : activationFun(sum);
    }

    public void setOutVal(double outVal) { m_outVal = outVal; }

    public double getOutVal() { return m_outVal; }

    public void finalLayerGradientCalculation(double outVal) {
        double delta = outVal - m_outVal;
        m_gradient = delta * activationFunDerivative(m_outVal);

    }

    public void hiddenLayerGradientCalculation(ArrayList<ArtNeuron> nextLayer) {
        double dow = delta_mul_weight_sum(nextLayer);
        m_gradient = dow * activationFunDerivative(m_outVal);
    }

    private double delta_mul_weight_sum(ArrayList<ArtNeuron> nextLayer) {
        double sum = 0.0;
        // sum our contributions of the errors at the nodes we feed
        for (int num_neuron = 0; num_neuron < nextLayer.size() - 1; num_neuron++) {
            sum += m_weights.get(num_neuron).weight * nextLayer.get(num_neuron).m_gradient;
        }

        return sum;
    }

    public void weightsUpdate(ArrayList<ArtNeuron> prevLayer) {
        // all neurons including the bias
        for (int num_neuron = 0; num_neuron < prevLayer.size(); num_neuron++) {
            ArtNeuron currNeuron = prevLayer.get(num_neuron);

            double newDeltaWeight
                    = // Individual input, magnified by the gradient and train rate
                    NNSolutionFour.eta
                    * currNeuron.getOutVal()
                    * m_gradient;

            currNeuron.m_weights.get(my_Index).deltaWeight = newDeltaWeight;
            currNeuron.m_weights.get(my_Index).weight += newDeltaWeight;

        }
    }

    public ArrayList<NeuronConnect> getWeights() { return m_weights; }
    
    private double m_outVal = new Double(0);
    private ArrayList<NeuronConnect> m_weights = new ArrayList<NeuronConnect>();
    private int my_Index;
    
    public double getGradient() { return m_gradient; }
    
    private double m_gradient;    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
//        public ArtNeuron(int numOp, int myIndex, boolean isBiasNeuron) {
////        System.out.println(" created Neuron ");
//
//        // Initialize all connections with random values
//        for (int num_connect = 0; num_connect < numOp; num_connect++) {
//            NeuronConnect newNeuronConnect = new NeuronConnect(isBiasNeuron);
//            m_weights.add(newNeuronConnect);
//        }
//        my_Index = myIndex;
//    }
//
//    static double activationFun(double input) {
//        return Math.max(0.0, input);
////        return tanh(input);
//    }
//
//    static double activationFunDerivative(double input) {
//        //when x<0 so the derivative of f(x) with respect to x gives result f'(x)=0.
//        //In the second case, it's clear to compute f'(x)=1    
//        return (input <= 0.0 ? 0.0 : 1.0);
////        return (1.0 - (input * input));
//    }
//
//    public void feedForward(ArrayList<ArtNeuron> prevLayer, boolean isFinalLayer) {
//        double sum = 0.0;
//
//        // Including the boas neuron in the previous layer
//        for (int n = 0; n < prevLayer.size(); n++) {
//            double val = prevLayer.get(n).getOutVal();
//            sum += (val * prevLayer.get(n).m_weights.get(my_Index).weight);
//        }
//        // If it is final layer then apply unit linearity otherwise for all other layers apply activation function
//        m_outVal = isFinalLayer ? sum : activationFun(sum);
//    }
//
//    public void setOutVal(double outVal) { m_outVal = outVal; }
//
//    public double getOutVal() { return m_outVal; }
//
//    public void finalLayerGradientCalculation(double outVal) {
//        double delta = outVal - m_outVal;
//        m_gradient = delta * activationFunDerivative(m_outVal);
//
//    }
//
//    public void hiddenLayerGradientCalculation(ArrayList<ArtNeuron> nextLayer) {
//        double dow = delta_mul_weight_sum(nextLayer);
//        m_gradient = dow * activationFunDerivative(m_outVal);
//    }
//
//    private double delta_mul_weight_sum(ArrayList<ArtNeuron> nextLayer) {
//        double sum = 0.0;
//        // sum our contributions of the errors at the nodes we feed
//        for (int num_neuron = 0; num_neuron < nextLayer.size() - 1; num_neuron++) {
//            sum += m_weights.get(num_neuron).weight * nextLayer.get(num_neuron).m_gradient;
//        }
//
//        return sum;
//    }
//
//    public void weightsUpdate(ArrayList<ArtNeuron> prevLayer) {
//        // all neurons including the bias
//        for (int num_neuron = 0; num_neuron < prevLayer.size(); num_neuron++) {
//            ArtNeuron currNeuron = prevLayer.get(num_neuron);
////        double oldDeltaWeight = currNeuron.m_weights.get(my_Index).deltaWeight;
//
//            double newDeltaWeight
//                    = // Individual input, magnified by the gradient and train rate
//                    eta
//                    * currNeuron.getOutVal()
//                    * m_gradient;
//            // Also add a momentum = a fraction of the previous delta weight
////                + alpha
////                * oldDeltaWeight;
//
//            currNeuron.m_weights.get(my_Index).deltaWeight = newDeltaWeight;
//            currNeuron.m_weights.get(my_Index).weight += newDeltaWeight;
//
//        }
//    }
//        public double getGradient() { return m_gradient; }
//
//
//    public ArrayList<NeuronConnect> getWeights() { return m_weights; }
//    public static void set_eta(double value) { eta = value; }
//    
//    private static double eta = 0.015;
//    private double m_outVal = new Double(0);
//    private ArrayList<NeuronConnect> m_weights = new ArrayList<NeuronConnect>();
//    private int my_Index;
//    private double m_gradient;
}



/*
public class ArtNeuron {

    public ArtNeuron(int numOp, int myIndex, boolean isBiasNeuron) {

        if (NNSolutionFour.debugMode) {
            System.out.println(" created ArtNeuron ");
        }

        // Initialize all connections with random values
        for (int num_connect = 0; num_connect < numOp; num_connect++) {
            NeuronConnect newNeuronConnect = new NeuronConnect(isBiasNeuron);
            m_weights.add(newNeuronConnect);
        }
        my_Index = myIndex;

    }

    public void feedForward(ArrayList<ArtNeuron> prevLayer, boolean isFinalLayer) {
        double sum = 0.0;

        // Including the boas neuron in the previous layer
        for (int n = 0; n < prevLayer.size(); n++) {
            double val = prevLayer.get(n).getOutVal();
            sum += (val * prevLayer.get(n).m_weights.get(my_Index).weight);
        }
        // If it is final layer then apply unit linearity otherwise for all other layers apply activation function
        m_outVal = isFinalLayer ? sum : activationFun(sum);
    }

    static double activationFun(double input) {
        return Math.max(0.0, input);
    }

    static double activationFunDerivative(double input) {
        //when x<0 so the derivative of f(x) with respect to x gives result f'(x)=0.
        //In the second case, it's clear to compute f'(x)=1    
        return (input <= 0.0 ? 0.0 : 1.0);
    }

    public void finalLayerGradientCalculation(double outVal) {
        double delta = outVal - m_outVal;
        m_gradient = delta * activationFunDerivative(m_outVal);

    }

    public void hiddenLayerGradientCalculation(ArrayList<ArtNeuron> nextLayer) {
        double dow = delta_mul_weight_sum(nextLayer);
        m_gradient = dow * activationFunDerivative(m_outVal);
    }

    private double delta_mul_weight_sum(ArrayList<ArtNeuron> nextLayer) {
        double sum = 0.0;
        // sum our contributions of the errors at the nodes we feed
        for (int num_neuron = 0; num_neuron < nextLayer.size() - 1; num_neuron++) {
            sum += m_weights.get(num_neuron).weight * nextLayer.get(num_neuron).m_gradient;
        }

        return sum;
    }

    
        public void weightsUpdate(ArrayList<ArtNeuron> prevLayer) {
        // all neurons including the bias
        for (int num_neuron = 0; num_neuron < prevLayer.size(); num_neuron++) {
            ArtNeuron currNeuron = prevLayer.get(num_neuron);
//        double oldDeltaWeight = currNeuron.m_weights.get(my_Index).deltaWeight;

            double newDeltaWeight
                    = // Individual input, magnified by the gradient and train rate
//                    NNSolutionFour.eta
                    0.015
                    * currNeuron.getOutVal()
                    * m_gradient;

            currNeuron.m_weights.get(my_Index).weight += newDeltaWeight;
        }
    }
        
    public void setOutVal(double outVal) {
        m_outVal = outVal;
    }

    public double getOutVal() {
        return m_outVal;
    }

    public ArrayList<NeuronConnect> getWeights() {
        return m_weights;
    }

    public double getGradient() { return m_gradient; }
    
    private double m_outVal = new Double(0);
    private int my_Index;
    private double m_gradient;

    private ArrayList<NeuronConnect> m_weights = new ArrayList<NeuronConnect>();
}
*/
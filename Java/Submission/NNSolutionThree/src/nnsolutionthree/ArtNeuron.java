/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package nnsolutionthree;

import java.util.ArrayList;

/**
 *
 * @author ubuntu-admin
 */
public class ArtNeuron {

    public ArtNeuron(int numOp, int myIndex, boolean isBiasNeuron) {

        if (NNSolutionThree.debugMode) {
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

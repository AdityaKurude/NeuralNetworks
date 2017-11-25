/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package nnsolutionthree;

import java.util.Random;

/**
 *
 * @author ubuntu-admin
 */
public class NeuronConnect {

    public Double weight;

    public NeuronConnect(boolean isBiasNeuron) {
        //If the neuron is bias neuron then initialize its weight to zero.
        weight = isBiasNeuron ? 0.0 : randomWeight();
    }

    public Double randomWeight() {
        Random rand = new Random();
        double standardDeviation = 0.1;
        double requiredMean = 0;
        Double num = rand.nextGaussian() * standardDeviation + requiredMean;

        return num;
    }

}

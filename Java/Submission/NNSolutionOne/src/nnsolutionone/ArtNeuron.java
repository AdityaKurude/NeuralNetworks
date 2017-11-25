/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package nnsolutionone;

import java.util.ArrayList;

/**
 *
 * @author ubuntu-admin
 */
public class ArtNeuron {

    public ArtNeuron(int numOp, int myIndex, boolean isBiasNeuron) {
        
        if (NNSolutionOne.debugMode) {
            System.out.println(" created ArtNeuron ");
        }

        // Initialize all connections with random values
        for (int num_connect = 0; num_connect < numOp; num_connect++) {
            NeuronConnect newNeuronConnect = new NeuronConnect(isBiasNeuron);
            m_weights.add(newNeuronConnect);
        }
    }

    public void setOutVal(double outVal) {
        m_outVal = outVal;
    }

    public ArrayList<NeuronConnect> getWeights() {
        return m_weights;
    }

    private double m_outVal = new Double(0);

    private ArrayList<NeuronConnect> m_weights = new ArrayList<NeuronConnect>();
}

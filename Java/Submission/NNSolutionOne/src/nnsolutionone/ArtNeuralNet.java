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
public class ArtNeuralNet {

    public ArtNeuralNet(ArrayList<Integer> arch) {

        if (NNSolutionOne.debugMode) {
            System.out.println(" Created New Neural Net");
        }

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
                ArtNeuron newArtNeuron = new ArtNeuron(numOp, num_neuron, isBiasNeuron);
                newLayer.add(newArtNeuron);
//                System.out.println("");
            }

            //force bias neuron output to 1
            newLayer.get(num_neuron - 1).setOutVal(1.0);
            m_layers.add(newLayer);

        }

        // Print the numbers of neurons in each layer as a required output for the assignment
        int l_num = 0;
        for (l_num = 0; l_num < num_layers - 1; l_num++) {
            System.out.print(arch.get(l_num)+",");
        }
        System.out.print(arch.get(l_num));
        System.out.println("");
        
        for (l_num = 1; l_num < num_layers; l_num++) {
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

    private ArrayList<ArrayList<ArtNeuron>> m_layers = new ArrayList<ArrayList<ArtNeuron>>();
}

/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package nnsolutionone;

import java.util.ArrayList;
import java.util.Scanner;

/**
 *
 * @author ubuntu-admin
 */
public class NNSolutionOne {

    public static final boolean debugMode = false;

    /* or false :-) */
    /**
     * @param args the command line arguments
     */
    public static void main(String[] args) {
        // TODO code application logic here
        
        System.out.println("Enter architecture of the ArtNeural Network: ");
        Scanner scanner = new Scanner(System.in);
        String str_architecture = scanner.nextLine();

        if (debugMode) {
            System.out.println("Your architecture is " + str_architecture);
        }

        String[] array = str_architecture.split(",", -1);
        ArrayList<Integer> architecture = new ArrayList<Integer>();

        for (int num_layers = 0; num_layers < array.length; num_layers++) {

            int num_neurons = Integer.parseInt(array[num_layers]);
            architecture.add(num_neurons);
        }

        ArtNeuralNet myNet = new ArtNeuralNet(architecture);
    }

}

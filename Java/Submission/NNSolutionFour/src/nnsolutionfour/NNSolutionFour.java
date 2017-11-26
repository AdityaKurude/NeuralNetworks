/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package nnsolutionfour;

import java.util.ArrayList;
import java.util.Scanner;

/**
 *
 * @author ubuntu-admin
 */
public class NNSolutionFour {

    public static final boolean debugMode = true;
    public static final boolean isWeightInput = false;
    public static double eta = 0.0;

    /* or false :-) */
    /**
     * @param args the command line arguments
     */
    public static void main(String[] args) {
        // TODO code application logic here

        System.out.println("Enter Teaching parameters of the ArtNeural Network: ");
        Scanner scanner = new Scanner(System.in);
        String str_teachingParam = scanner.nextLine();

        if (debugMode) {
            System.out.println("Your Teaching parameters are " + str_teachingParam);
        }

        String[] teaching_array = str_teachingParam.split(",", -1);
        int epocs = Integer.parseInt(teaching_array[0]);

        //assign learning rate
        eta = Double.parseDouble(teaching_array[1]);
        Double sample_div_factor = Double.parseDouble(teaching_array[2]);

        System.out.println("Enter architecture of the ArtNeural Network: ");
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

        
        //accept Input sample values from user
        System.out.println("Enter number of samples : ");
        String str_number_samples = scanner.nextLine();
        System.out.println(" Number of inputs " + str_number_samples);

        int num_samples = Integer.parseInt(str_number_samples);

        ArrayList<Double> inputVal = new ArrayList<Double>();
        ArrayList<ArrayList<Double>> inputSamples = new ArrayList<ArrayList<Double>>();

        for (int n_sample = 0; n_sample < num_samples; n_sample++) {
            System.out.println("Enter input values for sample " + n_sample);
            inputVal.clear();

            String str_inputVal = scanner.nextLine();

            String[] arr_inputVal = str_inputVal.split(",", -1);

            for (int num_val = 0; num_val < arr_inputVal.length - 1; num_val++) {

                Double input_val = Double.parseDouble(arr_inputVal[num_val]);
                inputVal.add(input_val);
            }

            inputSamples.add(inputVal);
        }

        int trainingSamples = (int) (inputSamples.size() * sample_div_factor);
        int validationSamples = (inputSamples.size() - trainingSamples);

        //Training process
        for (int n_epoc = 0; n_epoc < epocs; n_epoc++) {

            for (int num_trainingSample = 0; num_trainingSample < trainingSamples; num_trainingSample++) {

                ArrayList<Double> currentSample = inputSamples.get(num_trainingSample);

                //output value is not displayed for this part of the solution
                //myNet.showVector(outputVal);
                ArrayList<Double> targetVal = new ArrayList<Double>();

                //take the last value from input vector as target value
                Double input_val = currentSample.get(currentSample.size() - 1);
                targetVal.add(input_val);

                //last value is target value hence removed.
                int num_features = currentSample.size() - 2;
                myNet.forwardPass(currentSample, num_features);

                ArrayList<Double> outputVal = new ArrayList<Double>();

                myNet.getResult(outputVal);

                myNet.backwordPass(targetVal);
            }

        }

        //Validation Process
        for (int num_validationSample = 0; num_validationSample < validationSamples; num_validationSample++) {

            ArrayList<Double> currentSample = inputSamples.get(num_validationSample);

            //output value is not displayed for this part of the solution
            //myNet.showVector(outputVal);
            ArrayList<Double> targetVal = new ArrayList<Double>();

            //take the last value from input vector as target value
            Double input_val = currentSample.get(currentSample.size() - 1);
            targetVal.add(input_val);

            System.out.print(" Expected value : " + input_val);
            //last value is target value hence removed.
            int num_features = currentSample.size() - 2;
            myNet.forwardPass(currentSample, num_features);

            ArrayList<Double> outputVal = new ArrayList<Double>();

            myNet.getResult(outputVal);
            System.out.print(" Output result : ");
            myNet.showVector(outputVal);

        }
    
    }

}

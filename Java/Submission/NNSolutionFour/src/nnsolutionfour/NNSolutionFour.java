/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package nnsolutionfour;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Scanner;
import static nnsolutionfour.ArtNeuralNet.showVector;

/**
 *
 * @author ubuntu-admin
 */
public class NNSolutionFour {

    public static final boolean debugMode = false;
    public static final boolean isWeightInput = true;
    public static double eta = 0.0;

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

//        int testSamples = (int) (totalSamples * ArtNeuralNet.m_fraction_test_samples);
//        int verifySamples = (totalSamples - testSamples);


        ArrayList<ArrayList<Double>> arr_inputVal = new ArrayList<ArrayList<Double>>();
        ArrayList<ArrayList<Double>> arr_targetVal = new ArrayList<ArrayList<Double>>();
//        ArrayList<ArrayList<Double>> arr_resultVal = new ArrayList<ArrayList<Double>>();

        //accept Input sample values from user
        System.out.println("Enter number of samples : ");
        String str_number_samples = scanner.nextLine();
        System.out.println(" Number of samples " + str_number_samples);
        int totalSamples = Integer.parseInt(str_number_samples);

        int total_features = architecture.get(0);

        for (int sample_num = 0; sample_num < totalSamples; sample_num++) {
        ArrayList<Double> inputVal = new ArrayList<Double>();
        ArrayList<Double> targetVal = new ArrayList<Double>();
//        ArrayList<Double> resultVal = new ArrayList<Double>();

            System.out.println("Enter values of sample number : " + sample_num);
            String str_sample = scanner.nextLine();
            String[] array_features = str_sample.split(",", -1);

            for (int feature = 0; feature < total_features; feature++) {
//                    System.out.println(" feature " + feature + " is " + sample[feature]);
                inputVal.add(Double.parseDouble(array_features[feature]));
            }
            targetVal.add(Double.parseDouble(array_features[total_features]));

            arr_inputVal.add(inputVal);
            arr_targetVal.add(targetVal);

        }

        System.err.println(" total samples = " + arr_inputVal.size() + " total targetVal = " + arr_targetVal.size());

        int trainingSamples = (int) (totalSamples * sample_div_factor);

        if (debugMode) {
            for (int n_trSample = 0; n_trSample < totalSamples; n_trSample++) {

                System.out.print(" Input :: ");
                showVector(arr_inputVal.get(n_trSample));

                System.out.print(" Expected :: ");
                showVector(arr_targetVal.get(n_trSample));

            }
        }

        int validationSamples = (totalSamples - trainingSamples);
        for (int epoc = 0; epoc < epocs; epoc++) {

            System.out.println(" training epoc number : "+ epoc);
            for (int n_trSample = 0; n_trSample < trainingSamples; n_trSample++) {
                myNet.trainNetwork(arr_inputVal.get(n_trSample), arr_targetVal.get(n_trSample));
            }
            
            myNet.printRequiredResults();
        }
         //reset error count before you start the verification process
        myNet.error_count = 0;
        
        for (int n_trSample = trainingSamples; n_trSample < totalSamples; n_trSample++) {
            myNet.verifyNetwork(arr_inputVal.get(n_trSample), arr_targetVal.get(n_trSample));
        }

        System.out.println(" Total errors in prediction = " + myNet.error_count);
        
       
    }

}

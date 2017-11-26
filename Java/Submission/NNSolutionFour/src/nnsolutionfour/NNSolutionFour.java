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
        
        /*try {

                for (int epoc = 1; epoc < epocs; epoc++) {

                    if (epoc % 100 == 0) {
                        System.out.println(" Epoc number " + epoc + " RMS error " + myNet.getRMSerr());

                    }

                    int sample_num = 0;
                    br = new BufferedReader(new FileReader(csvFile));

                    // start training
                    while ((line = br.readLine()) != null) {
                        sample_num++;
//                    System.out.println(" *************************** Sample = " + sample_num);

                        //clear all previous inputs and outputs vectors
                        inputVal.clear();
                        targetVal.clear();
                        resultVal.clear();

                        // use comma as separator
                        String[] sample = line.split(cvsSplitBy);
                        for (int feature = 0; feature < 57; feature++) {
//                    System.out.println(" feature " + feature + " is " + sample[feature]);
                            inputVal.add(Double.parseDouble(sample[feature]));
                        }
                        targetVal.add(Double.parseDouble(sample[57]));
//                System.out.println("");

                        myNet.trainNetwork(inputVal, targetVal, resultVal);
                    }

                    if (epoc % 100 == 0) {
                        System.out.println(" Epoc number " + epoc + " RMS error " + myNet.getRMSerr());
                    System.out.println(" Trained for samples = " + sample_num);

                    }
                }

                String csvValidateFile = "/home/ubuntu-admin/SharedFolder/EIT_Studies/BME/Study/IntelligentDataAnalysis/validate.csv";
                BufferedReader brValidate = null;
                String validateline = "";
                String validatecvsSplitBy = ",";

                // verify results 
                int validatesample_num = 0;

                System.out.println(" Total Email samples = " + verifySamples);

                brValidate = new BufferedReader(new FileReader(csvValidateFile));

                // start training
                while ((validateline = brValidate.readLine()) != null) {
                    validatesample_num++;
                    //clear all previous inputs and outputs vectors
                    inputVal.clear();
                    targetVal.clear();
                    resultVal.clear();

                    // use comma as separator
                    String[] sample = validateline.split(cvsSplitBy);
                    for (int feature = 0; feature < 57; feature++) {
//                    System.out.println(" feature " + feature + " is " + sample[feature]);
                        inputVal.add(Double.parseDouble(sample[feature]));
                    }
                    targetVal.add(Double.parseDouble(sample[57]));
//                System.out.println("");

                    myNet.verifyNetwork(inputVal, targetVal, resultVal);
                }

                System.out.println(" Total validation samples = " + validatesample_num);
                System.out.println(" Total Emails mis-classified = " + myNet.error_count);

            } catch (FileNotFoundException e) {
                e.printStackTrace();
            } catch (IOException e) {
                e.printStackTrace();
            } finally {
                if (br != null) {
                    try {
                        br.close();
                    } catch (IOException e) {
                        e.printStackTrace();
                    }
                }
            }*/
 /*
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
                myNet.forwardPass(currentSample);

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
            myNet.forwardPass(currentSample);

            ArrayList<Double> outputVal = new ArrayList<Double>();

            myNet.getResult(outputVal);
            System.out.print(" Output result : ");
            myNet.showVector(outputVal);

        }
        
        
         */
    }

}

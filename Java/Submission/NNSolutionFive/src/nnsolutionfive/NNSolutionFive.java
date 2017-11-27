/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package nnsolutionfive;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Scanner;
import static nnsolutionfive.ArtNeuralNet.showVector;

/**
 *
 * @author ubuntu-admin
 */
public class NNSolutionFive {

    public static final boolean debugMode = false;
    public static final boolean isWeightInput = false;
    public static double eta = 0.0;
    public static Double alpha = 0.15;

    //Delimiter used in CSV file
    private static final String COMMA_DELIMITER = ",";
    private static final String NEW_LINE_SEPARATOR = "\n";
    private static final String FILE_HEADER = "epoc,error_test,error_validate,total_mispredictions";

    /**
     * @param args the command line arguments
     */
    public static void main(String[] args) {
        // TODO code application logic here

        String configFile = "/home/ubuntu-admin/SharedFolder/EIT_Studies/BME/Study/IntelligentDataAnalysis/Assignment/NeuralNetworks/Java/Output/config_spambase.csv";
        BufferedReader br_config = null;
        String line_config = "";
        int config_num = 0;
        int epocs = 0;
        Double sample_div_factor = 0.0;
        ArrayList<Integer> architecture = new ArrayList<Integer>();

        try {

            br_config = new BufferedReader(new FileReader(configFile));

            while ((line_config = br_config.readLine()) != null) {

                config_num++;

                String[] array_config = line_config.split(",", -1);
                epocs = Integer.parseInt(array_config[0]);
                eta = Double.parseDouble(array_config[1]);
                sample_div_factor = Double.parseDouble(array_config[2]);

                architecture.add(Integer.parseInt(array_config[3]));
                architecture.add(Integer.parseInt(array_config[4]));
                architecture.add(Integer.parseInt(array_config[5]));
                architecture.add(Integer.parseInt(array_config[6]));

                System.out.println(" epocs = " + epocs + " : eta = " + eta + " : sample div factor = " + sample_div_factor
                        + " : Architecture " + array_config[3] + ", " + array_config[4] + ", " + array_config[5] + ", " + array_config[6]);

                ArtNeuralNet myNet = new ArtNeuralNet(architecture);

//        int testSamples = (int) (totalSamples * ArtNeuralNet.m_fraction_test_samples);
//        int verifySamples = (totalSamples - testSamples);
                ArrayList<ArrayList<Double>> arr_inputVal = new ArrayList<ArrayList<Double>>();
                ArrayList<ArrayList<Double>> arr_targetVal = new ArrayList<ArrayList<Double>>();

                int total_features = architecture.get(0);

//        System.out.println("Enter absolute path of dataset : ");
//        String str_path = scanner.nextLine();
//        int totalSamples = Integer.parseInt(str_number_samples);
                String csvFile = "/home/ubuntu-admin/SharedFolder/EIT_Studies/BME/Study/IntelligentDataAnalysis/spambase_train.csv";
                BufferedReader br = null;
                String line = "";
                int sample_num = 0;

                try {

                    br = new BufferedReader(new FileReader(csvFile));

                    while ((line = br.readLine()) != null) {
                        sample_num++;
                        ArrayList<Double> inputVal = new ArrayList<Double>();
                        ArrayList<Double> targetVal = new ArrayList<Double>();
//        ArrayList<Double> resultVal = new ArrayList<Double>();

                        String[] array_features = line.split(",", -1);

                        for (int feature = 0; feature < total_features; feature++) {
//                    System.out.println(" feature " + feature + " is " + sample[feature]);
                            inputVal.add(Double.parseDouble(array_features[feature]));
                        }
                        targetVal.add(Double.parseDouble(array_features[total_features]));

                        arr_inputVal.add(inputVal);
                        arr_targetVal.add(targetVal);

                    }

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
                }

                /**
                 * *********************************write to CSV file
                 * *********************************************
                 */
                FileWriter fileWriter = null;
                try {

                    String filename = "/home/ubuntu-admin/SharedFolder/EIT_Studies/BME/Study/IntelligentDataAnalysis/Assignment/NeuralNetworks/Java/Output/";
                    filename = filename + "res_config" + Integer.toString(config_num) + ".csv";

                    fileWriter = new FileWriter(filename);

                    String hyperParam = " epocs = " + array_config[0] + " : eta = " + array_config[1] + " : sample div factor = " + array_config[2]
                            + " : Architecture " + array_config[3] + " : " + array_config[4] + " : " + array_config[5] + " : " + array_config[6];

                    fileWriter.append(hyperParam);
                    fileWriter.append(NEW_LINE_SEPARATOR);

                    //Write the CSV file header
                    fileWriter.append(FILE_HEADER);
                    //Add a new line separator after the header
                    fileWriter.append(NEW_LINE_SEPARATOR);

                    int totalSamples = sample_num;
                    int trainingSamples = (int) (totalSamples * sample_div_factor);
                    int validationSamples = (totalSamples - trainingSamples);

                    for (int epoc = 0; epoc < epocs; epoc++) {

                        //Write a epoc to the CSV file
                        fileWriter.append(Integer.toString(epoc));
                        fileWriter.append(COMMA_DELIMITER);

                        System.out.println(" training epoc number : " + epoc);
                        myNet.m_batch_error = 0.0;

                        for (int n_trSample = 0; n_trSample < trainingSamples; n_trSample++) {
                            myNet.trainNetwork(arr_inputVal.get(n_trSample), arr_targetVal.get(n_trSample));
                        }

                        //test RMS error
                        fileWriter.append(Double.toString(myNet.m_batch_error));
                        fileWriter.append(COMMA_DELIMITER);

                        //reset error count before you start the verification process
                        myNet.misPredictCount = 0;
                        myNet.m_batch_valid_error = 0.0;

                        for (int n_trSample = trainingSamples; n_trSample < totalSamples; n_trSample++) {
                            myNet.verifyNetwork(arr_inputVal.get(n_trSample), arr_targetVal.get(n_trSample));
                        }

                        //validation RMS error
                        fileWriter.append(Double.toString(myNet.m_batch_valid_error));
                        fileWriter.append(COMMA_DELIMITER);

                        //total mispredictions
                        fileWriter.append(Double.toString(myNet.misPredictCount));
                        fileWriter.append(COMMA_DELIMITER);
                        fileWriter.append(NEW_LINE_SEPARATOR);

                        System.out.println(" tr smpl : " + trainingSamples + " : valid smpls : " + validationSamples
                                + " tr err : " + myNet.m_batch_error + " valid err : " + myNet.m_batch_valid_error + " mispred : " + myNet.misPredictCount);

                        myNet.printRequiredResults();

                    }

                } catch (Exception e) {
                    System.out.println("Error in CsvFileWriter !!!");
                    e.printStackTrace();
                } finally {
                    try {
                        fileWriter.flush();
                        fileWriter.close();
                    } catch (IOException e) {
                        System.out.println("Error while flushing/closing fileWriter !!!");
                        e.printStackTrace();
                    }
                }

            }
            /**
             * **********************************************************************************************
             */
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        } catch (IOException e) {
            e.printStackTrace();
        } finally {
            if (br_config != null) {
                try {
                    br_config.close();
                } catch (IOException e) {
                    e.printStackTrace();
                }
            }
        }

    }
}

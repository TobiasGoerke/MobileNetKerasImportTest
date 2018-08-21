package de.goerke.tobias;

import org.datavec.image.loader.Java2DNativeImageLoader;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.modelimport.keras.KerasModelImport;
import org.deeplearning4j.nn.modelimport.keras.exceptions.InvalidKerasConfigurationException;
import org.deeplearning4j.nn.modelimport.keras.exceptions.UnsupportedKerasConfigurationException;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.nd4j.linalg.factory.Nd4j;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.net.URL;
import java.util.Arrays;


public class Main {

    public static void main(String[] args) {
        File dogImage = loadResource("dog.jpg");
        File flowerImage = loadResource("flower.jpg");
        //File xception = loadResource("xception.h5");
        File testResourcesModel = loadResource("original.h5");
        File alternative = loadResource("alternative.hdf5");
        File kerasExportV1 = loadResource("mobileNet.h5");
        File kerasExportV2 = loadResource("mobileNetV2.h5");

        // Tested e.g. xception works this way. Not included due to file-size
        //testModel(image, xception);

        /*
         * SEE HERE
         *
         * Both exported models throw an UnsupportedKerasConfigurationException:
         * Unsupported keras layer type ReLU. Please file an issue at http://github.com/deeplearning4j/deeplearning4j/issues.
         */
        testModel(dogImage, flowerImage, kerasExportV1);
        testModel(dogImage, flowerImage, kerasExportV2);

        // Alternative model works. However, output array is always the same so probably not a functioning model, too.
        // UPDATE: See example, both images result in [[0.9998, 0.0002]]. THIS MODEL DOES NOT WORK, TOO!
        testModel(dogImage, flowerImage, alternative);

        // Test model throws an error
        testModel(dogImage, flowerImage, testResourcesModel);
    }

    private static void testModel(File imageFile, File secondImageFile, File modelFile) {
        String errorMessage;
        ComputationGraph restoredCNN;
        int tileSize = 224;
        try {
            restoredCNN = KerasModelImport.importKerasModelAndWeights(modelFile.getAbsolutePath(), new int[]{tileSize, tileSize, 3}, false);
        } catch (InvalidKerasConfigurationException ex) {
            errorMessage = "Could not load CNN model: " + ex.getMessage() + "  Cause:  " + ex.getCause();
            System.out.println(errorMessage);
            return;
        } catch (UnsupportedKerasConfigurationException ex) {
            errorMessage = "Could not load CNN model: " + ex.getMessage();
            System.out.println(errorMessage);
            return;
        } catch (IOException ex) {
            errorMessage = "Could not load CNN model: IO Exception";
            System.out.println(errorMessage);
            return;
        }
        if (restoredCNN == null) {
            errorMessage = "CNN model is not valid";
            System.out.println(errorMessage);
            return;
        }

        try {
            System.out.println("*******************************************");
            System.out.println("*******************************************");
            System.out.println("*******************************************");
            System.out.println("Outputting sample predictions for File " + modelFile.getAbsolutePath());

            BufferedImage image = ImageIO.read(imageFile);
            DataNormalization scaler = new ImagePreProcessingScaler(-1.0, 1.0);
            Java2DNativeImageLoader loader = new Java2DNativeImageLoader(tileSize, tileSize, 3);

            INDArray indArray1 = loader.asMatrix(image);
            scaler.transform(indArray1);
            INDArray[] output1 = restoredCNN.output(false, indArray1);
            System.out.println("Output image 1: " + Arrays.toString(output1));

            INDArray indArray2 = loader.asMatrix(image);
            scaler.transform(indArray2);
            INDArray[] output2 = restoredCNN.output(false, indArray2);
            System.out.println("Output image 2: " + Arrays.toString(output2));

            INDArray[] outputOnesOnly = restoredCNN.output(Nd4j.ones(10, 3, 299, 299));
            INDArray[] outputZeroesOnly = restoredCNN.output(Nd4j.zeros(10, 3, 299, 299));
            System.out.println("Output Ones Only: " + Arrays.toString(outputOnesOnly));
            System.out.println("Output Zeroes Only: " + Arrays.toString(outputZeroesOnly));
        } catch (IOException ex) {
            errorMessage = "Error Loading File";
            System.out.println(errorMessage);
        }
    }

    private static File loadResource(String fileName) {
        try {
            URL url = ClassLoader.getSystemResource(fileName);
            return new File(url.toURI());
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }
}


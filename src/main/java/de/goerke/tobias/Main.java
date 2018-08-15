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


public class Main {

    public static void main(String[] args) {
        File image = loadResource("dog.jpg");
        //File xception = loadResource("xception.h5");
        File testResourcesModel = loadResource("original.h5");
        File alternative = loadResource("alternative.hdf5");

        // Tested e.g. xception works this way. Not included due to file-size
        //testModel(image, xception);

        // Alternative model works. However, output array is always the same so probably not a functioning model, too.
        testModel(image, alternative);

        // Test model throws an error
        testModel(image, testResourcesModel);
    }

    private static void testModel(File imageFile, File modelFile) {
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
            BufferedImage image = ImageIO.read(imageFile);
            DataNormalization scaler = new ImagePreProcessingScaler(-1.0, 1.0);
            Java2DNativeImageLoader loader = new Java2DNativeImageLoader(tileSize, tileSize, 3);

            INDArray indArray = loader.asMatrix(image);
            scaler.transform(indArray);

            INDArray output = restoredCNN.outputSingle(false, indArray);

            System.out.println(output);
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


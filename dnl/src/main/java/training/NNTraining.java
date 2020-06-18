package training;


import org.deeplearning4j.datasets.iterator.impl.EmnistDataSetIterator;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ops.impl.transforms.pairwise.Set;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;

public class NNTraining {
	public static void main(String[] args) {
//		
//		int batchSize = 16; // how many examples to simultaneously train in the network
//		org.deeplearning4j.datasets.iterator.impl.EmnistDataSetIterator.Set emnistSet = EmnistDataSetIterator.Set.BALANCED;
//				EmnistDataSetIterator emnistTrain = new EmnistDataSetIterator(emnistSet, batchSize, true);
//				EmnistDataSetIterator emnistTest = new EmnistDataSetIterator(emnistSet, batchSize, false);
//		int outputNum = EmnistDataSetIterator.numLabels(emnistSet); // total output classes
//				int rngSeed = 123; // integer for reproducability of a random number generator
//				int numRows = 33; // number of "pixel rows" in an mnist digit
//				int numColumns = 9;
//
//				NeuralNetConfiguration conf = new NeuralNetConfiguration.Builder()
//				            .seed(rngSeed)
//				            .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
//				            .updater(new Adam())
//				            .l2(1e-4)
//				            .list()
//				            .layer(new DenseLayer.Builder()
//				                .nIn(numRows * numColumns) // Number of input datapoints.
//				                .nOut(1000) // Number of output datapoints.
//				                .activation(Activation.RELU) // Activation function.
//				                .weightInit(WeightInit.XAVIER) // Weight initialization.
//				                .build())
//				            .layer(new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
//				                .nIn(1000)
//				                .nOut(outputNum)
//				                .activation(Activation.SOFTMAX)
//				                .weightInit(WeightInit.XAVIER)
//				                .build()).build();
	}
}

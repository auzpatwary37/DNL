package training;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

public class testMain {
public static void main(String[] args) {
	INDArray errorBPR=Nd4j.readTxt("Network/ND/ModelBPR/averagePredictionError.txt");
	INDArray errorMean=Nd4j.readTxt("Network/ND/ModelNormal/averagePredictionError.txt");
	INDArray errorDiff=errorMean.sub(errorBPR);
	System.out.println(errorDiff);
	System.out.println(errorDiff.sumNumber().doubleValue()/(33*9));
}
}

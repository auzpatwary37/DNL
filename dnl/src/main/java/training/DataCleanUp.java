package training;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Random;

import org.matsim.core.utils.collections.Tuple;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;

import kriging.KrigingInterpolator;

public class DataCleanUp {
	public static void main(String[] args) {
		List<Tuple<INDArray,INDArray>> datasetFull=new ArrayList<>();
		for(int i=0;i<6;i++) {
			
			for(Tuple<INDArray,INDArray> data:DataIO.readDataSet("Network/ND/DataSet"+i+".txt").values()) {
				boolean isDuplicate=false;
				for(Tuple<INDArray,INDArray> currentData:datasetFull) {
					if(data.getFirst().sub(currentData.getFirst()).norm2Number().doubleValue()<1*33*9) {
						isDuplicate=true;
						break;
					}
				}
				if(isDuplicate==false) {
					datasetFull.add(data);
				}
			}
		}
		
		DataSet dataset=new DataSet();
//		INDArray X=
//		
//		for(Tuple<INDArray,INDArray> data:datasetFull) {
//			
//		}
		
		DataIO.writeData(datasetFull, "Network/ND/DataSetNDFull.txt");
		TestAndTrainData testAndTrain=DataCleanUp.DevideDataInTestAndTrain(datasetFull, 0.1);
		DataIO.writeData(testAndTrain.getTestData(), "Network/ND/DataSetNDTest.txt");
		DataIO.writeData(testAndTrain.getTrainData(), "Network/ND/DataSetNDTrain.txt");
		
		
	}
	
	/**
	 * 
	 * @param fullData
	 * @param testRatio proportion of testing (0,1)
	 * @return
	 */
	public static TestAndTrainData DevideDataInTestAndTrain(Map<Integer,Tuple<INDArray,INDArray>>fullData,double testRatio) {
		return new TestAndTrainData(fullData,testRatio);
	}
	public static TestAndTrainData DevideDataInTestAndTrain(List<Tuple<INDArray,INDArray>>fullData,double testRatio) {
		return new TestAndTrainData(fullData,testRatio);
	}
}



class TestAndTrainData{
	private Map<Integer,Tuple<INDArray,INDArray>> testData=new HashMap<>();
	private Map<Integer,Tuple<INDArray,INDArray>> trainData=new HashMap<>();
	private double testRatio=0.1;
	public TestAndTrainData(Map<Integer,Tuple<INDArray,INDArray>>fullData,double testRatio) {
		List<Integer> testNumbers = new ArrayList<Integer>();
		if(testRatio<1 && testRatio>0) {
			this.testRatio=testRatio;
		}
		double  numberOfNumbersYouWant = fullData.size()*this.testRatio; // This has to be less than 11
		Random random = new Random();

		do
		{
			int next = random.nextInt(fullData.size()-1);
			if (!testNumbers.contains(next))
			{
				testNumbers.add(next);
			}
		} while (testNumbers.size() < numberOfNumbersYouWant);

		int testData=0;
		int trainData=0;

		for(Entry<Integer, Tuple<INDArray, INDArray>> dataPoint:fullData.entrySet()) {
			if(testNumbers.contains(dataPoint.getKey())) {
				this.testData.put(testData, dataPoint.getValue());
				testData++;
			}else {
				this.trainData.put(trainData, dataPoint.getValue());
				trainData++;
			}
		}
	    
	}
	public TestAndTrainData(List<Tuple<INDArray,INDArray>>fullDataList,double testRatio) {
		Map<Integer,Tuple<INDArray,INDArray>>fullData=new HashMap<>();
		for(int i=0;i<fullDataList.size();i++) {
			fullData.put(i,fullDataList.get(i));
		}
		List<Integer> testNumbers = new ArrayList<Integer>();
		if(testRatio<1 && testRatio>0) {
			this.testRatio=testRatio;
		}
		double  numberOfNumbersYouWant = fullData.size()*this.testRatio; // This has to be less than 11
		Random random = new Random();

		do
		{
			int next = random.nextInt(fullData.size()-1);
			if (!testNumbers.contains(next))
			{
				testNumbers.add(next);
			}
		} while (testNumbers.size() < numberOfNumbersYouWant);

		int testData=0;
		int trainData=0;

		for(Entry<Integer, Tuple<INDArray, INDArray>> dataPoint:fullData.entrySet()) {
			if(testNumbers.contains(dataPoint.getKey())) {
				this.testData.put(testData, dataPoint.getValue());
				testData++;
			}else {
				this.trainData.put(trainData, dataPoint.getValue());
				trainData++;
			}
		}
	}
	public Map<Integer, Tuple<INDArray, INDArray>> getTestData() {
		return testData;
	}
	public Map<Integer, Tuple<INDArray, INDArray>> getTrainData() {
		return trainData;
	}
	
	//TODO:identify and remove duplicate data
}
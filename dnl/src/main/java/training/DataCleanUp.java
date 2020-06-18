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

import kriging.Data;
import kriging.KrigingInterpolator;

public class DataCleanUp {
	public static void main(String[] args) {
		List<Data> datasetFull=new ArrayList<>();
		String baseloc="DataSet/";
		for(int i=0;i<27;i++) {
			Map<Integer,Data> dset=DataIO.readDataSet(baseloc+"DataSet"+i+".txt",baseloc+"KeySet"+i+".csv");
			for(Data data:dset.values()) {
				boolean isDuplicate=false;
				for(Data currentData:datasetFull) {
					if(data.getX().sub(currentData.getX()).lt(1).all()) {
						isDuplicate=true;
						break;
					}
				}
				if(isDuplicate==false) {
					datasetFull.add(data);
				}
			}
		}
		

//		INDArray X=
//		
//		for(Tuple<INDArray,INDArray> data:datasetFull) {
//			
//		}
		DataIO.writeData(datasetFull, baseloc+"DataSetFull.txt",baseloc+"KeySetFull.csv");
		//for(double i=.30;i<=.90;i=i+.10) {
		
		TestAndTrainData testAndTrain=DataCleanUp.DevideDataInTestAndTrain(datasetFull, (int)800);
		DataIO.writeData(testAndTrain.getTestData(), baseloc+"DataSetTest"+800+".txt",baseloc+"KeySetTest"+800+".csv");
		DataIO.writeData(testAndTrain.getTrainData(), baseloc+"DataSetTrain"+800+".txt", baseloc+"KeySetTrain"+800+".csv");
		//}
		
		
	}
	
	/**
	 * 
	 * @param fullData
	 * @param testRatio proportion of testing (0,1)
	 * @return
	 */
	public static TestAndTrainData DevideDataInTestAndTrain(Map<Integer,Data>fullData,double testRatio) {
		return new TestAndTrainData(fullData,testRatio);
	}
	public static TestAndTrainData DevideDataInTestAndTrain(List<Data>fullData,int trainDataSize) {
		return new TestAndTrainData(fullData,trainDataSize);
	}
}



class TestAndTrainData{
	private Map<Integer,Data> testData=new HashMap<>();
	private Map<Integer,Data> trainData=new HashMap<>();
	private double testRatio=0.1;
	private int trainDataSize=100;
	public TestAndTrainData(Map<Integer,Data>fullData,double testRatio) {
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

		for(Entry<Integer, Data> dataPoint:fullData.entrySet()) {
			if(testNumbers.contains(dataPoint.getKey())) {
				this.testData.put(testData, dataPoint.getValue());
				testData++;
			}else {
				this.trainData.put(trainData, dataPoint.getValue());
				trainData++;
			}
		}
	    
	}
	public TestAndTrainData(List<Data>fullDataList,double testRatio) {
		Map<Integer,Data>fullData=new HashMap<>();
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

		for(Entry<Integer, Data> dataPoint:fullData.entrySet()) {
			if(testNumbers.contains(dataPoint.getKey())) {
				this.testData.put(testData, dataPoint.getValue());
				testData++;
			}else {
				this.trainData.put(trainData, dataPoint.getValue());
				trainData++;
			}
		}
	}
	
	public TestAndTrainData(List<Data>fullDataList,int trainDataSize) {
		Map<Integer,Data>fullData=new HashMap<>();
		for(int i=0;i<fullDataList.size();i++) {
			fullData.put(i,fullDataList.get(i));
		}
		List<Integer> testNumbers = new ArrayList<Integer>();
		if(trainDataSize<fullData.size()) {
			this.trainDataSize=trainDataSize;
		}
		double  numberOfNumbersYouWant = fullData.size()-this.trainDataSize; // This has to be less than 11
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

		for(Entry<Integer, Data> dataPoint:fullData.entrySet()) {
			if(testNumbers.contains(dataPoint.getKey())) {
				this.testData.put(testData, dataPoint.getValue());
				testData++;
			}else {
				this.trainData.put(trainData, dataPoint.getValue());
				trainData++;
			}
		}
	}
	public Map<Integer, Data> getTestData() {
		return testData;
	}
	public Map<Integer, Data> getTrainData() {
		return trainData;
	}
	
	//TODO:identify and remove duplicate data
}
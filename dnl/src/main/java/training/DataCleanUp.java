package training;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Random;

import org.matsim.core.utils.collections.Tuple;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;

import kriging.Data;
import kriging.KrigingInterpolator;

public class DataCleanUp {
	public static void main(String[] args) {
		List<Data> datasetFull=new ArrayList<>();
		String baseloc="Network/ND/dataset_June2020/";
		RouteData routeData = DataIO.readRouteData(baseloc, 26);
		for(int i=0;i<26;i++) {
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
					//add the routeData to the data
					long T =data.getX().shape()[1];
					INDArray routeDemand = Nd4j.create(routeData.R,T);
					
					for(int r =0;r<routeData.R;r++) {
						for(int t =0;t<T;t++) {
							double demand = 0.;
							try {
								 demand = routeData.routeDemand.get(data.getKey()).get(t).get(routeData.routeList.get(r));
							}catch(Exception e) {
								if(routeData.routeDemand.get(data.getKey()).get(t)==null) {
									
								}else if(routeData.routeDemand.get(data.getKey()).get(t).get(routeData.routeList.get(r))==null) {
									
								}
								else {
									System.out.println("DEbug");
								}
							}
							routeDemand.putScalar(r, t, demand);
						}
					}
					data.setR(routeDemand);
					datasetFull.add(data);
				}
			}
		}
		

//		INDArray X=
//		
//		for(Tuple<INDArray,INDArray> data:datasetFull) {
//			
//		}
		DataIO.writeData(datasetFull, baseloc+"DataSetFull.txt",baseloc+"KeySetFull.csv",baseloc+"RouteSetFull.txt");
		//for(double i=.30;i<=.90;i=i+.10) {
		routeData.writeRouteDetails(baseloc+"routeInfo.csv");
		TestAndTrainData testAndTrain=DataCleanUp.DevideDataInTestAndTrain(datasetFull, (int)800);
		DataIO.writeData(testAndTrain.getTestData(), baseloc+"DataSetTest"+800+".txt",baseloc+"KeySetTest"+800+".csv",baseloc+"RouteSetTest"+800+".txt");
		DataIO.writeData(testAndTrain.getTrainData(), baseloc+"DataSetTrain"+800+".txt", baseloc+"KeySetTrain"+800+".csv",baseloc+"RouteSetTrain"+800+".txt");
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
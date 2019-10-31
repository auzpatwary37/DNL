package training;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import kriging.Data;

public class FarthestPointSampler {

	public static List<Integer> pickKFurthestPoint(INDArray distanceMatrix,int k) {
		List<Integer> outIndex=new ArrayList<>();
		INDArray distance=Nd4j.create(k,distanceMatrix.shape()[1]);
		int colInd=distanceMatrix.max(0).max(1).toIntVector()[0];
		int[] rowIndex=distanceMatrix.argMax(0).toIntVector();
		int rowInd=rowIndex[colInd];
		outIndex.add(rowInd);
		outIndex.add(colInd);
		distance.putRow(0, distanceMatrix.getRow(rowInd));
		distance.putRow(1, distanceMatrix.getRow(colInd));
		distance.getColumn(rowInd).muli(Double.NEGATIVE_INFINITY);
		distance.getColumn(colInd).muli(Double.NEGATIVE_INFINITY);
		
		for(int i=2;i<k;i++) {
			int newInd=distance.min(0).max(1).toIntVector()[0];
			outIndex.add(newInd);
			distance.putRow(i, distanceMatrix.getRow(newInd));
			distance.getColumn(newInd).muli(Double.NEGATIVE_INFINITY);
		}
//		INDArray newDistanceMatrix=Nd4j.create(k,k);
//		for(int i=0;i<outIndex.size();i++) {
//			newDistanceMatrix.put(i, i, 0);
//			for(int j=0;j<i;j++) {
//				newDistanceMatrix.put(i, j, distanceMatrix.getDouble(outIndex.get(i),outIndex.get(j)));
//				newDistanceMatrix.put(j, i, distanceMatrix.getDouble(outIndex.get(i),outIndex.get(j)));
//			}
//		}
		return outIndex;
	}
	
	public static List<Integer> pickAdditionalKFurthestPoint(INDArray distanceMatrix,List<Integer>outIndex,int k){
		INDArray distance=Nd4j.create(outIndex.size()+k,distanceMatrix.shape()[1]);
		for(int i=0;i<outIndex.size();i++) {
			distance.putRow(i, distanceMatrix.getRow(outIndex.get(i)));
			distance.getColumn(outIndex.get(i)).muli(Double.NEGATIVE_INFINITY);
		}
		for(int i=0;i<k;i++) {
			int newInd=distance.min(0).max(1).toIntVector()[0];
			outIndex.add(newInd);
			int index=outIndex.size()-1;
			distance.putRow(index, distanceMatrix.getRow(newInd));
			distance.getColumn(newInd).muli(Double.NEGATIVE_INFINITY);
		}
		
		return outIndex;
	}
	
	public static List<Integer> pickAdditionalKFurthestPoint(INDArray distanceMatrix,List<Integer>outIndex,int k,INDArray matrixMultiplier){
		INDArray distance=Nd4j.create(outIndex.size()+k,distanceMatrix.shape()[1]);
		distanceMatrix.muli(matrixMultiplier);
		for(int i=0;i<outIndex.size();i++) {
			distance.putRow(i, distanceMatrix.getRow(outIndex.get(i)));
			distance.getColumn(outIndex.get(i)).muli(Double.NEGATIVE_INFINITY);
		}
		for(int i=0;i<k;i++) {
			int newInd=distance.min(0).max(1).toIntVector()[0];
			outIndex.add(newInd);
			int index=outIndex.size()-1;
			distance.putRow(index, distanceMatrix.getRow(newInd));
			distance.getColumn(newInd).muli(Double.NEGATIVE_INFINITY);
		}
		
		return outIndex;
	}
	
	public static List<Integer> pickAdditionalKFurthestPoint(INDArray distanceMatrix,List<Integer>outIndex,int k,Map<Integer,Double>errorMultiplier){
		INDArray distance=Nd4j.create(outIndex.size()+k,distanceMatrix.shape()[1]);
		for(int i=0;i<outIndex.size();i++) {
			distance.putRow(i, distanceMatrix.getRow(outIndex.get(i)));
			distance.getColumn(outIndex.get(i)).muli(Double.NEGATIVE_INFINITY);
		}
		for(int i=0;i<k;i++) {
			INDArray minDistances=distance.min(0);
			for(Entry<Integer, Double> d:errorMultiplier.entrySet()) {
				minDistances.put(0, d.getKey(),minDistances.getDouble(0,d.getKey())*d.getValue());
			}
			int newInd=minDistances.max(1).toIntVector()[0];
			outIndex.add(newInd);
			int index=outIndex.size()-1;
			distance.putRow(index, distanceMatrix.getRow(newInd));
			distance.getColumn(newInd).muli(Double.NEGATIVE_INFINITY);
		}
		
		return outIndex;
	}
	
	public static void main(String[] args) {
		List<Data> datasetFull=new ArrayList<>();
		String baseloc="Network/ND/newLargeDataSet/";
		for(int i=0;i<27;i++) {
			Map<Integer,Data> dset=DataIO.readDataSet(baseloc+"DataSet"+i+".txt",baseloc+"KeySet"+i+".csv");
			for(Data data:dset.values()) {
				datasetFull.add(data);
			}
		}
		
		DataIO.writeData(datasetFull, baseloc+"DataSetFull.txt",baseloc+"KeySetFull.csv");
		//for(double i=.30;i<=.90;i=i+.10) {
		
		TestAndTrainData testAndTrain=DataCleanUp.DevideDataInTestAndTrain(datasetFull, (int)600);
		testAndTrain.equals(null);
	}
	
}

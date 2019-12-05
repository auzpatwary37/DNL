package training;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import kriging.Data;

public class FarthestPointSampler {

	/**
	 * Picks k farthest points in a set of points, where the distance matrix are given
	 * @param distanceMatrix
	 * @param k
	 * @return list with the index of the picked points
	 */
	public static List<Integer> pickKFurthestPoint(INDArray distanceMatrix,int k) {
		List<Integer> outIndex=new ArrayList<>();
		int row=k;
		int col=Math.toIntExact(distanceMatrix.shape()[1]);
		INDArray distance=null;
		try {
			double[][] dis=new double[row][col];
			for(int i=0;i<row;i++) {
				for(int j=0;j<col;j++) {
					dis[i][j]=Double.MAX_VALUE;
				}
			}
			distance=Nd4j.create(dis);
			//distance.muli(100000000);
		}catch(Exception e) {
			System.out.println(e);
		}
		
		int colInd=distanceMatrix.max(0).argMax().toIntVector()[0];
		int[] rowIndex=distanceMatrix.argMax(0).toIntVector();
		int rowInd=rowIndex[colInd];
		outIndex.add(rowInd);
		outIndex.add(colInd);
		distance.putRow(0, distanceMatrix.getRow(rowInd));
		distance.putRow(1, distanceMatrix.getRow(colInd));
		distance.getColumn(rowInd).muli(Double.NEGATIVE_INFINITY);
		distance.getColumn(colInd).muli(Double.NEGATIVE_INFINITY);
		
		for(int i=2;i<=k;i++) {
			int newInd=distance.min(0).argMax().toIntVector()[0];
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
	
	/**
	 * Adds k additional points to a set of points indexed as outIndex from a set of points whose distance matrix are given
	 * @param distanceMatrix
	 * @param outIndex
	 * @param k
	 * @return outIndex with additional k dimension
	 */
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
	
	/**
	 *  Adds k additional points to a set of points indexed as outIndex from a set of points whose distance matrix are given
	 *  the matrix multiplier is multiplied to the distance matrix prior to this operation. The original distance matrix will not be changed
	 * @param distanceMatrix
	 * @param outIndex
	 * @param k
	 * @param matrixMultiplier
	 * @return
	 */
	public static List<Integer> pickAdditionalKFurthestPoint(INDArray distanceMatrix,List<Integer>outIndex,int k,INDArray matrixMultiplier){
		INDArray distance=Nd4j.create(outIndex.size()+k,distanceMatrix.shape()[1]);
		INDArray distanceMatrix1=distanceMatrix.mul(matrixMultiplier);
		for(int i=0;i<outIndex.size();i++) {
			distance.putRow(i, distanceMatrix1.getRow(outIndex.get(i)));
			distance.getColumn(outIndex.get(i)).muli(Double.NEGATIVE_INFINITY);
		}
		for(int i=0;i<k;i++) {
			int newInd=distance.min(0).max(1).toIntVector()[0];
			outIndex.add(newInd);
			int index=outIndex.size()-1;
			distance.putRow(index, distanceMatrix1.getRow(newInd));
			distance.getColumn(newInd).muli(Double.NEGATIVE_INFINITY);
		}
		
		return outIndex;
	}
	
	/**
	 * Adds k additional points to a set of points indexed as outIndex from a set of points whose distance matrix are given
	 * the error multiplier is multiplied to the distance matrix prior to this operation. The original distance matrix will not be changed
	 * 
	 * @param distanceMatrix
	 * @param outIndex
	 * @param k
	 * @param errorMultiplier: index - error mapping a subset of the original distance matrix points
	 * @return
	 */
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
		
		TestAndTrainData testAndTrain=DataCleanUp.DevideDataInTestAndTrain(datasetFull, (int)800);
		DataIO.writeData(testAndTrain.getTestData(), baseloc+"DataSetTest"+800+".txt",baseloc+"KeySetTest"+800+".csv");
		DataIO.writeData(testAndTrain.getTrainData(), baseloc+"DataSetTrain"+800+".txt", baseloc+"KeySetTrain"+800+".csv");
	}
	
}

package training;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.concurrent.ConcurrentHashMap;

import org.matsim.core.utils.collections.Tuple;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.INDArrayIndex;
import org.nd4j.linalg.indexing.NDArrayIndex;

import kriging.Data;

public class DataIO {
	private final Map<Integer,Tuple<INDArray,INDArray>> dataSet;
	private final int N;
	private final int T;
	private final int I;
	public DataIO(Map<Integer,Tuple<INDArray,INDArray>> dataSet) {
		this.dataSet=dataSet;
		this.N=Math.toIntExact(this.dataSet.get(0).getFirst().size(0));
		this.T=Math.toIntExact(this.dataSet.get(0).getFirst().size(1));
		this.I=this.dataSet.size();
	}
	public DataIO(String fileLoc) {
		this.dataSet=readDataSet(fileLoc);
		this.N=Math.toIntExact(this.dataSet.get(0).getFirst().size(0));
		this.T=Math.toIntExact(this.dataSet.get(0).getFirst().size(1));
		this.I=this.dataSet.size();
	}
	
	public static void writeData(Map<Integer,Tuple<INDArray,INDArray>> dataSet,String fileLoc) {
		int N=Math.toIntExact(dataSet.get(0).getFirst().size(0));
		int T=Math.toIntExact(dataSet.get(0).getFirst().size(1));
		int I=dataSet.size();
		INDArray rawArray=Nd4j.create(new int[] {N,2*T,I});
		int i=0;
		for(Entry<Integer,Tuple<INDArray,INDArray>>dataPoint:dataSet.entrySet()) {
			INDArray joinedArray=Nd4j.concat(1, dataPoint.getValue().getFirst(),dataPoint.getValue().getSecond());
			rawArray.put(new INDArrayIndex[] {NDArrayIndex.all(),NDArrayIndex.all(),NDArrayIndex.point(i)},joinedArray);
			i++;
		}
		Nd4j.writeTxt(rawArray,fileLoc);
	}
	
	public static void writeData(Map<Integer,Data> dataSet,String fileLoc,String keyFileloc) {
		int N=Math.toIntExact(dataSet.get(0).getX().size(0));
		int T=Math.toIntExact(dataSet.get(0).getX().size(1));
		int I=dataSet.size();
		INDArray rawArray=Nd4j.create(new int[] {N,2*T,I});
		int i=0;
		try {
			FileWriter fw =new FileWriter(new File(keyFileloc));
			fw.append("NumberId,dataKey\n");
			
		for(Entry<Integer,Data>dataPoint:dataSet.entrySet()) {
			INDArray joinedArray=Nd4j.concat(1, dataPoint.getValue().getX(),dataPoint.getValue().getY());
			rawArray.put(new INDArrayIndex[] {NDArrayIndex.all(),NDArrayIndex.all(),NDArrayIndex.point(i)},joinedArray);
			fw.append(dataPoint.getKey()+","+dataPoint.getValue().getKey()+"\n");
			i++;
		}
		
		Nd4j.writeTxt(rawArray,fileLoc);
		fw.flush();
		fw.close();
		
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}
	
	
	public static void writeData(List<Data> dataSet,String fileLoc,String keyfileloc) {
		int N=Math.toIntExact(dataSet.get(0).getX().size(0));
		int T=Math.toIntExact(dataSet.get(0).getX().size(1));
		int I=dataSet.size();
		INDArray rawArray=Nd4j.create(new int[] {N,2*T,I});
		int i=0;
		try {
			FileWriter fw =new FileWriter(new File(keyfileloc));
			fw.append("NumberId,dataKey\n");
		for(Data dataPoint:dataSet) {
			INDArray joinedArray=Nd4j.concat(1, dataPoint.getX(),dataPoint.getY());
			rawArray.put(new INDArrayIndex[] {NDArrayIndex.all(),NDArrayIndex.all(),NDArrayIndex.point(i)},joinedArray);
			fw.append(i+","+dataPoint.getKey()+"\n");
			i++;
		}
		Nd4j.writeTxt(rawArray,fileLoc);
		fw.flush();
		fw.close();
		
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}
	
	
	public void writeData(String fileLoc) {
		DataIO.writeData(this.dataSet, fileLoc);
	}
	
	public Map<Integer,Tuple<INDArray,INDArray>> getDataSet(){
		return this.dataSet;
	}
	
	public static Map<Integer,Tuple<INDArray,INDArray>> readDataSet(String fileLoc){
		Map<Integer,Tuple<INDArray,INDArray>> dataSet=new ConcurrentHashMap<>();
		INDArray rawArray=Nd4j.readTxt(fileLoc);
		int T=Math.toIntExact(rawArray.size(1))/2;
		int I=Math.toIntExact(rawArray.size(2));
		
		for(int i=0;i<I;i++) {
			
			INDArray X=rawArray.get(new INDArrayIndex[] {NDArrayIndex.all(),NDArrayIndex.interval(0, T),NDArrayIndex.point(i)});
			INDArray Y=rawArray.get(new INDArrayIndex[] {NDArrayIndex.all(),NDArrayIndex.interval(T, 2*T),NDArrayIndex.point(i)});
			dataSet.put(i,new Tuple<INDArray,INDArray>(X,Y));
			
		}
		
		return dataSet;
	}
	
	public static Map<Integer,Data> readDataSet(String fileLoc,String keyFileloc){
		Map<Integer,Data> dataSet=new ConcurrentHashMap<>();
		INDArray rawArray=Nd4j.readTxt(fileLoc);
		int T=Math.toIntExact(rawArray.size(1))/2;
		int I=Math.toIntExact(rawArray.size(2));
		
		try {
			BufferedReader bf=new BufferedReader(new FileReader(new File(keyFileloc)));
			bf.readLine();//get rid of the header
		
		
		for(int i=0;i<I;i++) {
			String line= bf.readLine();
			INDArray X=rawArray.get(new INDArrayIndex[] {NDArrayIndex.all(),NDArrayIndex.interval(0, T),NDArrayIndex.point(i)});
			INDArray Y=rawArray.get(new INDArrayIndex[] {NDArrayIndex.all(),NDArrayIndex.interval(T, 2*T),NDArrayIndex.point(i)});
			dataSet.put(i,new Data(Y, Y, line.split(",")[1]));
			
		}
		} catch (IOException e) {
			// TODO Auto-generated catch block
			System.out.println("the key file and the dataset file is not consistant.");
		}
		return dataSet;
	}
	
	public int getN() {
		return N;
	}
	public int getT() {
		return T;
	}
	public int getI() {
		return I;
	}
	
	public static void writeWeights(Map<String,INDArray> weights,String fileLoc) {
		int N=Math.toIntExact(weights.get("0_0").size(0));
		int T=Math.toIntExact(weights.get("0_0").size(0));
		INDArray rawArray=Nd4j.create(weights.get("0_0").length(),weights.get("0_0").length());
		int k=0;
		for(int n=0;n<N;n++) {
			for(int t=0;t<T;t++) {
				INDArray rawRow=weights.get(Integer.toString(n)+"_"+Integer.toString(t)).reshape(weights.get(Integer.toString(n)+"_"+Integer.toString(t)).length());
				rawArray.putRow(k, rawRow);
				k++;
			}
		}
		Nd4j.writeTxt(rawArray,fileLoc);
	}
	
	public static Map<String,INDArray> readWeight(String fileLoc){
		Map<String,INDArray> weights=new ConcurrentHashMap<>();
		INDArray rawArray=Nd4j.readTxt(fileLoc);
		int N=Math.toIntExact(rawArray.size(0));
		int T=Math.toIntExact(rawArray.size(1));
		for(int n=0;n<N;n++) {
			for(int t=0;t<T;t++) {
				weights.put(Integer.toString(n)+"_"+Integer.toString(t), rawArray.get(new INDArrayIndex[] {NDArrayIndex.all(),NDArrayIndex.all(),NDArrayIndex.point(n),NDArrayIndex.point(t)}));
			}
			
		}
		
		return weights;
	}
	
	public static void writeVariance(Map<String,INDArray> variance,String fileLoc,int N,int T) {
		int I=Math.toIntExact(variance.get("0_0").size(0));
		INDArray rawArray=Nd4j.create(new int[] {I,I,N,T});
		for(int n=0;n<N;n++) {
			for(int t=0;t<T;t++) {
				rawArray.put(new INDArrayIndex[] {NDArrayIndex.all(),NDArrayIndex.all(),NDArrayIndex.point(n),NDArrayIndex.point(t)},variance.get(Integer.toString(n)+"_"+Integer.toString(t)));
			}
		}
		Nd4j.writeTxt(rawArray,fileLoc);
	}
	
	public static Map<String,INDArray> readVariance(String fileLoc){
		Map<String,INDArray> variance=new ConcurrentHashMap<>();
		INDArray rawArray=Nd4j.readTxt(fileLoc);
		int N=Math.toIntExact(rawArray.size(2));
		int T=Math.toIntExact(rawArray.size(3));
		for(int n=0;n<N;n++) {
			for(int t=0;t<T;t++) {
				variance.put(Integer.toString(n)+"_"+Integer.toString(t), rawArray.get(new INDArrayIndex[] {NDArrayIndex.all(),NDArrayIndex.all(),NDArrayIndex.point(n),NDArrayIndex.point(t)}));
			}
			
		}
		
		return variance;
	}
	
}

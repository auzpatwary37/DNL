package training;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.LinkedHashMap;
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
import linktolinkBPR.LinkToLinks;

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
	
	public static void writeData(Map<Integer,Data> dataSet,String fileLoc,String keyFileloc,String routeFileLoc) {
		int N=Math.toIntExact(dataSet.get(0).getX().size(0));
		int T=Math.toIntExact(dataSet.get(0).getX().size(1));
		int R = Math.toIntExact(dataSet.get(0).getR().size(0));
		int I=dataSet.size();
		INDArray rawArray=Nd4j.create(new int[] {N,2*T,I});
		INDArray routeArray = Nd4j.create(new int[] {R,T,I});
		int i=0;
		try {
			FileWriter fw =new FileWriter(new File(keyFileloc));
			fw.append("NumberId,dataKey\n");
			
		for(Entry<Integer,Data>dataPoint:dataSet.entrySet()) {
			INDArray joinedArray=Nd4j.concat(1, dataPoint.getValue().getX(),dataPoint.getValue().getY());
			rawArray.put(new INDArrayIndex[] {NDArrayIndex.all(),NDArrayIndex.all(),NDArrayIndex.point(i)},joinedArray);
			routeArray.put(new INDArrayIndex[] {NDArrayIndex.all(),NDArrayIndex.all(),NDArrayIndex.point(i)},dataPoint.getValue().getR());
			fw.append(dataPoint.getKey()+","+dataPoint.getValue().getKey()+"\n");
			i++;
		}
		
		Nd4j.writeTxt(rawArray,fileLoc);
		Nd4j.writeTxt(routeArray,routeFileLoc);
		fw.flush();
		fw.close();
		
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
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
	
	public static void writeData(List<Data> dataSet,String fileLoc,String keyfileloc,String routeFileLoc) {
		int N=Math.toIntExact(dataSet.get(0).getX().size(0));
		int T=Math.toIntExact(dataSet.get(0).getX().size(1));
		int R = 0;
		if(dataSet.get(0).getR()!=null)R = Math.toIntExact(dataSet.get(0).getR().size(0)); 
		int I=dataSet.size();
		INDArray rawArray=Nd4j.create(new int[] {N,2*T,I});
		INDArray routeArray = Nd4j.create(new int[] {R,T,I});
		int i=0;
		try {
			FileWriter fw =new FileWriter(new File(keyfileloc));
			fw.append("NumberId,dataKey\n");
		for(Data dataPoint:dataSet) {
			INDArray joinedArray=Nd4j.concat(1, dataPoint.getX(),dataPoint.getY());
			rawArray.put(new INDArrayIndex[] {NDArrayIndex.all(),NDArrayIndex.all(),NDArrayIndex.point(i)},joinedArray);
			routeArray.put(new INDArrayIndex[] {NDArrayIndex.all(),NDArrayIndex.all(),NDArrayIndex.point(i)},dataPoint.getR());
			fw.append(i+","+dataPoint.getKey()+"\n");
			i++;
		}
		Nd4j.writeTxt(rawArray,fileLoc);
		Nd4j.writeTxt(routeArray,routeFileLoc);
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
			if(!(X.size(0)==33 && X.size(1)==9 && Y.size(0)==33 && Y.size(1)==9)) {
				//System.out.println();
			}
			dataSet.put(i,new Data(X, Y, line.split(",")[1]));
			
		}
		} catch (IOException e) {
			// TODO Auto-generated catch block
			System.out.println("the key file and the dataset file is not consistant.");
		}
		return dataSet;
	}
	
	public static Map<Integer,Data> readDataSet(String fileLoc,String keyFileloc,String routeFileLoc){
		Map<Integer,Data> dataSet=new ConcurrentHashMap<>();
			INDArray rawArray=Nd4j.readTxt(fileLoc);
			INDArray routeArray = Nd4j.readTxt(routeFileLoc);
			int T=Math.toIntExact(rawArray.size(1))/2;
			int I=Math.toIntExact(rawArray.size(2));
			int R = Math.toIntExact(routeArray.size(1));
			try {
				BufferedReader bf=new BufferedReader(new FileReader(new File(keyFileloc)));
				bf.readLine();//get rid of the header
			
			
			for(int i=0;i<I;i++) {
				String line= bf.readLine();
				INDArray X=rawArray.get(new INDArrayIndex[] {NDArrayIndex.all(),NDArrayIndex.interval(0, T),NDArrayIndex.point(i)});
				INDArray Y=rawArray.get(new INDArrayIndex[] {NDArrayIndex.all(),NDArrayIndex.interval(T, 2*T),NDArrayIndex.point(i)});
				INDArray RD = routeArray.get(new INDArrayIndex[] {NDArrayIndex.all(),NDArrayIndex.all(),NDArrayIndex.point(i)});
				if(!(X.size(0)==33 && X.size(1)==9 && Y.size(0)==33 && Y.size(1)==9)) {
					//System.out.println();
				}
				dataSet.put(i,new Data(X, Y, line.split(",")[1]));
				dataSet.get(i).setR(RD);
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
	public static void writeINDArray(INDArray array,String fileLoc) {
		try {
			System.out.println(Arrays.toString(array.shape()));
			FileWriter fw=new FileWriter(new File(fileLoc));
			for(int i=0;i<array.size(0);i++) {
				String seperator="";
				for(int j=0;j<array.size(1);j++) {
					fw.append(seperator+array.getDouble(i,j));
					seperator=",";
				}
				fw.append("\n");
				fw.flush();
			}
			
			fw.close();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}
	
	public static INDArray readINDArray(String fileLoc) {
		Map<Integer,double[]> rows=new HashMap<>();
		INDArray outArray=null;
		try {
			BufferedReader bf=new BufferedReader(new FileReader(new File(fileLoc)));
			String line=null;
			int rowNum=0;
			while((line=bf.readLine())!=null) {
				double[] row=new double[line.split(",").length];
				int i=0;
				for(String s:line.split(",")) {
					row[i]=Double.parseDouble(s);
					i++;
				}
				rows.put(rowNum, row);
				rowNum++;
			}
			outArray=Nd4j.create(rows.size(),rows.get(0).length);
			for(Entry<Integer, double[]> r:rows.entrySet()) {
				outArray.putRow(r.getKey(), Nd4j.create(r.getValue()));
			}
		} catch (FileNotFoundException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		return outArray;
	}
	
	/**
	 * Creates a 
	 * @param data
	 * @param l2ls
	 * @param writeLoc
	 *
	 */
	public static void createMatlabData(Map<Integer,Data> data,Map<Integer,Data> testdata, LinkToLinks l2ls,String writeLoc) {
		INDArray x = Nd4j.create(data.size(),Math.toIntExact((data.get(0).getX().shape()[0]*data.get(0).getX().shape()[1])));
		INDArray y = Nd4j.create(x.shape());
		INDArray r = Nd4j.create(data.size(),Math.toIntExact((data.get(0).getR().shape()[0]*data.get(0).getR().shape()[1])));
		INDArray w = Nd4j.create(x.shape()[1],x.shape()[1]);
		
		INDArray x_1 = Nd4j.create(testdata.size(),Math.toIntExact((testdata.get(0).getX().shape()[0]*testdata.get(0).getX().shape()[1])));
		INDArray r_1 = Nd4j.create(testdata.size(),Math.toIntExact((testdata.get(0).getR().shape()[0]*testdata.get(0).getR().shape()[1])));
		INDArray y_1 = Nd4j.create(x_1.shape());
		
		
		try {
			FileWriter fw = new FileWriter(new File(writeLoc+"/datakeys.csv"));
			FileWriter fw1 = new FileWriter(new File(writeLoc+"/testdatakeys.csv"));
		
		
		//create the x and y matrix 
		for(int i=0;i<data.size();i++) {
			fw.append(data.get(i).getKey()+"\n");
			INDArray X = data.get(i).getX().reshape('f',1,data.get(i).getX().length());
			x.putRow(i, X);
			INDArray Y = data.get(i).getY().reshape('f',1,data.get(i).getY().length());
			y.putRow(i, Y);
			INDArray R = data.get(i).getR().reshape('f',1,data.get(i).getR().length());
			r.putRow(i, R);
			fw.flush();
		}
		
		fw.close();
		
		writeINDArray(x,writeLoc+"/x.csv");
		writeINDArray(y,writeLoc+"/y.csv");
		writeINDArray(y,writeLoc+"/r.csv");
		
		//create xtest and ytest
		for(int i=0;i<testdata.size();i++) {
			fw1.append(testdata.get(i).getKey()+"\n");
			INDArray X = testdata.get(i).getX().reshape('f',1,testdata.get(i).getX().length());
			x_1.putRow(i, X);
			INDArray Y = testdata.get(i).getY().reshape('f',1,testdata.get(i).getY().length());
			y_1.putRow(i, Y);
			INDArray R = testdata.get(i).getR().reshape('f',1,testdata.get(i).getR().length());
			r_1.putRow(i, R);
			fw1.flush();
		}
		
		fw1.close();
		
		writeINDArray(x_1,writeLoc+"/xtst.csv");
		writeINDArray(y_1,writeLoc+"/ytst.csv");
		writeINDArray(r_1,writeLoc+"/rtst.csv");
		
		//create and write weight matrix 
		
		int N = l2ls.getLinkToLinks().size();
		int T = l2ls.getTimeBean().size();
		
		int j=0;
		for(int t=0;t<T;t++) {
			for(int n=0;n<N;n++) {
				INDArray ow=Nd4j.create(l2ls.getWeightMatrix(n, t).getData());
				INDArray W = ow.reshape('f',1,N*T);
				w.putRow(j, W);
				j++;
			}
		}
		
		writeINDArray(w,writeLoc+"/weights.csv");
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}
	
//	public static void main(String[] args) {
//		INDArray a= Nd4j.rand(2,3);
//		System.out.println(a);
//		System.out.println(a.reshape('f',1,a.length()));
//	}
	
	
	public static RouteData readRouteData(String fileLoc, int maxIterNo) {
		LinkedHashMap<String,List<Integer>> routes = new LinkedHashMap<>();
		int T = 0;
		Map<String,Map<Integer,Map<String,Double>>> routeDemand = new HashMap<>();

		try {
			for(int i = 0; i<=maxIterNo;i++) {
				BufferedReader bf = new BufferedReader(new FileReader(new File(fileLoc+"routeDemand"+i+".csv")));
				String line;
				while((line=bf.readLine())!=null) {
					String[] part = line.split(",");
					String key = part[0];
					String routeId = part[1];
					int t = Integer.parseInt(part[2]);
					double demand = Double.parseDouble(part[3]);
					List<Integer> linkList = new ArrayList<>();
					for(int j = 4;j<part.length;j++) {
						linkList.add(Integer.parseInt(part[j]));
					}
					
					if (t>T) T = t;
 					if(routes.containsKey(routeId)) {
 						routes.put(routeId, linkList);
 					}
 					if(!routeDemand.containsKey(key))routeDemand.put(key, new HashMap<>());
 					if(!routeDemand.get(key).containsKey(t))routeDemand.get(key).put(t, new HashMap<>());
 					routeDemand.get(key).get(t).compute(routeId, (k,v)->v==null?demand:v+demand);
				}
			}
			
		} catch (FileNotFoundException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		int R = routes.size();
		return new RouteData(routes,routeDemand,T);
	}
}

class RouteData{
	public final LinkedHashMap<String,List<Integer>> routes;
	public final List<String> routeList;
	public final int T;
	public final Map<String,Map<Integer,Map<String,Double>>> routeDemand;
	public final int R;
	public RouteData(LinkedHashMap<String,List<Integer>> routes,Map<String,Map<Integer,Map<String,Double>>> routeDemand, int T) {
		this.R = routes.size();
		this.routeDemand = routeDemand;
		this.routes = routes;
		this.T = T;
		this.routeList = new ArrayList<>(this.routes.keySet());
	}
	public void writeRouteDetails(String fileLoc) {
		try {
			FileWriter fw = new FileWriter(new File(fileLoc));
			fw.append("RouteNo, RouteId, L2ls\n");
			for(int i=0;i<this.routeList.size();i++) {
				fw.append(i+","+this.routeList.get(i));
				for(int j:this.routes.get(this.routeList.get(i))) {
					fw.append(","+j);
				}
				fw.append("\n");
			}
			fw.flush();
			fw.close();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}
}

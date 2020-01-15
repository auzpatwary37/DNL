package training;

import java.io.File;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.stream.IntStream;

import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.checkutil.CheckUtil;
import org.nd4j.linalg.factory.Nd4j;

import kriging.Data;
import kriging.Variogram;
import linktolinkBPR.LinkToLinks;

/**
 * This class will control the n_t specific dataset generation, testing and training and new data set addition
 * @author h
 *
 */
public class TrainingController {

	private LinkToLinks l2ls;
	private Map<Integer,Data> trainingData;
	private Map<String,INDArray> fullDistanceMatrixMap=new HashMap<>();
	private Map<String,INDArray> fullDistanceMatrixMapY=new HashMap<>();
	private final int N;
	private final int T;
	public final String defaultWriteLoc="distanceMatrix.csv";
	
	public TrainingController(LinkToLinks l2ls,Map<Integer,Data> trainingData) {
		this.l2ls=l2ls;
		this.trainingData=trainingData;
		this.N=Math.toIntExact(trainingData.get(0).getX().size(0));
		this.T=Math.toIntExact(trainingData.get(0).getX().size(1));
		this.calculateDistanceMatrix();
		//this.calculateDistanceMatrixY();
		this.writeDistanceMatrix(defaultWriteLoc);
	}
	
	public TrainingController(LinkToLinks l2ls,Map<Integer,Data> trainingData,boolean realOld) {
		this.l2ls=l2ls;
		this.trainingData=trainingData;
		this.N=Math.toIntExact(trainingData.get(0).getX().size(0));
		this.T=Math.toIntExact(trainingData.get(0).getX().size(1));
		if(realOld==true && new File(this.defaultWriteLoc).exists()) {
			this.fullDistanceMatrixMap=readDistanceMatrix(defaultWriteLoc, N, T, trainingData.size());
		}else {
			this.calculateDistanceMatrix();
			//this.calculateDistanceMatrixY();
			this.writeDistanceMatrix(defaultWriteLoc);
		}
	}

	private void calculateDistanceMatrix() {
		long starttime=System.currentTimeMillis();
		IntStream.rangeClosed(0,N-1).parallel().forEach((n)->
		{
			IntStream.rangeClosed(0,T-1).parallel().forEach((t)->{
				double[][] distanceMatrix=new double[this.trainingData.size()][this.trainingData.size()];
				for(int i=0;i<this.trainingData.size();i++) {
					for(int j=0;j<=i;j++) {
						if(j==i) {
							distanceMatrix[i][i]=0;
						}
						INDArray weight=CheckUtil.convertFromApacheMatrix(this.l2ls.getWeightMatrix(n, t),DataType.DOUBLE);
						double distance=Variogram.calcDistance(this.trainingData.get(i).getX(), this.trainingData.get(j).getX(), n, t, weight);
						distanceMatrix[i][j]=distance;
						distanceMatrix[j][i]=distance;
					}
				}
				this.fullDistanceMatrixMap.put(Integer.toString(n)+"_"+Integer.toString(t), Nd4j.create(distanceMatrix));
				
			});
		});
		System.out.println("total time to create distance matrix = " + Long.toString(System.currentTimeMillis()-starttime));
	}
	
	private void calculateDistanceMatrixY() {
		long starttime=System.currentTimeMillis();
		IntStream.rangeClosed(0,N-1).parallel().forEach((n)->
		{
			IntStream.rangeClosed(0,T-1).parallel().forEach((t)->{
				double[][] distanceMatrix=new double[this.trainingData.size()][this.trainingData.size()];
				for(int i=0;i<this.trainingData.size();i++) {
					for(int j=0;j<=i;j++) {
						if(j==i) {
							distanceMatrix[i][i]=0;
						}
						INDArray weight=CheckUtil.convertFromApacheMatrix(this.l2ls.getWeightMatrix(n, t),DataType.DOUBLE);
						double distance=Variogram.calcDistance(this.trainingData.get(i).getY(), this.trainingData.get(j).getY(), n, t, weight);
						distanceMatrix[i][j]=distance;
						distanceMatrix[j][i]=distance;
					}
				}
				this.fullDistanceMatrixMapY.put(Integer.toString(n)+"_"+Integer.toString(t), Nd4j.create(distanceMatrix));
			});
		});
		System.out.println("total time to write distance matrix = " + Long.toString(System.currentTimeMillis()-starttime));
	}
	
	
	public void writeDistanceMatrix(String fileLoc) {
		long starttime=System.currentTimeMillis();
		int k=0;
		INDArray bigArray=Nd4j.create(this.N*this.T,this.trainingData.size()*this.trainingData.size());
		for(int n=0;n<this.N;n++) {
			for(int t=0;t<this.T;t++) {
				INDArray distanceMatrix=this.fullDistanceMatrixMap.get(Integer.toString(n)+"_"+Integer.toString(t));
				bigArray.putRow(k, distanceMatrix.reshape(distanceMatrix.length()));
				k++;
			}
		}
		DataIO.writeINDArray(bigArray, fileLoc);
		System.out.println("total time to write distance matrix = " + Long.toString(System.currentTimeMillis()-starttime));
	}
	
	public static Map<String,INDArray> readDistanceMatrix(String fileLoc, int N, int T, int I) {
		long starttime=System.currentTimeMillis();
		Map<String,INDArray> distances= new HashMap<>();
		INDArray bigArray=DataIO.readINDArray(fileLoc);
		int k=0;
		for(int n=0;n<N;n++) {
			for(int t=0;t<T;t++) {
				INDArray distanceMatrix=bigArray.getRow(k);
				distanceMatrix=distanceMatrix.reshape(I,I);
				distances.put(Integer.toString(n)+"_"+Integer.toString(t), distanceMatrix);
				k++;
			}
		}
		System.out.println("total time to read distance matrix = " + Long.toString(System.currentTimeMillis()-starttime));
		return distances;
	}
	
	
	/**
	 * Creates a k sized training set indices set for each n and t pair
	 * @param k
	 * @return
	 */
	public Map<String,List<Integer>> createN_TSpecificTrainingSet(int k){
		Map<String,List<Integer>> n_tSpecificTrainingIndices=new HashMap<>();
		for(int n=0;n<this.N;n++) {
			for(int t=0;t<this.T;t++) {
				String key=Integer.toString(n)+"_"+Integer.toString(t);
				n_tSpecificTrainingIndices.put(key, FarthestPointSampler.pickKFurthestPoint(this.fullDistanceMatrixMap.get(key), k));
			}
		}
		return n_tSpecificTrainingIndices;
	}
	
	/**
	 * Creates a k sized training set indices set for each n and t pair
	 * @param k
	 * @return
	 */
	public Map<String,List<Integer>> createN_TSpecificTrainingSetWithYAsWeight(int k){
		Map<String,List<Integer>> n_tSpecificTrainingIndices=new HashMap<>();
		for(int n=0;n<this.N;n++) {
			for(int t=0;t<this.T;t++) {
				String key=Integer.toString(n)+"_"+Integer.toString(t);
				n_tSpecificTrainingIndices.put(key, FarthestPointSampler.pickAdditionalKFurthestPoint(this.fullDistanceMatrixMap.get(key), new ArrayList<>(), k,this.fullDistanceMatrixMapY.get(key)));
			}
		}
		return n_tSpecificTrainingIndices;
	}
	
	
	/**
	 * This creates n_t specific training data indices for specific n and t of size k
	 * the weight is taken as input, so the distance matrix is calculated inside 
	 * Should be used only for train-able weight case.
	 * @param n
	 * @param t
	 * @param k
	 * @param weight
	 * @return
	 */
	public List<Integer> createN_TSpecificTrainingSet(int n,int t,int k,INDArray weight){
		double[][] distanceMatrix=new double[this.trainingData.size()][this.trainingData.size()];
		for(int i=0;i<this.trainingData.size();i++) {
			for(int j=0;j<=i;j++) {
				if(j==i) {
					distanceMatrix[i][i]=0;
				}
				double distance=Variogram.calcDistance(this.trainingData.get(i).getX(), this.trainingData.get(j).getX(), n, t, weight);
				distanceMatrix[i][j]=distance;
				distanceMatrix[j][i]=distance;
			}
		}
		return FarthestPointSampler.pickKFurthestPoint(Nd4j.create(distanceMatrix), k);
	}
	
	/**
	 * Adds k points to each n_t pair in the current indices 
	 * error map will contain n-t key vs trainingDataIdentifier vs error
	 * @param currentIndices
	 * @param k
	 * @param errorMap
	 * @return
	 */
	public Map<String,List<Integer>> createAdditionalN_TSpecificTrainingSetWithErrorWeight(Map<String,List<Integer>> currentIndices, int k, Map<String,Map<Integer,Double>> errorMap){
		System.out.println();
		for(int n=0;n<this.N;n++) {
			for(int t=0;t<this.T;t++) {
				String key=Integer.toString(n)+"_"+Integer.toString(t);
				currentIndices.put(key, FarthestPointSampler.pickAdditionalKFurthestPointScaled(this.fullDistanceMatrixMap.get(key), currentIndices.get(key), k, errorMap.get(key)));
			}
		}
		return currentIndices;
	}
	
	
	/**
	 * Adds k points to the n_t pair in the current indices 
	 * error map will contain n-t key vs trainingDataIdentifier vs error
	 * The weight is input so distance matrix is calculated inside 
	 * Should be only used in case of train-able weight
	 * @param currentIndices
	 * @param k
	 * @param errorMap
	 * @return
	 */
	public List<Integer> createAdditionalN_TSpecificTrainingSetWithErrorWeight(int n, int t, int k, List<Integer> currentIndices,  Map<Integer,Double> errorMap, INDArray weight){
		double[][] distanceMatrix=new double[this.trainingData.size()][this.trainingData.size()];
		for(int i=0;i<this.trainingData.size();i++) {
			for(int j=0;j<=i;j++) {
				if(j==i) {
					distanceMatrix[i][i]=0;
				}
				double distance=Variogram.calcDistance(this.trainingData.get(i).getX(), this.trainingData.get(j).getX(), n, t, weight);
				distanceMatrix[i][j]=distance;
				distanceMatrix[j][i]=distance;
			}
		}
		return FarthestPointSampler.pickAdditionalKFurthestPoint(Nd4j.create(distanceMatrix), currentIndices, k, errorMap);
	}
	
}

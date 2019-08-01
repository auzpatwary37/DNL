package kriging;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.stream.IntStream;

import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.stat.descriptive.moment.StandardDeviation;
import org.matsim.core.utils.collections.Tuple;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.Op;
import org.nd4j.linalg.api.ops.impl.reduce.longer.MatchCondition;
import org.nd4j.linalg.checkutil.CheckUtil;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.conditions.Conditions;
import org.nd4j.linalg.ops.transforms.Transforms;

import linktolinkBPR.LinkToLinks;



/**
 * This class will calculate the required covariance marices 
 * @author Ashraf
 *
 */
public class Variogram {
	
	private final Map<Integer,Data> trainingDataSet;
	private final INDArray sigmaMatrix;
	private final Map<String,RealMatrix> weights;
	private INDArray theta;
	private Map<String,INDArray> varianceMapAll;
	private final int N;
	private final int T;
	private Map<String,INDArray>distances=new ConcurrentHashMap<>();
	private INDArray distanceScale;
	private INDArray ttScale;
	private Map<String,Map<Integer,Data>> ntSpecificTrainingSet=new ConcurrentHashMap<>();
	private Map<String,List<Integer>>ntSpecificOriginalIndices=new ConcurrentHashMap<>();
	private boolean scaleData=false;
	private LinkToLinks l2ls;
	
	//TODO: Add a writer to save the trained model
	
	/**
	 * This will initialize the theta matrix and calculate and store the IxI variance matrix by default
	 * @param trainingData
	 * @param l2ls
	 */
	public Variogram(Map<Integer, Data> trainingData,LinkToLinks l2ls) {
		long starttime=System.currentTimeMillis();
		this.trainingDataSet=trainingData;
		this.weights=l2ls.getWeightMatrices();
		this.N=Math.toIntExact(trainingData.get(0).getX().size(0));
		this.T=Math.toIntExact(trainingData.get(0).getX().size(1));
		this.distanceScale=Nd4j.ones(N,T);
		this.ttScale=Nd4j.ones(N,T);
		this.sigmaMatrix=this.calcSigmaMatrix(trainingData);
		//Initialize theta to a nxt matrix of one
		this.theta=Nd4j.zeros(N,T).addi(.1);
		this.l2ls=l2ls;
		this.calcDistances();
		//this.varianceMapAll=this.calculateVarianceMatrixAll(this.theta);
		System.out.println("Finished setting up initial variogram. Total required time = "+Long.toString(System.currentTimeMillis()-starttime));
		
	}
	
	//TODO: fix the scales
	public Variogram(Map<Integer,Data>trainingDataSet,Map<String,RealMatrix>weights,INDArray theta) {
		this.trainingDataSet=trainingDataSet;
		this.N=Math.toIntExact(trainingDataSet.get(0).getX().size(0));
		this.T=Math.toIntExact(trainingDataSet.get(0).getX().size(1));
		this.weights=weights;
		this.sigmaMatrix=this.calcSigmaMatrix(trainingDataSet);
		//Initialize theta to a nxt matrix of one
		this.theta=theta;
		this.calcDistances();
		//Will be fixed later
		this.distanceScale=Nd4j.ones(N,T);
		this.ttScale=Nd4j.ones(N,T);
		//this.varianceMapAll=this.calculateVarianceMatrixAll(this.theta);
		
	}
	
	public Variogram(Map<Integer,Data>trainingDataSet,LinkToLinks l2ls,INDArray theta,INDArray Cn,INDArray Ct) {
		this.trainingDataSet=trainingDataSet;
		this.l2ls=l2ls;
		this.weights=l2ls.getWeightMatrices(Cn,Ct);
		this.theta=theta;
		this.N=(int) this.trainingDataSet.get(0).getY().size(0);
		this.T=(int) this.trainingDataSet.get(0).getY().size(1);
		this.distanceScale=Nd4j.ones(N,T);		
		this.ttScale=Nd4j.ones(N,T);
		this.sigmaMatrix=this.calcSigmaMatrix(trainingDataSet);
		this.calcDistances();
	}
	/**
	 * 
	 * @param a first tensor
	 * @param b second tensor
	 * @param n link in calculation
	 * @param t time in calculation
	 * @param ka number of connected l2l to consider
	 * @param kt number of connected time to consider
	 * @param theta parameters for the variogram
	 * @return
	 */
	public INDArray calcSigmaMatrix(Map<Integer,Data>trainingDataSet) {
		INDArray sd=Nd4j.create(N,T);
		StandardDeviation sdCalculator= new StandardDeviation();
		IntStream.rangeClosed(0,N-1).parallel().forEach((n)->
		{
			IntStream.rangeClosed(0,T-1).parallel().forEach((t)->{
				double[] data=new double[trainingDataSet.size()];
				for(int i=0;i<trainingDataSet.size();i++) {
					data[i]=trainingDataSet.get(i).getY().getDouble(n, t);
				}
				INDArray dat=Nd4j.create(data);
				if(this.scaleData) {
					double scale=1/dat.maxNumber().doubleValue();
					this.ttScale.put(n, t,scale);
				}
				
				sd.putScalar(n, t, Math.pow(sdCalculator.evaluate(dat.mul(this.ttScale.getDouble(n,t)).toDoubleVector()),2));
			});
		});
		KrigingInterpolator.writeINDArray(sd, "sd.csv");
		return sd;
	}
	
	public double calcDistance(INDArray a,INDArray b,int n,int t) {
		INDArray weight=CheckUtil.convertFromApacheMatrix(this.weights.get(Integer.toString(n)+"_"+Integer.toString(t)),DataType.DOUBLE);
		INDArray aa=a.sub(b).mul(weight);
		double distance= aa.norm2Number().doubleValue();
		return distance;
	}
	
	//For now this function is useless but it can be later used to make the weights trainable
	public double calcDistance(INDArray a,INDArray b,Map<String,INDArray> weights,int n,int t) {
		return a.sub(b).mul(weights.get(Integer.toString(n)+"_"+Integer.toString(t))).norm2Number().doubleValue();
		//return distance;
	}
	
	public double calcDistance(INDArray a,INDArray b,int n,int t,INDArray weights) {
		
		double aa=a.sub(b).mul(weights).norm2Number().doubleValue();
		return aa;
		 //return distance;
	}
	/**
	 * This function will calculate the K matrix for a (n,t) pair the dimension of the matrix is IxI where I is the number of data point.
	 * THe auto co-variance function is sigma^2exp(-1*distance*theta)
	 * @param n the l2l id 
	 * @param t time id  
	 * @return K dim IxI
	 */
	public INDArray calcVarianceMatrix(int n, int t,INDArray theta){
		double sigma=sigmaMatrix.getDouble(n, t);
		int I=this.ntSpecificTrainingSet.get(Integer.toString(n)+"_"+Integer.toString(t)).size();
		INDArray K=Nd4j.create(I,I);
		for(int ii=0;ii<I;ii++) {
			for(int jj=0;jj<=ii;jj++) {
				if(ii!=jj) {
					double dist=(-1*this.distances.get(Integer.toString(n)+"_"+Integer.toString(t)).getDouble(ii,jj)*this.distanceScale.getDouble(n,t)*theta.getDouble(n,t));
					double v=sigma*Math.exp(dist);
					if(Double.isNaN(v)||!Double.isFinite(v)) {
						throw new IllegalArgumentException("is infinity");
					}
					K.putScalar(ii, jj, v);
					K.putScalar(jj, ii, v);
				}else {
					K.putScalar(ii, ii,sigma);
				}
			}
		}
		int i=0;
		//INDArray KK=Transforms.exp(this.distances.get(Integer.toString(n)+"_"+Integer.toString(t)).mul(-1*this.distanceScale.getDouble(n,t)*theta.getDouble(n, t)),true).mul(sigma);
		return K;
	}
	/**
	 * this will calculate the variance matrix for n,t with variogram parameter theta
	 * @param n LinkToLink
	 * @param t	Time
	 * @param theta variogram parameter 
	 * @return 
	 */
	public INDArray calcVarianceMatrix(int n, int t,double theta){
		int I=this.ntSpecificTrainingSet.get(Integer.toString(n)+"_"+Integer.toString(t)).size();
		double sigma=sigmaMatrix.getDouble(n, t);
		INDArray K=Nd4j.create(I,I);
		int i=0;
		
		for(int ii=0;ii<I;ii++) {
			for(int jj=0;jj<=ii;jj++) {
				if(ii!=jj) {
					double dist= (-1*this.distances.get(Integer.toString(n)+"_"+Integer.toString(t)).getDouble(ii,jj)*this.distanceScale.getDouble(n,t)*theta);
					double v=(sigma*Math.exp(dist));
					if(Double.isNaN(v)||!Double.isFinite(v)) {
						throw new IllegalArgumentException("is infinity");
					}
					K.putScalar(ii, jj, v);
					K.putScalar(jj, ii, v);
					//System.out.println("finished putting");
				}else {
					K.putScalar(ii, ii,sigma);
				}
			}
		}
		return K;
	}
	public void calcDistances() {
		
		IntStream.rangeClosed(0,N-1).parallel().forEach((n)->
		{
			IntStream.rangeClosed(0,T-1).parallel().forEach((t)->{
				this.distances.put(Integer.toString(n)+"_"+Integer.toString(t),this.calcDistanceMatrix(n, t));
			});
		});
	}
	public void calcDistances(double cn,double ct) {
		
		IntStream.rangeClosed(0,N-1).parallel().forEach((n)->
		{
			IntStream.rangeClosed(0,T-1).parallel().forEach((t)->{
				this.calcDistanceMatrix(n, t,cn,ct);
			});
		});
	}
	private INDArray calcDistanceMatrix(int n, int t) {
		
		INDArray weight = CheckUtil.convertFromApacheMatrix(this.weights.get(Integer.toString(n)+"_"+Integer.toString(t)),DataType.DOUBLE);
		this.ntSpecificTrainingSet.put(Integer.toString(n)+"_"+Integer.toString(t), new HashMap<>());
		Map<Integer,Data> ntTrainData=this.ntSpecificTrainingSet.get(Integer.toString(n)+"_"+Integer.toString(t));
		int i=0;
		Map<String,Double>distMap=new HashMap<>();
		List<Integer> intMap=new ArrayList<>();
		//INDArray boolArray;
		for(int ii=0;ii<this.trainingDataSet.size();ii++) {
			
			boolean repeat=false;
			for(int jj=0;jj<i;jj++) {
				if(ii!=jj) {
					double dist=this.calcDistance(this.trainingDataSet.get(ii).getX(), ntTrainData.get(jj).getX(), n, t);
					if(Transforms.abs(this.trainingDataSet.get(ii).getX().sub(ntTrainData.get(jj).getX()).mul(weight)).norm2Number().doubleValue()<1.0){
						repeat=true;
						break;
					}else {
						distMap.put(Integer.toString(i)+"_"+Integer.toString(jj), dist);
					}	
				}
			}
			if(repeat==false) {
				intMap.add(ii);
				ntTrainData.put(i, this.trainingDataSet.get(ii));
				i++;
			}
		}
		this.ntSpecificOriginalIndices.put(Integer.toString(n)+"_"+Integer.toString(t), intMap);
		if(intMap.size()==1) {
			System.out.println("Debug!!!");
		}
		INDArray K=Nd4j.create(i,i);
		IntStream.rangeClosed(0,i-1).parallel().forEach((ii)->
		{
			IntStream.rangeClosed(0,ii).parallel().forEach((jj)->{
				if(ii!=jj) {
					double dist=distMap.get(Integer.toString(ii)+"_"+Integer.toString(jj));
					K.putScalar(ii, jj, dist);
					K.putScalar(jj, ii, dist);
				}else {
					K.putScalar(ii, ii,0);
				}
			});
		});
		if(this.scaleData) {
			this.distanceScale.putScalar(n,t,1.0/K.maxNumber().doubleValue());
		}
		if(intMap.size()!=K.size(0)) {
			System.out.println("Debug Here");
		}
		//KrigingInterpolator.writeINDArray(K, Integer.toString(n)+"_"+Integer.toString(t)+"distance.csv");
		return K;
	}

	public void calcDistanceMatrix(int n, int t,double cn, double ct) {
		RealMatrix weightss=this.l2ls.generateWeightMatrix(n, t, cn, ct);
		INDArray weights=CheckUtil.convertFromApacheMatrix(this.l2ls.generateWeightMatrix(n, t, cn, ct),DataType.DOUBLE);//The only difference
		this.weights.put(Integer.toString(n)+"_"+Integer.toString(t), weightss);
		this.ntSpecificTrainingSet.put(Integer.toString(n)+"_"+Integer.toString(t), new HashMap<>());
		int i=0;
		Map<String,Double>distMap=new HashMap<>();
		List<Integer> intMap=new ArrayList<>();
		for(int ii=0;ii<this.trainingDataSet.size();ii++) {
			boolean repeat=false;
			for(int jj=0;jj<i;jj++) {
				if(ii!=jj) {
					double dist=this.calcDistance(this.trainingDataSet.get(ii).getX(), this.ntSpecificTrainingSet.get(Integer.toString(n)+"_"+Integer.toString(t)).get(jj).getX(), n, t);
					if(Transforms.abs(this.trainingDataSet.get(ii).getX().sub(this.ntSpecificTrainingSet.get(Integer.toString(n)+"_"+Integer.toString(t)).get(jj).getX()).mul(weights)).lt(1).all()){
						repeat=true;
						break;
					}else {
						distMap.put(Integer.toString(i)+"_"+Integer.toString(jj), dist);
					}	
				}
			}
			if(repeat==false) {
				intMap.add(ii);
				this.ntSpecificTrainingSet.get(Integer.toString(n)+"_"+Integer.toString(t)).put(i, this.trainingDataSet.get(ii));
				i++;
			}
		}
		this.ntSpecificOriginalIndices.put(Integer.toString(n)+"_"+Integer.toString(t), intMap);
		
		INDArray K=Nd4j.create(i,i);
		IntStream.rangeClosed(0,i-1).parallel().forEach((ii)->
		{
			IntStream.rangeClosed(0,ii).parallel().forEach((jj)->{
				if(ii!=jj) {
					double dist=distMap.get(Integer.toString(ii)+"_"+Integer.toString(jj));
					K.putScalar(ii, jj, dist);
					K.putScalar(jj, ii, dist);
				}else {
					K.putScalar(ii, ii,0);
				}
			});
		});
		if(intMap.size()!=K.size(0)) {
			System.out.println("Debug Here");
		}
		if(this.scaleData) {
			this.distanceScale.putScalar(n,t,1.0/K.maxNumber().doubleValue());
		}
		
		//long endTime=System.currentTimeMillis();
		//System.out.println("Done = "+Integer.toString(n)+"_"+Integer.toString(t));

//		}
		this.distances.put(Integer.toString(n)+"_"+Integer.toString(t), K);//The only difference
	}
	/**
	 * Similar to calcVarianceMatrix but it will calculate variance for a particular point to all other point
	 * @param n
	 * @param t
	 * @param X
	 * @param theta
	 * @return the covariance vector with dimension Ix1
	 */
	public INDArray calcVarianceVector(int n, int t,INDArray X, INDArray theta){
		double sigma=sigmaMatrix.getDouble(n, t);
		Map<Integer,Data> dataset=this.ntSpecificTrainingSet.get(Integer.toString(n)+"_"+Integer.toString(t));
		int i=0;
		double[][] k=new double[dataset.size()][1]; 
		for(Data dataPair:dataset.values()) {
			double v=sigma*Math.exp(-1*this.calcDistance(X, dataPair.getX(), n, t)*theta.getDouble(n, t)*this.distanceScale.getDouble(n,t));
			k[i][0]=v;
			i++;
		}
		return Nd4j.create(k);
	}
	
	/**
	 * calculate IxI covariance matrix for all n x t outputs
	 * @param theta
	 * @return the map with n_t -> IxI realMatrix covariance matrix
	 */
	public Map<String,INDArray>calculateVarianceMatrixAll(INDArray theta){
		if(theta.isNaN().any()||theta.isInfinite().any()) {
			throw new IllegalArgumentException("Theta is nan or infinity!!!");
		}
		long startTime=System.currentTimeMillis();
		Map<String,INDArray> varianceMatrixAll=new ConcurrentHashMap<>();
		IntStream.rangeClosed(0,N-1).parallel().forEach((n)->
		{
			IntStream.rangeClosed(0,T-1).parallel().forEach((t)->{
				//System.out.println(Integer.toString(n)+"_"+Integer.toString(t));
				varianceMatrixAll.put(Integer.toString(n)+"_"+Integer.toString(t),this.calcVarianceMatrix(n, t, theta));
				
			});
		});
		System.out.println("Time for all Matrix = "+(System.currentTimeMillis()-startTime));
		return varianceMatrixAll;
	}
	
	/**
	 * calculate Ix1 covariance vector for all n x t outputs
	 * @param theta
	 * @return the map with n_t -> Ix1 realMatrix covariance matrix
	 */
	public Map<String,INDArray>calculateVarianceVectorAll(INDArray X,INDArray theta){
		Map<String,INDArray> varianceVectorAll=new ConcurrentHashMap<>();
		List<String>n_tlist=new ArrayList<>();
		for(int n=0;n<N;n++) {
			for(int t=0;t<T;t++) {
				n_tlist.add(Integer.toString(n)+"_"+Integer.toString(t));
				}
			}
			n_tlist.parallelStream().forEach((e)->{
				int n=Integer.parseInt(e.split("_")[0]);
				int t=Integer.parseInt(e.split("_")[1]);
			varianceVectorAll.put(e,this.calcVarianceVector(n, t, X, theta));
			});
		
		return varianceVectorAll;
	}
	/**
	 * Update theta and recalculate the IxI variance matrix
	 * @param theta
	 */
	public void updatetheta(INDArray theta) {
		this.theta=theta;
		//this.varianceMapAll=this.calculateVarianceMatrixAll(this.theta);
	}
	
	//Getters and Setters

	public Map<Integer, Data> getTrainingDataSet() {
		return trainingDataSet;
	}


	public INDArray getSigmaMatrix() {
		return sigmaMatrix;
	}

	public INDArray gettheta() {
		return theta;
	}


	public Map<String, RealMatrix> getWeights() {
		return weights;
	}

	public INDArray getDistanceScale() {
		return distanceScale;
	}

	public INDArray getTtScale() {
		return ttScale;
	}

	public Map<String, Map<Integer, Data>> getNtSpecificTrainingSet() {
		return ntSpecificTrainingSet;
	}

	public Map<String, List<Integer>> getNtSpecificOriginalIndices() {
		return ntSpecificOriginalIndices;
	}

	public Map<String, INDArray> getDistances() {
		return distances;
	}

	public LinkToLinks getL2ls() {
		return l2ls;
	}
	
	
	
}



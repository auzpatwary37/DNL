package kriging;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
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

//TODO: take n_t specific training dataset generation outside of the main variogram class

public class Variogram {
	
	private final Map<Integer,Data> trainingDataSet;
	private INDArray sigmaMatrix;
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
	private INDArray nugget;
	private boolean useNugget=false;
	private boolean useFlatWeightMatrix=true;
	
	//TODO: Add a writer to save the trained model
	
	/**
	 * This will initialize the theta matrix and calculate and store the IxI variance matrix by default
	 * @param trainingData
	 * @param l2ls
	 */
	public Variogram(Map<Integer, Data> trainingData,LinkToLinks l2ls,Map<String,List<Integer>>n_tSpecificTrainingIndices) {
		long starttime=System.currentTimeMillis();
		this.trainingDataSet=trainingData;
		this.ntSpecificOriginalIndices=n_tSpecificTrainingIndices;
		this.weights=l2ls.getWeightMatrices();
		this.N=Math.toIntExact(trainingData.get(0).getX().size(0));
		this.T=Math.toIntExact(trainingData.get(0).getX().size(1));
		this.distanceScale=Nd4j.ones(N,T);
		this.ttScale=Nd4j.ones(N,T);
		
		//Initialize theta to a nxt matrix of one
		this.theta=Nd4j.zeros(N,T).addi(.1);
		this.l2ls=l2ls;
		this.calcDistances();
		this.sigmaMatrix=this.calcSigmaMatrix(trainingData);
		this.nugget=Nd4j.zeros(N,T);
		//this.varianceMapAll=this.calculateVarianceMatrixAll(this.theta);
		System.out.println("Finished setting up initial variogram. Total required time = "+Long.toString(System.currentTimeMillis()-starttime));
		
	}
	
	//TODO: fix the scales
	public Variogram(Map<Integer,Data>trainingDataSet,Map<String,RealMatrix>weights,INDArray theta,INDArray nugget,Map<String,List<Integer>>n_tSpecificTrainingIndices) {
		this.trainingDataSet=trainingDataSet;
		this.ntSpecificOriginalIndices=n_tSpecificTrainingIndices;
		this.N=Math.toIntExact(trainingDataSet.get(0).getX().size(0));
		this.T=Math.toIntExact(trainingDataSet.get(0).getX().size(1));
		this.weights=weights;
		this.nugget=nugget;
		//Initialize theta to a nxt matrix of one
		this.theta=theta;
		this.calcDistances();
		this.sigmaMatrix=this.calcSigmaMatrix(trainingDataSet);
		//Will be fixed later
		this.distanceScale=Nd4j.ones(N,T);
		this.ttScale=Nd4j.ones(N,T);
		//this.varianceMapAll=this.calculateVarianceMatrixAll(this.theta);
		
	}
	
	public Variogram(Map<Integer,Data>trainingDataSet,LinkToLinks l2ls,INDArray theta,INDArray nugget, INDArray Cn,INDArray Ct,Map<String,List<Integer>>n_tSpecificTrainingIndices) {
		this.trainingDataSet=trainingDataSet;
		this.ntSpecificOriginalIndices=n_tSpecificTrainingIndices;
		this.nugget=nugget;
		this.l2ls=l2ls;
		this.weights=l2ls.getWeightMatrices(Cn,Ct);
		this.theta=theta;
		this.N=(int) this.trainingDataSet.get(0).getY().size(0);
		this.T=(int) this.trainingDataSet.get(0).getY().size(1);
		this.distanceScale=Nd4j.ones(N,T);		
		this.ttScale=Nd4j.ones(N,T);
		
		this.calcDistances();
		this.sigmaMatrix=this.calcSigmaMatrix(trainingDataSet);
		
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
				double[] data=new double[this.ntSpecificOriginalIndices.get(Integer.toString(n)+"_"+Integer.toString(t)).size()];
				int j=0;
				for(int i:this.ntSpecificOriginalIndices.get(Integer.toString(n)+"_"+Integer.toString(t))) {
					data[j]=trainingDataSet.get(i).getY().getDouble(n, t);
					j++;
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
	
	
	
	
	public void setSigmaMatrix(INDArray sigmaMatrix) {
		this.sigmaMatrix = sigmaMatrix;
	}

	public boolean isScaleData() {
		return scaleData;
	}

	public void setScaleData(boolean scaleData) {
		this.scaleData = scaleData;
	}

	public void setDistanceScale(INDArray distanceScale) {
		this.distanceScale = distanceScale;
	}

	public void setTtScale(INDArray ttScale) {
		this.ttScale = ttScale;
	}

	public INDArray getNugget() {
		return nugget;
	}

	public void setNugget(INDArray nugget) {
		this.nugget = nugget;
	}

	public double calcDistance(INDArray a,INDArray b,int n,int t) {
		INDArray weight=CheckUtil.convertFromApacheMatrix(this.weights.get(Integer.toString(n)+"_"+Integer.toString(t)),DataType.DOUBLE);
		INDArray aa=a.sub(b).mul(weight);
		double distance= Math.pow(aa.norm2Number().doubleValue(),2);
		return distance;
	}
	
	//For now this function is useless but it can be later used to make the weights trainable
	public static double calcDistance(INDArray a,INDArray b,Map<String,INDArray> weights,int n,int t) {
		return a.sub(b).mul(weights.get(Integer.toString(n)+"_"+Integer.toString(t))).norm2Number().doubleValue();
		//return distance;
	}
	
	public static double calcDistance(INDArray a,INDArray b,int n,int t,INDArray weights) {
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
	public INDArray calcVarianceMatrix(int n, int t,INDArray theta, INDArray nugget){
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
					K.putScalar(ii, ii,sigma + nugget.getDouble(n,t));
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
	public INDArray calcVarianceMatrix(int n, int t,double theta, double nugget){
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
					K.putScalar(ii, ii,sigma+nugget);
				}
			}
		}
		return K;
	}
	
	/**
	 * this will calculate the variance matrix for n,t with variogram parameter theta
	 * @param n LinkToLink
	 * @param t	Time
	 * @param theta variogram parameter 
	 * @return 
	 */
	public INDArray calcVarianceMatrixWithoutSigma(int n, int t,double theta, double nugget){
		int I=this.ntSpecificTrainingSet.get(Integer.toString(n)+"_"+Integer.toString(t)).size();
		double sigma=1.;
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
					K.putScalar(ii, ii,sigma+nugget);
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
		if(this.ntSpecificOriginalIndices==null) {
			this.ntSpecificOriginalIndices=new HashMap<>();
		}
		INDArray weight = CheckUtil.convertFromApacheMatrix(this.weights.get(Integer.toString(n)+"_"+Integer.toString(t)),DataType.DOUBLE);
		this.ntSpecificTrainingSet.put(Integer.toString(n)+"_"+Integer.toString(t), new HashMap<>());
		Map<Integer,Data> ntTrainData=this.ntSpecificTrainingSet.get(Integer.toString(n)+"_"+Integer.toString(t));
		int i=0;
		Map<String,Double>distMap=new HashMap<>();
		List<Integer> intMap=new ArrayList<>();
		//INDArray boolArray;
		if(this.ntSpecificOriginalIndices.get(Integer.toString(n)+"_"+Integer.toString(t))!=null) {
			for(int ii:this.ntSpecificOriginalIndices.get(Integer.toString(n)+"_"+Integer.toString(t))) {
				boolean repeat=false;
				for(int jj=0;jj<i;jj++) {
					//if(ii!=jj) {// maybe an unnecessary check.... ii will never reach jj as ii>=i and jj<i only in case of ordered index of n_t specific training set
						double dist=this.calcDistance(this.trainingDataSet.get(ii).getX(), ntTrainData.get(jj).getX(), n, t);
						if(Transforms.abs(this.trainingDataSet.get(ii).getX().sub(ntTrainData.get(jj).getX()).mul(weight)).norm2Number().doubleValue()<1.0){
							repeat=true;
							break;
						}else {
							distMap.put(Integer.toString(i)+"_"+Integer.toString(jj), dist);
						}	
					//}
				}
				if(repeat==false) {
					intMap.add(ii);
					ntTrainData.put(i, this.trainingDataSet.get(ii));
					i++;
				}
			}
		}else {
			
			for(int ii=0;ii<this.trainingDataSet.size();ii++) {
				boolean repeat=false;
				for(int jj=0;jj<i;jj++) {
					if(ii!=jj) {// maybe an unnecessary check.... ii will never reach jj as ii>=i and jj<i
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
			this.distanceScale.putScalar(n,t,1.0/K.maxNumber().doubleValue()*10000);
		}
		if(intMap.size()!=K.size(0)) {
			System.out.println("Debug Here");
		}
		//KrigingInterpolator.writeINDArray(K, Integer.toString(n)+"_"+Integer.toString(t)+"distance.csv");
		return K;
	}
	
	

	public void setNtSpecificOriginalIndices(Map<String, List<Integer>> ntSpecificOriginalIndices) {
		this.ntSpecificOriginalIndices = ntSpecificOriginalIndices;
	}

	public void calcDistanceMatrix(int n, int t,double cn, double ct) {
		RealMatrix weightss=this.l2ls.generateWeightMatrix(n, t, cn, ct,this.useFlatWeightMatrix);
		INDArray weights=CheckUtil.convertFromApacheMatrix(this.l2ls.generateWeightMatrix(n, t, cn, ct,this.useFlatWeightMatrix),DataType.DOUBLE);//The only difference
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
	public INDArray calcVarianceVector(int n, int t,INDArray X, INDArray theta,INDArray nugget){
		double sigma=sigmaMatrix.getDouble(n, t);
		Map<Integer,Data> dataset=this.ntSpecificTrainingSet.get(Integer.toString(n)+"_"+Integer.toString(t));
		int i=0;
		double[][] k=new double[dataset.size()][1]; 
		for(Data dataPair:dataset.values()) {
			double v=sigma*Math.exp(-1*this.calcDistance(X, dataPair.getX(), n, t)*theta.getDouble(n, t)*this.distanceScale.getDouble(n,t))+nugget.getDouble(n,t);
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
	public Map<String,INDArray>calculateVarianceMatrixAll(INDArray theta,INDArray nugget){
		if(theta.isNaN().any()||theta.isInfinite().any()) {
			throw new IllegalArgumentException("Theta is nan or infinity!!!");
		}
		long startTime=System.currentTimeMillis();
		Map<String,INDArray> varianceMatrixAll=new ConcurrentHashMap<>();
		IntStream.rangeClosed(0,N-1).parallel().forEach((n)->
		{
			IntStream.rangeClosed(0,T-1).parallel().forEach((t)->{
				//System.out.println(Integer.toString(n)+"_"+Integer.toString(t));
				varianceMatrixAll.put(Integer.toString(n)+"_"+Integer.toString(t),this.calcVarianceMatrix(n, t, theta, nugget));
				
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
	public Map<String,INDArray>calculateVarianceVectorAll(INDArray X,INDArray theta, INDArray nugget){
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
			varianceVectorAll.put(e,this.calcVarianceVector(n, t, X, theta, nugget));
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
	
	
	
	public boolean isUseNugget() {
		return useNugget;
	}

	public void setUseNugget(boolean useNugget) {
		if(useNugget) {
			this.nugget=this.sigmaMatrix.mul(.15);
		}else {
			this.nugget=Nd4j.zeros(N,T);
		}
	}

	public String writeN_T_SpecificIndices() {
		String s="";
		String elementSeperator="";
		for(Entry<String, List<Integer>> d:this.ntSpecificOriginalIndices.entrySet()) {
			s=s+elementSeperator+d.getKey()+"___";
			String e="";
			for(int i:d.getValue()) {
				s=s+e+i;
				e=",";
			}
			elementSeperator=" ";
		}
		return s;
	}
	
	public static Map<String,List<Integer>> parseN_T_SpecificIndicies(String s){
		Map<String,List<Integer>> n_tSpecificIndices=new HashMap<>();
		for(String e1:s.split(" ")) {
			String key = e1.split("___")[0];
			ArrayList<Integer> list = new ArrayList<>();
			for(String e2:e1.split("___")[1].split(",")) {
				list.add(Integer.parseInt(e2));
			}
			n_tSpecificIndices.put(key, list);
		}
		return n_tSpecificIndices;
	}
	
}



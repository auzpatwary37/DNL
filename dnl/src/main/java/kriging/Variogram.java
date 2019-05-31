package kriging;

import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.stream.IntStream;

import org.apache.commons.math3.stat.descriptive.moment.StandardDeviation;
import org.matsim.core.utils.collections.Tuple;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import linktolinkBPR.LinkToLinks;



/**
 * This class will calculate the required covariance marices 
 * @author Ashraf
 *
 */
public class Variogram {
	
	private final Map<Integer,Tuple<INDArray,INDArray>> trainingDataSet;
	private final INDArray sigmaMatrix;
	private final Map<String,INDArray> weights;
	private INDArray theta;
	private Map<String,INDArray> varianceMapAll;
	private final int N;
	private final int T;
	private final int I;
	
	
	//TODO: Add a writer to save the trained model
	
	/**
	 * This will initialize the theta matrix and calculate and store the IxI variance matrix by default
	 * @param trainingDataSet
	 * @param l2ls
	 */
	public Variogram(Map<Integer,Tuple<INDArray,INDArray>>trainingDataSet,LinkToLinks l2ls) {
		this.trainingDataSet=trainingDataSet;
		this.weights=l2ls.getWeightMatrices();
		this.N=Math.toIntExact(trainingDataSet.get(0).getFirst().size(0));
		this.T=Math.toIntExact(trainingDataSet.get(0).getFirst().size(1));
		this.I=trainingDataSet.size();
		this.sigmaMatrix=this.calcSigmaMatrix(trainingDataSet);
		//Initialize theta to a nxt matrix of one
		this.theta=Nd4j.zeros(N,T).addi(1);
		this.varianceMapAll=this.calculateVarianceMatrixAll(this.theta);
	}
	
	public Variogram(Map<Integer,Tuple<INDArray,INDArray>>trainingDataSet,Map<String,INDArray>weights,INDArray theta) {
		this.trainingDataSet=trainingDataSet;
		this.N=Math.toIntExact(trainingDataSet.get(0).getFirst().size(0));
		this.T=Math.toIntExact(trainingDataSet.get(0).getFirst().size(1));
		this.I=trainingDataSet.size();
		this.weights=weights;
		this.sigmaMatrix=this.calcSigmaMatrix(trainingDataSet);
		//Initialize theta to a nxt matrix of one
		this.theta=theta;
		this.varianceMapAll=this.calculateVarianceMatrixAll(this.theta);
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
	public INDArray calcSigmaMatrix(Map<Integer,Tuple<INDArray,INDArray>>trainingDataSet) {
		INDArray sd=Nd4j.create(N,T);
		StandardDeviation sdCalculator= new StandardDeviation();
		IntStream.rangeClosed(0,N-1).parallel().forEach((n)->
		{
			IntStream.rangeClosed(0,T-1).parallel().forEach((t)->{
				double[] data=new double[trainingDataSet.size()];
				for(int i=0;i<trainingDataSet.size();i++) {
					data[i]=trainingDataSet.get(i).getSecond().getDouble(n, t);
				}
				sd.putScalar(n, t, Math.pow(sdCalculator.evaluate(data),2));
			});
		});
		return sd;
	}
	
	public double calcDistance(INDArray a,INDArray b,int n,int t) {
		double aa=a.sub(b).mul(this.weights.get(Integer.toString(n)+"_"+Integer.toString(t))).norm2Number().doubleValue();
		return aa;
		 //return distance;
	}
	
	//For now this function is useless but it can be later used to make the weights trainable
	public double calcDistance(INDArray a,INDArray b,Map<String,INDArray> weights,int n,int t) {
		return a.sub(b).mul(weights.get(Integer.toString(n)+"_"+Integer.toString(t))).norm2Number().doubleValue();
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
		long startTime=System.currentTimeMillis();
		double sigma=sigmaMatrix.getDouble(n, t);
		INDArray K=Nd4j.create(I,I);
		int i=0;
//		IntStream.rangeClosed(0,I-1).parallel().forEach((ii)->
//		{
//			IntStream.rangeClosed(0,ii).parallel().forEach((jj)->{
//
//				if(ii!=jj) {
//					double v=sigma*Math.exp(-1*this.calcDistance(this.trainingDataSet.get(ii).getFirst(), this.trainingDataSet.get(jj).getFirst(), n, t)*theta.getDouble(n, t));
//					K.putScalar(ii, jj, v);
//					K.putScalar(jj, ii, v);
//				}else {
//					K.putScalar(ii, ii,sigma);
//				}
//			});
//		});
		
		for(int ii=0;ii<I;ii++) {
			for(int jj=0;jj<=ii;jj++) {
				if(ii!=jj) {
					double v=sigma*Math.exp(-1*this.calcDistance(this.trainingDataSet.get(ii).getFirst(), this.trainingDataSet.get(jj).getFirst(), n, t)*theta.getDouble(n, t));
					K.putScalar(ii, jj, v);
					K.putScalar(jj, ii, v);
				}else {
					K.putScalar(ii, ii,sigma);
				}
			}
		}
		
		
		long endTime=System.currentTimeMillis();
		System.out.println("Took time = "+(endTime-startTime));
		return K;
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
		INDArray K=Nd4j.create(I,1);
		int i=0;
		for(Tuple<INDArray,INDArray> dataPair:trainingDataSet.values()) {
			double v=sigma*Math.exp(-1*this.calcDistance(X, dataPair.getFirst(), n, t)*theta.getDouble(n, t));
			K.putScalar(i, 1, v);
			i++;
		}
		return K;
	}
	
	/**
	 * calculate IxI covariance matrix for all n x t outputs
	 * @param theta
	 * @return the map with n_t -> IxI realMatrix covariance matrix
	 */
	public Map<String,INDArray>calculateVarianceMatrixAll(INDArray theta){
		long startTime=System.currentTimeMillis();
		Map<String,INDArray> varianceMatrixAll=new ConcurrentHashMap<>();
		IntStream.rangeClosed(0,N-1).parallel().forEach((n)->
		{
			IntStream.rangeClosed(0,T-1).parallel().forEach((t)->{
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
		IntStream.rangeClosed(0,N-1).parallel().forEach((n)->
		{
			IntStream.rangeClosed(0,T-1).parallel().forEach((t)->{
				varianceVectorAll.put(Integer.toString(n)+"_"+Integer.toString(t),this.calcVarianceVector(n, t, X, theta));
			});
		});
		return varianceVectorAll;
	}
	/**
	 * Update theta and recalculate the IxI variance matrix
	 * @param theta
	 */
	public void updatetheta(INDArray theta) {
		this.theta=theta;
		this.varianceMapAll=this.calculateVarianceMatrixAll(this.theta);
	}
	
	//Getters and Setters

	public Map<Integer, Tuple<INDArray, INDArray>> getTrainingDataSet() {
		return trainingDataSet;
	}


	public INDArray getSigmaMatrix() {
		return sigmaMatrix;
	}

	public INDArray gettheta() {
		return theta;
	}

	public Map<String, INDArray> getVarianceMapAll() {
		return varianceMapAll;
	}

	public Map<String, INDArray> getWeights() {
		return weights;
	}
	
	
	
}



package kriging;

import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.stream.IntStream;

import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.linear.OpenMapRealMatrix;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.stat.descriptive.moment.StandardDeviation;
import org.matsim.core.utils.collections.Tuple;
import org.nd4j.linalg.checkutil.CheckUtil;
import org.nd4j.linalg.factory.Nd4j;

import linktolinkBPR.LinkToLinks;

/**
 * This class will calculate the required covariance marices 
 * @author Ashraf
 *
 */
public class Variogram {
	private final Map<Integer,Tuple<RealMatrix,RealMatrix>> trainingDataSet;
	private final LinkToLinks l2ls;
	private final RealMatrix sigmaMatrix;
	private RealMatrix beta;
	private Map<String,RealMatrix> varianceMapAll;
	
	public Variogram(Map<Integer,Tuple<RealMatrix,RealMatrix>>trainingDataSet,LinkToLinks l2ls) {
		this.trainingDataSet=trainingDataSet;
		this.l2ls=l2ls;
		this.sigmaMatrix=this.calcSigmaMatrix(trainingDataSet);
		//Initialize beta to a nxt matrix of one
		this.beta=CheckUtil.convertToApacheMatrix((Nd4j.zeros(trainingDataSet.get(0).getFirst().getRowDimension(),trainingDataSet.get(0).getFirst().getColumnDimension()).addi(1)));
		this.varianceMapAll=this.calculateVarianceMatrixAll(this.beta);
	}
	
	/**
	 * 
	 * @param a first tensor
	 * @param b second tensor
	 * @param l2ls Network info
	 * @param n link in calculation
	 * @param t time in calculation
	 * @param ka number of connected l2l to consider
	 * @param kt number of connected time to consider
	 * @param beta parameters for the variogram
	 * @return
	 */
	public RealMatrix calcSigmaMatrix(Map<Integer,Tuple<RealMatrix,RealMatrix>>trainingDataSet) {
		RealMatrix sd=new Array2DRowRealMatrix(trainingDataSet.get(0).getFirst().getRowDimension(),trainingDataSet.get(0).getFirst().getColumnDimension());
		StandardDeviation sdCalculator= new StandardDeviation();
		IntStream.rangeClosed(0,trainingDataSet.get(0).getSecond().getRowDimension()-1).forEach((n)->
		{
			IntStream.rangeClosed(0,trainingDataSet.get(0).getSecond().getColumnDimension()-1).forEach((t)->{
				double[] data=new double[trainingDataSet.size()];
				for(int i=0;i<trainingDataSet.size();i++) {
					data[i]=trainingDataSet.get(i).getSecond().getEntry(n, t);
				}
				sd.setEntry(n, t, Math.pow(sdCalculator.evaluate(data),2));
			});
		});
		return sd;
	}
	
	public double calcDistance(RealMatrix a,RealMatrix b,LinkToLinks l2ls,int n,int t) {
		return Nd4j.create(a.getData()).sub(Nd4j.create(b.getData())).mul(Nd4j.create(l2ls.getWeightMatrix(n, t).getData())).norm2Number().doubleValue();
		//return distance;
	}
	
	//For now this function is useless but it can be later used to make the weights trainable
	public double calcDistance(RealMatrix a,RealMatrix b,Map<String,RealMatrix> weights,int n,int t) {
		return Nd4j.create(a.getData()).sub(Nd4j.create(b.getData())).mul(Nd4j.create(weights.get(Integer.toString(n)+"_"+Integer.toString(t)).getData())).norm2Number().doubleValue();
		//return distance;
	}
	/**
	 * This function will calculate the K matrix for a (n,t) pair the dimension of the matrix is IxI where I is the number of data point.
	 * THe auto co-variance function is sigma^2exp(-1*distance*beta)
	 * @param n the l2l id 
	 * @param t time id  
	 * @return K dim IxI
	 */
	public RealMatrix calcVarianceMatrix(int n, int t,RealMatrix beta){
		double sigma=sigmaMatrix.getEntry(n, t);
		RealMatrix K=new OpenMapRealMatrix(trainingDataSet.size(),trainingDataSet.size());
		int i=0;
		int j=0;
		for(Tuple<RealMatrix,RealMatrix> dataPair1:trainingDataSet.values()) {
			for(Tuple<RealMatrix,RealMatrix> dataPair2:trainingDataSet.values()) {
				if(i==j) {
					K.setEntry(i, i, sigma);
				}else {
					double v=sigma*Math.exp(-1*this.calcDistance(dataPair1.getFirst(), dataPair2.getFirst(), this.l2ls, n, t)*beta.getEntry(n, t));
					K.setEntry(i, j, v);
					K.setEntry(j, i, v);
				}
				j++;
			}
			i++;
		}
		return K;
	}
	/**
	 * Similar to calcVarianceMatrix but it will calculate variance for a particular point to all other point
	 * @param n
	 * @param t
	 * @param X
	 * @param beta
	 * @return the covariance vector with dimension Ix1
	 */
	public RealMatrix calcVarianceVector(int n, int t,RealMatrix X, RealMatrix beta){
		double sigma=sigmaMatrix.getEntry(n, t);
		RealMatrix K=new Array2DRowRealMatrix(trainingDataSet.size(),1);
		int i=0;
		for(Tuple<RealMatrix,RealMatrix> dataPair:trainingDataSet.values()) {
			double v=sigma*Math.exp(-1*this.calcDistance(X, dataPair.getFirst(), this.l2ls, n, t)*beta.getEntry(n, t));
			K.setEntry(i, 1, v);
			i++;
		}
		return K;
	}
	
	/**
	 * calculate IxI covariance matrix for all n x t outputs
	 * @param beta
	 * @return the map with n_t -> IxI realMatrix covariance matrix
	 */
	public Map<String,RealMatrix>calculateVarianceMatrixAll(RealMatrix beta){
		Map<String,RealMatrix> varianceMatrixAll=new ConcurrentHashMap<>();
		IntStream.rangeClosed(0,trainingDataSet.get(0).getSecond().getRowDimension()-1).forEach((n)->
		{
			IntStream.rangeClosed(0,trainingDataSet.get(0).getSecond().getColumnDimension()-1).forEach((t)->{
				varianceMatrixAll.put(Integer.toString(n)+"_"+Integer.toString(t),this.calcVarianceMatrix(n, t, beta));
			});
		});
		return varianceMatrixAll;
	}
	
	/**
	 * calculate Ix1 covariance vector for all n x t outputs
	 * @param beta
	 * @return the map with n_t -> Ix1 realMatrix covariance matrix
	 */
	public Map<String,RealMatrix>calculateVarianceVectorAll(RealMatrix X,RealMatrix beta){
		Map<String,RealMatrix> varianceVectorAll=new ConcurrentHashMap<>();
		IntStream.rangeClosed(0,trainingDataSet.get(0).getSecond().getRowDimension()-1).forEach((n)->
		{
			IntStream.rangeClosed(0,trainingDataSet.get(0).getSecond().getColumnDimension()-1).forEach((t)->{
				varianceVectorAll.put(Integer.toString(n)+"_"+Integer.toString(t),this.calcVarianceVector(n, t, X, beta));
			});
		});
		return varianceVectorAll;
	}
	/**
	 * Update beta and recalculate the IxI variance matrix
	 * @param beta
	 */
	public void updateBeta(RealMatrix beta) {
		this.beta=beta;
		this.varianceMapAll=this.calculateVarianceMatrixAll(this.beta);
	}
}

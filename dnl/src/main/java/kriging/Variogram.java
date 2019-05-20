package kriging;

import java.util.Map;
import java.util.stream.IntStream;

import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.linear.OpenMapRealMatrix;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.stat.descriptive.moment.StandardDeviation;
import org.matsim.core.utils.collections.Tuple;
import org.nd4j.linalg.factory.Nd4j;

import linktolinkBPR.LinkToLinks;

/**
 * This class will calculate the distance between two input of dimension nxt where n is the number of link2link and t is the time slice
 * @author Ashraf
 *
 */
public class Variogram {
	private LinkToLinks l2ls;
	private RealMatrix sigmaMatrix;
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
	public void calcSigmaMatrix(Map<Integer,Tuple<RealMatrix,RealMatrix>>trainingDataSet,double[][]beta) {
		RealMatrix sd=new Array2DRowRealMatrix();
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
		this.sigmaMatrix=sd;
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
	 * THe auto co-variance function is sigma^2exp(-1*distance)
	 * @param n the l2l id 
	 * @param t time id 
	 * @param trainingDataSet a map containing the training data in id -> (x,y) mapping where x and y are in RealMatrix format 
	 * @return K dim IxI
	 */
	public RealMatrix calcVarianceMatrix(int n, int t,Map<Integer,Tuple<RealMatrix,RealMatrix>>trainingDataSet,double[][]beta){
		double sigma=sigmaMatrix.getEntry(n, t);
		RealMatrix K=new OpenMapRealMatrix(trainingDataSet.size(),trainingDataSet.size());
		int i=0;
		int j=0;
		for(Tuple<RealMatrix,RealMatrix> dataPair1:trainingDataSet.values()) {
			for(Tuple<RealMatrix,RealMatrix> dataPair2:trainingDataSet.values()) {
				if(i==j) {
					K.setEntry(i, i, sigma);
				}else {
					double v=sigma*Math.exp(-1*this.calcDistance(dataPair1.getFirst(), dataPair2.getFirst(), this.l2ls, n, t)*beta[n][t]);
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
	 * @param trainingDataSet
	 * @param beta
	 * @return the covariance vector with dimension Ix1
	 */
	public RealMatrix calcVarianceVector(int n, int t,RealMatrix X, Map<Integer,Tuple<RealMatrix,RealMatrix>>trainingDataSet,double[][]beta){
		double sigma=sigmaMatrix.getEntry(n, t);
		RealMatrix K=new OpenMapRealMatrix(trainingDataSet.size(),1);
		int i=0;
		for(Tuple<RealMatrix,RealMatrix> dataPair:trainingDataSet.values()) {
			double v=sigma*Math.exp(-1*this.calcDistance(X, dataPair.getFirst(), this.l2ls, n, t)*beta[n][t]);
			K.setEntry(i, 1, v);
			i++;
		}
		return K;
	}
}

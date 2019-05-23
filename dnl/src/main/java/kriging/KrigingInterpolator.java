package kriging;

import java.lang.reflect.InvocationTargetException;
import java.lang.reflect.Method;
import java.util.HashMap;
import java.util.Map;
import java.util.Map.Entry;
import java.util.concurrent.ConcurrentHashMap;
import java.util.stream.IntStream;

import org.apache.commons.math3.linear.LUDecomposition;
import org.apache.commons.math3.linear.RealMatrix;
import org.matsim.core.utils.collections.Tuple;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.checkutil.CheckUtil;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.INDArrayIndex;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.linalg.inverse.InvertMatrix;

import linktolinkBPR.LinkToLinks;

/**
 * 
 * A kriging framework
 * @author Ashraf
 *
 */
public class KrigingInterpolator{
	private final Map<Integer,Tuple<INDArray,INDArray>> trainingDataSet;
	private LinkToLinks l2ls;
	private Variogram variogram;
	private INDArray beta;
	private BaseFunction baseFunction;
	private final int N;
	private final int T;
	private int I;
	
	
	public KrigingInterpolator(Map<Integer,Tuple<INDArray,INDArray>> trainingDataSet, LinkToLinks l2ls) {
		this.trainingDataSet=trainingDataSet;
		this.l2ls=l2ls;
		this.variogram=new Variogram(trainingDataSet, this.l2ls);
		this.N=Math.toIntExact(trainingDataSet.get(0).getFirst().size(0));
		this.T=Math.toIntExact(trainingDataSet.get(0).getFirst().size(1));
		this.I=trainingDataSet.size();
	}
	
	
	//Constructor for creating in the reader
	public KrigingInterpolator(Variogram v, INDArray beta, BaseFunction bf) {
		this.trainingDataSet=v.getTrainingDataSet();
		this.variogram=v;
		this.beta=beta;
		this.variogram.getClass().toString();
		this.baseFunction=bf;
		this.N=Math.toIntExact(trainingDataSet.get(0).getFirst().size(0));
		this.T=Math.toIntExact(trainingDataSet.get(0).getFirst().size(1));
		this.I=trainingDataSet.size();
	}
	
	public Map<Integer, Tuple<INDArray, INDArray>> getTrainingDataSet() {
		return trainingDataSet;
	}


	public Variogram getVariogram() {
		return variogram;
	}


	public INDArray getBeta() {
		return beta;
	}


	public BaseFunction getBaseFunction() {
		return baseFunction;
	}
	
	public void updateVariogramParameter(INDArray theta) {
		this.variogram.updatetheta(theta);
	}
	
	public INDArray getY(INDArray X,VarianceInfoHolder info) {
		return this.getY(X, this.beta, this.variogram.gettheta(),info);
	}
	
	public VarianceInfoHolder preProcessData(INDArray beta,INDArray theta) {
		//This is the Z-MB
		INDArray Z_MB=Nd4j.create(this.N,this.T,this.I);
		Map<String,INDArray> varianceMatrixAll=this.variogram.calculateVarianceMatrixAll(theta);
		Map<String,INDArray> varianceMatrixInverseAll=this.variogram.calculateVarianceMatrixAll(theta);
		for(Entry<String,INDArray> n_t_K:varianceMatrixAll.entrySet()) {
			varianceMatrixInverseAll.put(n_t_K.getKey(),InvertMatrix.pinvert(n_t_K.getValue(), false));
		}
		for(Entry<Integer,Tuple<INDArray,INDArray>>dataPoint:this.trainingDataSet.entrySet()) {
			Z_MB.put(new INDArrayIndex[] {NDArrayIndex.all(),NDArrayIndex.all(),NDArrayIndex.point(dataPoint.getKey())},dataPoint.getValue().getSecond().sub(this.baseFunction.getY(dataPoint.getValue().getFirst()).mul(beta)));
		}
		
		return new VarianceInfoHolder(Z_MB,varianceMatrixAll,varianceMatrixInverseAll);
	}
	
	public INDArray getY(INDArray X,INDArray beta,INDArray theta,VarianceInfoHolder info) {
		INDArray Y=Nd4j.create(X.size(0),X.size(1));
		//This is m(x_0);
		INDArray Y_b=this.baseFunction.getY(X);
		INDArray Z_MB=info.getZ_MB();
		Map<String,INDArray> varianceVectorAll=this.variogram.calculateVarianceVectorAll(X, theta);
		IntStream.rangeClosed(0,Math.toIntExact(X.size(0))-1).forEach((n)->
		{
			IntStream.rangeClosed(0,Math.toIntExact(X.size(1))-1).forEach((t)->{
				String key=Integer.toString(n)+"_"+Integer.toString(t);
				INDArray KInverse=info.getVarianceMatrixInverseAll().get(key);
				double y=Y_b.getDouble(n,t)*beta.getDouble(n,t)+
						varianceVectorAll.get(key).mmul(KInverse).mmul(Z_MB.get(new INDArrayIndex[] {NDArrayIndex.point(n),NDArrayIndex.point(t),NDArrayIndex.all()})).getDouble(0,0);
				Y.putScalar(n,t,y);
			});
		});
		return X;
	}
	
	
	
	//Intentionally not made parallel
	public double calcCombinedLogLikelihood(INDArray theta, INDArray beta) {
		double logLikelihood=0;
		VarianceInfoHolder info=this.preProcessData(beta, theta);
		
		for(int n=0;n<this.N;n++) {
			for(int t=0;t<this.T;t++) {
				String key=Integer.toString(n)+"_"+Integer.toString(t);
				INDArray Z_MB=info.getZ_MB().get(new INDArrayIndex[] {NDArrayIndex.point(n),NDArrayIndex.point(t),NDArrayIndex.all()});
				double d=-1*this.I/2.0*Math.log(2*Math.PI)-
						.5*Math.log(new LUDecomposition(CheckUtil.convertToApacheMatrix(info.getVarianceMatrixAll().get(key))).getDeterminant())
						-.5*Z_MB.mmul(info.getVarianceMatrixInverseAll().get(key)).mmul(Z_MB).getDouble(0,0);
				logLikelihood+=d;
			}
		}

		return logLikelihood;
	}
	
	public double calcCombinedLogLikelihood() {
		return this.calcCombinedLogLikelihood(this.variogram.gettheta(),this.getBeta());
	}
	public static void main(String[] args) throws NoSuchMethodException, SecurityException, ClassNotFoundException, IllegalAccessException, IllegalArgumentException, InvocationTargetException {
		BaseFunction bf=new MeanBaseFunction();
		String s=bf.getClass().getName().toString();
		System.out.println(s);
		Class.forName(s);
		Method parse=Class.forName(s).getMethod("shout");
		parse.invoke(null);
	}
	

}

class VarianceInfoHolder{
	private final INDArray Z_MB;
	private final Map<String,INDArray>varianceMatrixAll; 
	private final Map<String,INDArray> varianceMatrixInverseAll;
	
	public VarianceInfoHolder(INDArray Z_MB,Map<String,INDArray>varianceMatrixAll,Map<String,INDArray> varianceMatrixInverseAll) {
		this.Z_MB=Z_MB;
		this.varianceMatrixAll=varianceMatrixAll;
		this.varianceMatrixInverseAll=varianceMatrixInverseAll;
	}

	public INDArray getZ_MB() {
		return Z_MB;
	}

	public Map<String, INDArray> getVarianceMatrixInverseAll() {
		return varianceMatrixInverseAll;
	}

	public Map<String, INDArray> getVarianceMatrixAll() {
		return varianceMatrixAll;
	}
	
}




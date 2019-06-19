package kriging;

import java.awt.image.DataBuffer;
import java.lang.reflect.InvocationTargetException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Vector;
import java.util.concurrent.ConcurrentHashMap;
import java.util.stream.IntStream;

import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.SingularValueDecomposition;
import org.matsim.api.core.v01.network.Network;
import org.matsim.core.network.NetworkUtils;
import org.matsim.core.utils.collections.Tuple;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.buffer.util.DataTypeUtil;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.INDArrayIndex;
import org.nd4j.linalg.indexing.NDArrayIndex;

import de.xypron.jcobyla.Cobyla;
import de.xypron.jcobyla.Calcfc;
import de.xypron.jcobyla.CobylaExitStatus;
import linktolinkBPR.LinkToLinks;
import linktolinkBPR.SignalFlowReductionGenerator;
import training.DataIO;
import training.Evaluator;
import training.FunctionSPSA;

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
	private VarianceInfoHolder info;
	
	
	public KrigingInterpolator(Map<Integer,Tuple<INDArray,INDArray>> trainingDataSet, LinkToLinks l2ls,BaseFunction bf) {
		this.trainingDataSet=trainingDataSet;
		this.l2ls=l2ls;
		this.baseFunction=bf;
		this.variogram=new Variogram(trainingDataSet, this.l2ls);
		this.N=Math.toIntExact(trainingDataSet.get(0).getFirst().size(0));
		this.T=Math.toIntExact(trainingDataSet.get(0).getFirst().size(1));
		this.beta=Nd4j.zeros(N,T).addi(1);
		this.info=this.preProcessData();
	}
	
	//TODO: the reading and writing are very messed up right now. Will fix soon. [Ashraf, June 2019].
	//Constructor for creating in the reader
	public KrigingInterpolator(Variogram v, INDArray beta, BaseFunction bf) {
		this.trainingDataSet=v.getTrainingDataSet();
		this.variogram=v;
		this.beta=beta;
		this.variogram.getClass().toString();
		this.baseFunction=bf;
		this.N=Math.toIntExact(trainingDataSet.get(0).getFirst().size(0));
		this.T=Math.toIntExact(trainingDataSet.get(0).getFirst().size(1));
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
	
	public INDArray getY(INDArray X) {
		return this.getY(X, this.beta, this.variogram.gettheta(),info);
	}
	
	public VarianceInfoHolder preProcessData(INDArray beta,INDArray theta) {
		long startTime=System.currentTimeMillis();
		//This is the Z-MB
		INDArray Z_MB=Nd4j.create(this.N,this.T,this.trainingDataSet.size());
		Map<String,INDArray> varianceMatrixAll=this.variogram.calculateVarianceMatrixAll(theta);
		Map<String,INDArray> varianceMatrixInverseAll=new ConcurrentHashMap<>();
		Map<String,double[]> singularValuesAll=new ConcurrentHashMap<>();
		varianceMatrixAll.entrySet().parallelStream().forEach((n_t_K)->{
			//n_t_K.getValue()
			SingularValueDecomposition svd=new SingularValueDecomposition(MatrixUtils.createRealMatrix(n_t_K.getValue().toDoubleMatrix()));
			RealMatrix inv=svd.getSolver().getInverse();
			double[] singularValues=svd.getSingularValues();
			varianceMatrixInverseAll.put(n_t_K.getKey(),Nd4j.create(toFloatArray(inv.getData()))) ;
			singularValuesAll.put(n_t_K.getKey(), singularValues);
		});
		for(Entry<Integer,Tuple<INDArray,INDArray>>dataPoint:this.trainingDataSet.entrySet()) {
			Z_MB.put(new INDArrayIndex[] {NDArrayIndex.all(),NDArrayIndex.all(),NDArrayIndex.point(dataPoint.getKey())},dataPoint.getValue().getSecond().sub(this.baseFunction.getY(dataPoint.getValue().getFirst()).mul(beta)).mul(this.variogram.getTtScale()));// the Y scale is directly applied on Z-MB
		}
		System.out.println("Total Time for info preperation (Inversing and SVD) = "+Long.toString(System.currentTimeMillis()-startTime));
		return new VarianceInfoHolder(Z_MB,varianceMatrixAll,varianceMatrixInverseAll,singularValuesAll);
	}
	
	public VarianceInfoHolder preProcessNtSpecificData(int n, int t,double beta,double theta,VarianceInfoHolder info) {
		long startTime=System.currentTimeMillis();
		//This is the Z-MB
		INDArray Z_MB=info.getZ_MB();
		INDArray varianceMatrix=this.variogram.calcVarianceMatrix(n, t, theta);
		String key=Integer.toString(n)+"_"+Integer.toString(t);
		info.getVarianceMatrixAll().put(key, varianceMatrix);
		if(info.getVarianceMatrixAll().get(key)==null) {
			System.out.println();
		}
		SingularValueDecomposition svd=new SingularValueDecomposition(MatrixUtils.createRealMatrix(info.getVarianceMatrixAll().get(key).toDoubleMatrix()));
		RealMatrix inv=svd.getSolver().getInverse();
		double[] singularValues=svd.getSingularValues();
		info.getVarianceMatrixInverseAll().put(key,Nd4j.create(toFloatArray(inv.getData()))) ;
		info.getSingularValues().put(key, singularValues);
		//Only the n_t has been changed
		for(Entry<Integer,Tuple<INDArray,INDArray>>dataPoint:this.trainingDataSet.entrySet()) {
			Z_MB.putScalar(new int[] {n,t,dataPoint.getKey()},(dataPoint.getValue().getSecond().getDouble(n,t)-this.baseFunction.getY(dataPoint.getValue().getFirst()).getDouble(n,t)*beta)*this.variogram.getTtScale().getDouble(n,t));// the Y scale is directly applied on Z-MB
		}
		//System.out.println("Total Time for info preperation (Inversing and SVD) = "+Long.toString(System.currentTimeMillis()-startTime));
		return info;
	}
	
	public INDArray getY(INDArray X,INDArray beta,INDArray theta,VarianceInfoHolder info) {
		INDArray Y=Nd4j.create(X.size(0),X.size(1));
		//This is m(x_0);
		INDArray Y_b=this.baseFunction.getY(X);
		INDArray Z_MB=info.getZ_MB();
		
		Map<String,INDArray> varianceVectorAll=this.variogram.calculateVarianceVectorAll(X, theta);
		IntStream.rangeClosed(0,Math.toIntExact(X.size(0))-1).parallel().forEach((n)->
		{
			IntStream.rangeClosed(0,Math.toIntExact(X.size(1))-1).parallel().forEach((t)->{
				String key=Integer.toString(n)+"_"+Integer.toString(t);
				INDArray z_mb=Nd4j.create(this.variogram.getNtSpecificTrainingSet().get(key).size(),1);
				INDArray KInverse=info.getVarianceMatrixInverseAll().get(key);
				for(int i=0;i<z_mb.size(0);i++) {
					z_mb.putScalar(i, 0,Z_MB.getDouble(n,t,this.variogram.getNtSpecificOriginalIndices().get(key).get(i)));
				}
				double y=Y_b.getDouble(n,t)*beta.getDouble(n,t)+
						varianceVectorAll.get(key).mmul(KInverse).mmul(z_mb).getDouble(0,0)/this.variogram.getTtScale().getDouble(n,t);//Fix the Z_MB part!!!
				Y.putScalar(n,t,y);
			});
		});
		return Y;
	}
	
	public static float[][] toFloatArray(double[][] arr) {
		  if (arr == null) return null;
		  int n = arr.length;
		  float[][] ret = new float[n][arr[0].length];
		  for (int i = 0; i < n; i++) {
			  for(int j=0;j<arr[i].length;j++) {
				  ret[i][j] = (float)arr[i][j];
			  }
		  }
		  return ret;
		}
	
	//Intentionally not made parallel
	public double calcCombinedLogLikelihood(INDArray theta, INDArray beta) {
		VarianceInfoHolder info=this.preProcessData(beta, theta);
		INDArray logLiklihood=Nd4j.create(this.N,this.T);
		for(int n=0;n<this.N;n++) {
			for(int t=0;t<this.T;t++) {
				String key=Integer.toString(n)+"_"+Integer.toString(t);
				INDArray Z_MB=Nd4j.create(this.variogram.getNtSpecificTrainingSet().get(key).size(),1);
				for(int j=0;j<Z_MB.size(0);j++) {
					Z_MB.putScalar(j,0,info.getZ_MB().getDouble(n,t,this.variogram.getNtSpecificOriginalIndices().get(key).get(j)));
				}
				double Logdet_k=0;
				//log here for ease

				
				for(double dd:info.getSingularValues().get(key)) {
					if(dd!=0) {
						Logdet_k+=Math.log(dd);
					}
				}
				INDArray secondLLTerm=Z_MB.transpose().mmul(info.getVarianceMatrixInverseAll().get(key)).mmul(Z_MB);
				double d=-1*Z_MB.size(0)/2.0*Math.log(2*Math.PI)-
						.5*Logdet_k
						-.5*secondLLTerm.getDouble(0,0);
				if(d==Double.NaN) {
					System.out.println("case NAN");

				}else if(d==Double.NEGATIVE_INFINITY){
					System.out.println("Negative Infinity");
				}else if (d==Double.POSITIVE_INFINITY) {
					System.out.println("");
				}
				logLiklihood.putScalar(n,t,d);
			}
		}
		System.out.println("Complete");
		double sum=logLiklihood.sumNumber().doubleValue();
		return sum;
	}
	
	/**
	 * This decouples the optimization process to only theta and beta specific to one n_t pair
	 * @param n
	 * @param t
	 * @param theta
	 * @param beta
	 * @param info
	 * @return
	 */
	public double calcNtSpecificLogLikelihood(int n, int t,double theta, double beta,VarianceInfoHolder info) {
		this.preProcessNtSpecificData(n,t,beta, theta,info);
		double logLiklihood=0;
		String key=Integer.toString(n)+"_"+Integer.toString(t);
		INDArray Z_MB=Nd4j.create(this.variogram.getNtSpecificTrainingSet().get(key).size(),1);
		for(int j=0;j<Z_MB.size(0);j++) {
			Z_MB.putScalar(j,0,info.getZ_MB().getDouble(n,t,this.variogram.getNtSpecificOriginalIndices().get(key).get(j)));
		}
		double Logdet_k=0;
		//log here for ease	
		for(double dd:info.getSingularValues().get(key)) {
			if(dd!=0) {
				Logdet_k+=Math.log(dd);
			}
		}
		INDArray secondLLTerm=Z_MB.transpose().mmul(info.getVarianceMatrixInverseAll().get(key)).mmul(Z_MB);
		double d=-1*Z_MB.size(0)/2.0*Math.log(2*Math.PI)-
				.5*Logdet_k
				-.5*secondLLTerm.getDouble(0,0);
		if(d==Double.NaN) {
			System.out.println("case NAN");

		}else if(d==Double.NEGATIVE_INFINITY){
			System.out.println("Negative Infinity");
		}else if (d==Double.POSITIVE_INFINITY) {
			System.out.println("");
		}
		logLiklihood=d;
		//System.out.println("Complete");
		return logLiklihood;
	}
	
	public double calcCombinedLogLikelihood() {
		return this.calcCombinedLogLikelihood(this.variogram.gettheta(),this.getBeta());
	}
	
	public static void main(String[] args) throws NoSuchMethodException, SecurityException, ClassNotFoundException, IllegalAccessException, IllegalArgumentException, InvocationTargetException {
		DataTypeUtil.setDTypeForContext(DataType.FLOAT);
		Nd4j.setDefaultDataTypes(DataType.FLOAT, DataType.FLOAT);
		Map<Integer,Tuple<INDArray,INDArray>> trainingData=DataIO.readDataSet("Network/ND/DataSetNDTrain.txt");
		Network network=NetworkUtils.readNetwork("Network/ND/ndNetwork.xml");
		//Network network=NetworkUtils.readNetwork("Network/SiouxFalls/network.xml");
		SignalFlowReductionGenerator sg = null;
		//config.network().setInputFile("Network/SiouxFalls/network.xml");
		Map<Integer,Tuple<Double,Double>> timeBean=new HashMap<>();
		for(int i=15;i<24;i++) {
			timeBean.put(i,new Tuple<Double,Double>(i*3600.,i*3600.+3600));
		}
		LinkToLinks l2ls=new LinkToLinks(network,timeBean,3,3,sg);
		KrigingInterpolator kriging=new KrigingInterpolator(trainingData, l2ls, new MeanBaseFunction(trainingData));
		System.out.println(kriging.calcCombinedLogLikelihood());
		//System.out.println("Finished!!!");
		//System.out.println(kriging.calcCombinedLogLikelihood());
		kriging.trainKriging();
		System.out.println(kriging.calcCombinedLogLikelihood());
		
		
	}
	
	public void trainKriging() {

//		int n=0;
//		int t=0;
		List<String> n_tlist=new ArrayList<>();
		for(int n=0;n<N;n++) {
			for(int t=0;t<T;t++) {
				n_tlist.add(Integer.toString(n)+"_"+Integer.toString(t));
			}
		}
		n_tlist.parallelStream().forEach((key)->{
		//for(String key:n_tlist) {
			int n=Integer.parseInt(key.split("_")[0]);
			int t=Integer.parseInt(key.split("_")[1]);
		
		
				Calcfc calcfc = new Calcfc() {
					int it=0;

					@Override
					public double compute(int N, int m, double[] x, double[] con) {
						double theta=x[0];
						double beta=x[1];
						double obj=KrigingInterpolator.this.calcNtSpecificLogLikelihood(n, t, theta, beta,info);
						if(theta==0) {
							obj=10000000000000.;
						}
						con[0]=x[0];
						it++;
						if(it==1) {
							System.out.println("initial obj = "+-1*obj);
						}
						return -1*obj;
					}
				};
				double[] x = {1.0, 1.0 };
				CobylaExitStatus result = Cobyla.findMinimum(calcfc, 2, 1, x, 0.5, .001, 1, 800);
				this.beta.putScalar(n, t,x[1]);
				this.variogram.gettheta().putScalar(n,t,x[0]);
				//System.out.println("current total liklihood after "+key+" = "+this.calcCombinedLogLikelihood());
		});
		//}
		//KrigingModelWriter writer=new KrigingModelWriter(this);
		//writer.writeModel("Network/ND/Model1/");

	}
	
	private VarianceInfoHolder preProcessData() {
		return this.preProcessData(this.beta,this.variogram.gettheta());
	}


	private double[] scaleToVector(INDArray beta, INDArray theta) {
		INDArray linearBeta=beta.reshape(beta.length());
		INDArray linearTheta=theta.reshape(theta.length());
		return Nd4j.concat(0, linearBeta,linearTheta).toDoubleVector();
	}
	//beta is the first INDArray and theta is the second INDArray
	private Tuple<INDArray,INDArray> scaleFromVecor(double[] vector) {
//		float[] floatArray = new float[vector.length];
//		for (int i = 0 ; i < vector.length; i++)
//		{
//		    floatArray[i] = (float) vector[i];
//		}
		Nd4j.setDefaultDataTypes(DataType.FLOAT, DataType.FLOAT);
		INDArray rawarray=Nd4j.create(vector);
		INDArray beta=Nd4j.create(rawarray.reshape(this.N*2,this.T).get(new INDArrayIndex[] {NDArrayIndex.interval(0,this.N),NDArrayIndex.all()}).toFloatMatrix());
		INDArray theta=Nd4j.create(rawarray.reshape(this.N*2,this.T).get(new INDArrayIndex[] {NDArrayIndex.interval(this.N,this.N*2),NDArrayIndex.all()}).toFloatMatrix());
		return new Tuple<>(beta,theta);
	}
}






package kriging;

import java.awt.image.DataBuffer;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.lang.reflect.InvocationTargetException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Vector;
import java.util.concurrent.ConcurrentHashMap;
import java.util.stream.IntStream;

import org.apache.commons.math3.linear.CholeskyDecomposition;
import org.apache.commons.math3.linear.EigenDecomposition;
import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.SingularValueDecomposition;
import org.matsim.api.core.v01.Id;
import org.matsim.api.core.v01.network.Link;
import org.matsim.api.core.v01.network.Network;
import org.matsim.api.core.v01.population.Leg;
import org.matsim.api.core.v01.population.Plan;
import org.matsim.api.core.v01.population.PlanElement;
import org.matsim.api.core.v01.population.Population;
import org.matsim.core.network.NetworkUtils;
import org.matsim.core.utils.collections.Tuple;
import org.nd4j.linalg.api.blas.Lapack;
import org.nd4j.linalg.api.blas.impl.BaseLapack;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.buffer.util.DataTypeUtil;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.checkutil.CheckUtil;
import org.nd4j.linalg.cpu.nativecpu.blas.CpuLapack;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.INDArrayIndex;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.linalg.ops.transforms.Transforms;

import de.xypron.jcobyla.Cobyla;
import de.xypron.jcobyla.Calcfc;
import de.xypron.jcobyla.CobylaExitStatus;
import linktolinkBPR.LinkToLink;
import linktolinkBPR.LinkToLinks;
import linktolinkBPR.SignalFlowReductionGenerator;
import training.DataIO;
import training.Evaluator;
import training.FunctionSPSA;
import training.MatlabObj;
import training.MatlabOptimizer;
import training.MatlabResult;
import training.TrainingController;

/**
 * 
 * A kriging framework
 * @author Ashraf
 *
 */
public class KrigingInterpolator{
	private final Map<Integer,Data> trainingDataSet;
	private Variogram variogram;
	private INDArray beta;
	private BaseFunction baseFunction;
	private final int N;
	private final int T;
	private VarianceInfoHolder info;
	private INDArray Cn;
	private INDArray Ct;
	private int numDone=0;
	private double trainingTime=0;
	private double averagePredictionTime=0;
	
	public KrigingInterpolator(Map<Integer, Data> trainingData, LinkToLinks l2ls,BaseFunction bf,Map<String,List<Integer>>n_tSpecificTrainingIndices) {
		this.trainingDataSet=trainingData;
		this.baseFunction=bf;
		this.variogram=new Variogram(trainingData, l2ls,n_tSpecificTrainingIndices);
		this.N=Math.toIntExact(trainingData.get(0).getX().size(0));
		this.T=Math.toIntExact(trainingData.get(0).getX().size(1));
		this.beta=Nd4j.zeros(N,T).addi(1);
		this.info=this.preProcessData();
		this.Cn=Nd4j.ones(N,T);
		this.Ct=Nd4j.ones(N,T);
	}
	
	//TODO: the reading and writing are very messed up right now. Will fix soon. [Ashraf, June 2019].
	//Constructor for creating in the reader
	public KrigingInterpolator(Variogram v, INDArray beta, BaseFunction bf,INDArray Cn,INDArray Ct) {
		this.trainingDataSet=v.getTrainingDataSet();
		this.variogram=v;
		this.beta=beta;
		this.baseFunction=bf;
		this.N=Math.toIntExact(trainingDataSet.get(0).getX().size(0));
		this.T=Math.toIntExact(trainingDataSet.get(0).getX().size(1));
		this.Cn=Cn;
		this.Ct=Ct;
		this.info=this.preProcessData();
	}
	
	public Map<Integer, Data> getTrainingDataSet() {
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
		return this.getY(X, this.beta, this.variogram.gettheta(), this.variogram.getNugget(),info);
	}
	
	public double getY(INDArray X,int n, int t) {
		return this.getY(X, this.beta, this.variogram.gettheta(),this.variogram.getNugget(), info,n,t);
	}
	
	
	public VarianceInfoHolder preProcessData(INDArray beta,INDArray theta, INDArray nugget) {
		long startTime=System.currentTimeMillis();
		//This is the Z-MB
		INDArray Z_MB=Nd4j.create(this.N,this.T,this.trainingDataSet.size());
		Map<String,INDArray> varianceMatrixAll=this.variogram.calculateVarianceMatrixAll(theta, nugget);
		Map<String,INDArray> varianceMatrixInverseAll=new ConcurrentHashMap<>();
		Map<String,double[]> singularValuesAll=new ConcurrentHashMap<>();
		Map<String,Double> varCondNum=new ConcurrentHashMap<>();
		Map<String,Double> logDet=new ConcurrentHashMap<>();
		varianceMatrixAll.entrySet().parallelStream().forEach((n_t_K)->{
			if(this.variogram.getSigmaMatrix().getDouble(Integer.parseInt(n_t_K.getKey().split("_")[0]), Integer.parseInt(n_t_K.getKey().split("_")[1]))==0){
				return;
			}
			//SingularValueDecomposition svd=null;
			CholeskyDecomposition cd=null;
			if(!n_t_K.getValue().isMatrix()) {
				System.out.println("Debug!!!");
				INDArray inv=Nd4j.create(n_t_K.getValue().shape());
				inv.put(0, 0,1/n_t_K.getValue().getDouble(0,0));
				varianceMatrixInverseAll.put(n_t_K.getKey(),inv);
				logDet.put(n_t_K.getKey(), Math.log(n_t_K.getValue().getDouble(0,0)));
				
			}else {
			cd= new CholeskyDecomposition(MatrixUtils.createRealMatrix(n_t_K.getValue().toDoubleMatrix()));

			RealMatrix inv=cd.getSolver().getInverse();
			varianceMatrixInverseAll.put(n_t_K.getKey(),Nd4j.create(inv.getData())) ;
			double logdet=0;
			for(int i=0;i<cd.getL().getRowDimension();i++) {
				logdet+=2*Math.log(cd.getL().getData()[i][i]);
			}
			logDet.put(n_t_K.getKey(),logdet);
			}
			
		});
		for(Entry<Integer,Data>dataPoint:this.trainingDataSet.entrySet()) {
			Z_MB.put(new INDArrayIndex[] {NDArrayIndex.all(),NDArrayIndex.all(),NDArrayIndex.point(dataPoint.getKey())},dataPoint.getValue().getY().sub(this.baseFunction.getY(dataPoint.getValue().getX()).mul(beta)).mul(this.variogram.getTtScale()));// the Y scale is directly applied on Z-MB
		}
		System.out.println("Total Time for info preperation (Inversing and SVD) = "+Long.toString(System.currentTimeMillis()-startTime));
		VarianceInfoHolder info=new VarianceInfoHolder(Z_MB,varianceMatrixAll,varianceMatrixInverseAll,singularValuesAll,varCondNum);
		info.setLogDeterminant(logDet);
		return info;
	}
	
	public VarianceInfoHolder preProcessNtSpecificData(int n, int t,double beta,double theta,double nugget, VarianceInfoHolder info) {
		if(this.variogram.getSigmaMatrix().getDouble(n,t)==0) {
			return info;
		}
		//This is the Z-MB
		INDArray Z_MB=info.getZ_MB();
		INDArray varianceMatrix=this.variogram.calcVarianceMatrix(n, t, theta, nugget);
		String key=Integer.toString(n)+"_"+Integer.toString(t);
		info.getVarianceMatrixAll().put(key, varianceMatrix);
		if(info.getVarianceMatrixAll().get(key)==null) {
			System.out.println();
		}
		CholeskyDecomposition cd= new CholeskyDecomposition(MatrixUtils.createRealMatrix(info.getVarianceMatrixAll().get(key).toDoubleMatrix()));
		RealMatrix inv=cd.getSolver().getInverse();
		
		double logdet=0;
		for(int i=0;i<cd.getL().getRowDimension();i++) {
			logdet+=2*Math.log(cd.getL().getData()[i][i]);
		}
		info.getVarianceMatrixInverseAll().put(key,Nd4j.create(inv.getData())) ;
		info.getLogDeterminant().put(key, logdet);
		for(Entry<Integer,Data>dataPoint:this.trainingDataSet.entrySet()) {
			Z_MB.putScalar(new int[] {n,t,dataPoint.getKey()},(dataPoint.getValue().getY().getDouble(n,t)-this.baseFunction.getntSpecificY(dataPoint.getValue().getX(),n,t)*beta)*this.variogram.getTtScale().getDouble(n,t));// the Y scale is directly applied on Z-MB
		}
		return info;
	}
	
	public VarianceInfoHolder preProcessNtSpecificDataImplicit(int n, int t,double theta,double nugget, VarianceInfoHolder info) {
		if(Double.isNaN(theta)) {
			throw new IllegalArgumentException("theta is null");
		}
//		if(this.variogram.getSigmaMatrix().getDouble(n,t)==0) {
//			return info;
//		}
		//This is the Z-MB
		INDArray Z_MB=info.getZ_MB();
		INDArray varianceMatrix=this.variogram.calcVarianceMatrixWithoutSigma(n, t, theta, nugget);
		String key=Integer.toString(n)+"_"+Integer.toString(t);
		info.getVarianceMatrixAll().put(key, varianceMatrix);
		if(info.getVarianceMatrixAll().get(key)==null) {
			System.out.println();
		}
		CholeskyDecomposition cd=null;
		try {
		cd= new CholeskyDecomposition(MatrixUtils.createRealMatrix(info.getVarianceMatrixAll().get(key).toDoubleMatrix()));
		}catch(Exception e) {
			DataIO.writeINDArray(info.getVarianceMatrixAll().get(key), "trailVar.csv");
		}
		
		RealMatrix inv=cd.getSolver().getInverse();
		
		double logdet=0;
		for(int i=0;i<cd.getL().getRowDimension();i++) {
			logdet+=2*Math.log(cd.getL().getData()[i][i]);
		}
		info.getVarianceMatrixInverseAll().put(key,Nd4j.create(inv.getData())) ;
		info.getLogDeterminant().put(key, logdet);
		INDArray M=Nd4j.create(this.variogram.getNtSpecificTrainingSet().get(key).size(),1);
		INDArray Z=Nd4j.create(this.variogram.getNtSpecificTrainingSet().get(key).size(),1);
		for(Entry<Integer,Data>dataPoint:this.variogram.getNtSpecificTrainingSet().get(key).entrySet()) {
			Z.putScalar(dataPoint.getKey(), dataPoint.getValue().getY().getDouble(n,t));
			M.putScalar(dataPoint.getKey(), this.baseFunction.getntSpecificY(dataPoint.getValue().getX(),n,t));
			//Z_MB.putScalar(new int[] {n,t,dataPoint.getKey()},(dataPoint.getValue().getY().getDouble(n,t)-this.baseFunction.getntSpecificY(dataPoint.getValue().getX(),n,t)*beta)*this.variogram.getTtScale().getDouble(n,t));// the Y scale is directly applied on Z-MB
		}
		INDArray invv=info.getVarianceMatrixInverseAll().get(key);
		double beta=M.transpose().mmul(invv).mmul(M).getDouble(0);
		if(beta<0) {
			throw new IllegalArgumentException("beta cannot be less than zero");
		}
		beta=M.transpose().mmul(invv).mmul(Z).getDouble(0)/beta;
		
		if(beta<0) {
			throw new IllegalArgumentException("beta cannot be less than zero");
		}
		
		INDArray z_mb=Z.sub(M.mul(beta));
		INDArray result=z_mb.transpose().mmul(invv);
		result=result.mmul(z_mb);
		double sigma=1./Z.size(0)*result.getDouble(0);
		if(sigma<=0  || sigma>50) {
		//	System.out.println();
		}
		this.beta.put(n,t,beta);
		this.variogram.getSigmaMatrix().put(n, t,sigma);
		int j=0;
		for(int i:this.variogram.getNtSpecificOriginalIndices().get(key)) {
			Z_MB.putScalar(new int[] {n,t,i},Z.getDouble(j)-M.getDouble(j)*beta);
			j++;
		}
		
		if(Double.isNaN(theta)) {
			throw new IllegalArgumentException("theta is null");
		}
		return info;
	}
	
	public INDArray getY(INDArray X,INDArray beta,INDArray theta,INDArray nugget, VarianceInfoHolder info) {
		INDArray Y=Nd4j.create(X.size(0),X.size(1));
		//This is m(x_0);
		INDArray Y_b=this.baseFunction.getY(X);
		INDArray Z_MB=info.getZ_MB();

		Map<String,INDArray> varianceVectorAll=this.variogram.calculateVarianceVectorAll(X, theta, nugget);
		List<String>n_tlist=new ArrayList<>();
		for(int n=0;n<N;n++) {
			for(int t=0;t<T;t++) {
				n_tlist.add(Integer.toString(n)+"_"+Integer.toString(t));
			}
		}
		n_tlist.parallelStream().forEach((e)->{
			//for(String e:n_tlist) {
			int n=Integer.parseInt(e.split("_")[0]);
			int t=Integer.parseInt(e.split("_")[1]);
			if(this.variogram.getSigmaMatrix().getDouble(n,t)==0) {
				Y.putScalar(n,t,Y_b.getDouble(n,t));
			}else {
				String key=Integer.toString(n)+"_"+Integer.toString(t);
				INDArray z_mb=Nd4j.create(this.variogram.getNtSpecificTrainingSet().get(key).size(),1);
				INDArray KInverse=info.getVarianceMatrixInverseAll().get(key);
				for(int i=0;i<z_mb.size(0);i++) {
					z_mb.putScalar(i, 0,Z_MB.getDouble(n,t,this.variogram.getNtSpecificOriginalIndices().get(key).get(i)));
				}
				
				INDArray weights=varianceVectorAll.get(key).transpose().mmul(KInverse);
				//weights=this.postProcessWeight(weights, varianceVectorAll.get(key));
				
				double y=Y_b.getDouble(n,t)*beta.getDouble(n,t)+
						weights.mmul(z_mb).getDouble(0,0)/this.variogram.getTtScale().getDouble(n,t);//Fix the Z_MB part!!!
				if(y<0) {
					System.out.println("y is negative Stop!!! debug!!!");
				}
				Y.putScalar(n,t,y);
			}

		});
	//}
		return Y;
	}
	
	
	public double getY(INDArray X,INDArray beta,INDArray theta,INDArray nugget, VarianceInfoHolder info, int n, int t) {
		
		//This is m(x_0);
		INDArray Y_b=this.baseFunction.getY(X);
		INDArray varianceVector=this.variogram.calcVarianceVector(n, t, X, theta, nugget);
		INDArray Z_MB=info.getZ_MB();
		if(this.variogram.getSigmaMatrix().getDouble(n,t)==0) {
			return Y_b.getDouble(n,t);
		}else {
			String key=Integer.toString(n)+"_"+Integer.toString(t);
			INDArray z_mb=Nd4j.create(this.variogram.getNtSpecificTrainingSet().get(key).size(),1);
			INDArray KInverse=info.getVarianceMatrixInverseAll().get(key);
			for(int i=0;i<z_mb.size(0);i++) {
				z_mb.putScalar(i, 0,Z_MB.getDouble(n,t,this.variogram.getNtSpecificOriginalIndices().get(key).get(i)));
			}
			INDArray weights=varianceVector.transpose().mmul(KInverse);
			//weights=this.postProcessWeight(weights, varianceVector);
			double y=Y_b.getDouble(n,t)*beta.getDouble(n,t)+
					weights.mmul(z_mb).getDouble(0,0)/this.variogram.getTtScale().getDouble(n,t);//Fix the Z_MB part!!!
			return y;
		}
		
	}
	
	public Tuple<INDArray,INDArray> getXYIterative(Population population){
		return this.getXYIterative(this.beta,this.variogram.gettheta(),this.variogram.getNugget(), this.info,population);
	}
	
	public Tuple<INDArray,INDArray> getXYIterative(INDArray beta,INDArray theta, INDArray nugget, VarianceInfoHolder info,Population population) {
		INDArray XX=KrigingInterpolator.generateXFromPop(population, this.variogram.getL2ls());
		INDArray Y=Nd4j.create(XX.size(0),XX.size(1));
		INDArray X=Nd4j.create(XX.toDoubleMatrix());
		INDArray Yold=Nd4j.create(Y.shape());
		for(int iter=0;iter<100;iter++) {
			Yold.put(new INDArrayIndex[] {NDArrayIndex.all(),NDArrayIndex.all()}, Nd4j.create(Y.toFloatMatrix()));
			//This is m(x_0);
			INDArray Y_b=this.baseFunction.getY(X);
			INDArray Z_MB=info.getZ_MB();

			Map<String,INDArray> varianceVectorAll=this.variogram.calculateVarianceVectorAll(X, theta, nugget);
			IntStream.rangeClosed(0,Math.toIntExact(X.size(0))-1).parallel().forEach((n)->
			{
				IntStream.rangeClosed(0,Math.toIntExact(X.size(1))-1).parallel().forEach((t)->{
					if(this.variogram.getSigmaMatrix().getDouble(n,t)==0) {
						Y.putScalar(n,t,Y_b.getDouble(n,t));
					}else {
						String key=Integer.toString(n)+"_"+Integer.toString(t);
						INDArray z_mb=Nd4j.create(this.variogram.getNtSpecificTrainingSet().get(key).size(),1);
						INDArray KInverse=info.getVarianceMatrixInverseAll().get(key);
						for(int i=0;i<z_mb.size(0);i++) {
							z_mb.putScalar(i, 0,Z_MB.getDouble(n,t,this.variogram.getNtSpecificOriginalIndices().get(key).get(i)));
						}
						double y=Y_b.getDouble(n,t)*beta.getDouble(n,t)+
								varianceVectorAll.get(key).transpose().mmul(KInverse).mmul(z_mb).getDouble(0,0)/this.variogram.getTtScale().getDouble(n,t);//Fix the Z_MB part!!!
						Y.putScalar(n,t,y);
					}
				});
			});
			INDArray Xnew=updateX(Y, population, this.variogram.getL2ls());
			X.addi(Xnew.sub(X).mul(1./(iter+1)));
			//X.put(new INDArrayIndex[] {NDArrayIndex.all(),NDArrayIndex.all()},updateX(Y, population, this.variogram.getL2ls()));
			if(Transforms.abs(X.sub(Xnew)).lt(1).all()) {
				break;
			}
		}
		return new Tuple<>(X,Y);
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
	public double calcCombinedLogLikelihood(INDArray theta, INDArray beta, INDArray nugget) {
		VarianceInfoHolder info=this.preProcessData(beta, theta, nugget);
		INDArray logLiklihood=Nd4j.create(this.N,this.T);
		for(int n=0;n<this.N;n++) {
			for(int t=0;t<this.T;t++) {
				if(this.variogram.getSigmaMatrix().getDouble(n,t)==0) {
					continue;
				}
				String key=Integer.toString(n)+"_"+Integer.toString(t);
				Nd4j.setDefaultDataTypes(DataType.DOUBLE, DataType.DOUBLE);
				INDArray Z_MB=Nd4j.create(this.variogram.getNtSpecificTrainingSet().get(key).size(),1);
				for(int j=0;j<Z_MB.size(0);j++) {
					Z_MB.putScalar(j,0,info.getZ_MB().getDouble(n,t,this.variogram.getNtSpecificOriginalIndices().get(key).get(j)));
				}
				double Logdet_k=info.getLogDeterminant().get(key);

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
	public double calcCombinedLogLikelihood(VarianceInfoHolder info) {
		INDArray logLiklihood=Nd4j.create(this.N,this.T);
		for(int n=0;n<this.N;n++) {
			for(int t=0;t<this.T;t++) {
				if(this.variogram.getSigmaMatrix().getDouble(n,t)==0) {
					continue;
				}
				String key=Integer.toString(n)+"_"+Integer.toString(t);
				Nd4j.setDefaultDataTypes(DataType.DOUBLE, DataType.DOUBLE);
				INDArray Z_MB=Nd4j.create(this.variogram.getNtSpecificTrainingSet().get(key).size(),1);
				for(int j=0;j<Z_MB.size(0);j++) {
					Z_MB.putScalar(j,0,info.getZ_MB().getDouble(n,t,this.variogram.getNtSpecificOriginalIndices().get(key).get(j)));
				}
				double Logdet_k=info.getLogDeterminant().get(key);

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
	public double calcNtSpecificLogLikelihood(int n, int t,double theta, double beta,double nugget, VarianceInfoHolder info) {
		try {
			//theta=theta*100;
			this.preProcessNtSpecificData(n,t,beta, theta, nugget, info);
		}catch(Exception e) {
			System.out.println(n);
			System.out.println(t);
			INDArray weight=CheckUtil.convertFromApacheMatrix(this.variogram.getL2ls().getWeightMatrix(n, t),DataType.DOUBLE);
			INDArray variance=info.getVarianceMatrixAll().get(Integer.toString(n)+"_"+Integer.toString(t));
			this.writeINDArray(variance, "variancefor"+n+"_"+t+".csv");
			this.writeINDArray(weight, "weightfor"+n+"_"+t+".csv");
			this.writeINDArray(this.variogram.getDistances().get(Integer.toString(n)+"_"+Integer.toString(t)), "distancefor"+n+"_"+t+".csv");
			System.out.println(theta);
			System.out.println(this.variogram.getSigmaMatrix().getDouble(n,t));
			System.out.println(this.variogram.getNugget().getDouble(n,t));
		}
		double logLiklihood=0;
		String key=Integer.toString(n)+"_"+Integer.toString(t);
		INDArray Z_MB=Nd4j.create(this.variogram.getNtSpecificTrainingSet().get(key).size(),1);
		for(int j=0;j<Z_MB.size(0);j++) {
			Z_MB.putScalar(j,0,info.getZ_MB().getDouble(n,t,this.variogram.getNtSpecificOriginalIndices().get(key).get(j)));
		}
		double Logdet_k=info.getLogDeterminant().get(key);
		
		INDArray inverseMatrix=info.getVarianceMatrixInverseAll().get(key);

		INDArray secondLLTerm=Z_MB.transpose().mmul(inverseMatrix).mmul(Z_MB);
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
		
		return logLiklihood;
	}
	
	public double calcNtSpecificLogLikelihoodImplicit(int n, int t,double theta, double nugget, VarianceInfoHolder info) {
		try {
			//theta=theta*100;
			//double sigma=this.variogram.getSigmaMatrix().getDouble(n,t);
			this.preProcessNtSpecificDataImplicit(n,t, theta, nugget, info);
		}catch(Exception e) {
			INDArray varianceMatrix=this.variogram.calcVarianceMatrixWithoutSigma(n, t, theta, nugget);
			EigenDecomposition decomp=new EigenDecomposition(MatrixUtils.createRealMatrix(varianceMatrix.toDoubleMatrix()));
			double[] eigenValues=decomp.getRealEigenvalues();
			double minEigen=0;
			for(double d:eigenValues) {
				if(d<minEigen)minEigen=d;
			}
			if(minEigen<=0) {
				this.variogram.getNugget().putScalar(n, t,-1*minEigen+.0001);
				this.preProcessNtSpecificDataImplicit(n,t, theta, this.variogram.getNugget().getDouble(n,t), info);
			}else {
				throw new IllegalArgumentException("Matrix is already positive definite!!!");
			}
			
			
		}
		double logLiklihood=0;
		String key=Integer.toString(n)+"_"+Integer.toString(t);
//		INDArray Z_MB=Nd4j.create(this.variogram.getNtSpecificTrainingSet().get(key).size(),1);
//		for(int j=0;j<Z_MB.size(0);j++) {
//			Z_MB.putScalar(j,0,info.getZ_MB().getDouble(n,t,this.variogram.getNtSpecificOriginalIndices().get(key).get(j)));
//		}
		double Logdet_k=info.getLogDeterminant().get(key);
		int I=this.variogram.getNtSpecificTrainingSet().get(key).size();
		
		//INDArray inverseMatrix=info.getVarianceMatrixInverseAll().get(key);

//		INDArray secondLLTerm=Z_MB.transpose().mmul(inverseMatrix).mmul(Z_MB);
//		double d=-1*Z_MB.size(0)/2.0*Math.log(2*Math.PI)-
//				.5*Logdet_k
//				-.5*secondLLTerm.getDouble(0,0);
		double sigma=this.variogram.getSigmaMatrix().getDouble(n,t);
		
		double liklihood=-1*I/2.*Math.log(2*Math.PI*sigma)-0.5*Logdet_k-I/2;
		
		if(liklihood==Double.NaN) {
			System.out.println("case NAN");

		}else if(liklihood==Double.NEGATIVE_INFINITY){
			System.out.println("Negative Infinity");
		}else if (liklihood==Double.POSITIVE_INFINITY) {
			System.out.println("");
		}
		logLiklihood=liklihood;
		return logLiklihood;
	}
	
	public double calcCombinedLogLikelihood() {
		return this.calcCombinedLogLikelihood(info);
	}
	
	public static void main(String[] args) throws NoSuchMethodException, SecurityException, ClassNotFoundException, IllegalAccessException, IllegalArgumentException, InvocationTargetException {
		String modelFolderName="ModelMeanDeep";
		
		DataTypeUtil.setDTypeForContext(DataType.DOUBLE);
		Nd4j.setDefaultDataTypes(DataType.DOUBLE, DataType.DOUBLE);
		Map<Integer,Data> trainingData=DataIO.readDataSet("Network/ND/DataSetNDTrain.txt","Network/ND/KeySetNDTrain.csv");
		
//		INDArray fullArray=Nd4j.create(trainingData.size(),trainingData.get(0).getFirst().length());
//		for(int i=0;i<trainingData.size();i++) {
//			INDArray rowArray=trainingData.get(i).getFirst().reshape('C',new long[] {1,trainingData.get(i).getFirst().length()});
//			INDArray orginalArray=trainingData.get(i).getFirst();
//			fullArray.put(new INDArrayIndex[] {NDArrayIndex.point(i),NDArrayIndex.all()},rowArray);
//			System.out.println(trainingData.get(i).getFirst());
//		}
//		KrigingInterpolator.writeINDArray(fullArray,"FullData.csv");
		
		
		Network network=NetworkUtils.readNetwork("Network/ND/ndNetwork.xml");
		//Network network=NetworkUtils.readNetwork("Network/SiouxFalls/network.xml");
		SignalFlowReductionGenerator sg = null;
		//config.network().setInputFile("Network/SiouxFalls/network.xml");
		Map<Integer,Tuple<Double,Double>> timeBean=new HashMap<>();
		for(int i=15;i<24;i++) {
			timeBean.put(i,new Tuple<Double,Double>(i*3600.,i*3600.+3600));
		}
		LinkToLinks l2ls=new LinkToLinks(network,timeBean,3,3,sg);
		TrainingController tc=new TrainingController(l2ls, trainingData);
		KrigingInterpolator kriging=new KrigingInterpolator(trainingData, l2ls, new MeanBaseFunction(trainingData),tc.createN_TSpecificTrainingSet(trainingData.size()));
		System.out.println("Model created");
		//System.out.println(kriging.calcCombinedLogLikelihood());
		//System.out.println("Finished!!!");
		System.out.println(kriging.calcCombinedLogLikelihood());
		kriging.trainKriging();
		System.out.println(kriging.calcCombinedLogLikelihood());
		new KrigingModelWriter(kriging).writeModel("Network/ND/"+modelFolderName+"/");
		KrigingInterpolator krigingnew=new KrigingModelReader().readModel("Network/ND/"+modelFolderName+"/modelDetails.xml");
		System.out.println(krigingnew.calcCombinedLogLikelihood());
		Map<Integer,Data> testingData=DataIO.readDataSet("Network/ND/DataSetNDTest.txt","Network/ND/KeySetNDTest.csv");
		kriging=krigingnew;
		INDArray averageError=Nd4j.create(kriging.N,kriging.T);
		for(Data testData:testingData.values()) {
			INDArray Yreal=testData.getY();
			INDArray y=kriging.getY(testData.getX());
			INDArray errorArray=Yreal.sub(y).div(Yreal).mul(100);
			errorArray=Transforms.abs(errorArray);
//			for(int i=0;i<errorArray.size(0);i++) {
//				for(int j=0;j<errorArray.size(1);j++) {
//					errorArray.putScalar(i, j,Math.abs(errorArray.getDouble(i,j)));
//				}
//			}
			KrigingInterpolator.writeINDArray(errorArray, "Network/ND/"+modelFolderName+"/"+testData.getKey()+".csv");
			averageError.addi(errorArray);
		}
		averageError.divi(testingData.size());
		Nd4j.writeTxt(averageError, "Network/ND/"+modelFolderName+"/averagePredictionError.txt");
		System.out.println("Model Read Succesful!!!");
		System.out.println(averageError.maxNumber().doubleValue());
	}
	
	
	
	public void trainKriging() {
		long startTime=System.currentTimeMillis();

		List<String> n_tlist=new ArrayList<>();
		for(int n=0;n<N;n++) {
			for(int t=0;t<T;t++) {
				n_tlist.add(Integer.toString(n)+"_"+Integer.toString(t));
			}
		}
		this.numDone=0;
		n_tlist.parallelStream().forEach((key)->{
			//for(String key:n_tlist) {
			int n=Integer.parseInt(key.split("_")[0]);
			int t=Integer.parseInt(key.split("_")[1]);
			if(this.variogram.getSigmaMatrix().getDouble(n,t)==0) {
				//continue;
				return;
			}
			double initialBeta=this.beta.getDouble(n,t);
			if(this.variogram.getDistances().get(key).maxNumber().doubleValue()==0) {
				KrigingInterpolator.writeINDArray(this.variogram.getDistances().get(key),key+"distance.csv");
			}
			double initialTheta=1/this.variogram.getDistances().get(key).maxNumber().doubleValue()*10;
			//double initialTheta=.1;
			Calcfc calcfc = new Calcfc() {
					int it=0;

					@Override
					public double compute(int N, int m, double[] x, double[] con) {
  						double theta=initialTheta+initialTheta*x[0]/100;
						double beta=initialBeta+initialBeta*x[1]/100;
						//double nugget=initialNugget+initialNugget*x[2]/100;
						double nugget=KrigingInterpolator.this.getVariogram().getNugget().getDouble(n,t);
						//double beta=1;
						double obj=KrigingInterpolator.this.calcNtSpecificLogLikelihood(n, t, theta, beta,nugget, info);
						if(theta==0) {
							obj=-10000000000000.;
						}
						con[0]=100*(theta-0.00001);
						//con[1]=nugget*100;

						it++;

						return -1*obj;
					}
				};
				
//			MatlabObj obj=new MatlabObj() {
//				int it=0;
//				@Override
//				public double evaluateFunction(double[] x) {
//					double theta=initialTheta+initialTheta*x[0]/100;
//					double beta=initialBeta+initialBeta*x[1]/100;
//					//double beta=1;
//					double obj=KrigingInterpolator.this.calcNtSpecificLogLikelihood(n, t, theta, beta,info);
//					if(theta==0) {
//						obj=-10000000000000.;
//					}
//					it++;
//
//					return -1*obj;
//				}
//
//				@Override
//				public double evaluateConstrain(double[] x) {
//					return 5;
//				}
//
//				@Override
//				public LinkedHashMap<String, Double> ScaleUp(double[] x) {
//					// TODO Auto-generated method stub
//					return null;
//				}
//
//			};

			double[] x = {1,1};
//			double[] xlb = {-99.999,-5};
//			double[] xup = {300,5};
 			CobylaExitStatus result = Cobyla.findMinimum(calcfc, 2, 1, x, .5, .00001, 1, 350);
			//MatlabResult result=new MatlabOptimizer(obj, x, xlb, xup).performOptimization();
//			x[0]=result.getX()[0];
//			x[1]=result.getX()[1];
			
			this.beta.putScalar(n, t,initialBeta+initialBeta*x[1]/100);
			this.variogram.gettheta().putScalar(n,t,initialTheta+initialTheta*x[0]/100);
			//this.variogram.getNugget().putScalar(n,t,initialTheta+initialTheta*x[2]/100);
			this.numDone++;
			//System.out.println("current total liklihood after "+key+" = "+this.calcCombinedLogLikelihood());
			System.out.println("Finished training "+this.numDone+" out of "+n_tlist.size());

		});
		//		}
		this.trainingTime=System.currentTimeMillis()-startTime;
		KrigingModelWriter writer=new KrigingModelWriter(this);

		System.out.println("Total time for training for "+N+" links and "+T+" time setps = "+Double.toString(this.trainingTime));
	}
	
	public void trainKrigingImplicit() {
		long startTime=System.currentTimeMillis();

		List<String> n_tlist=new ArrayList<>();
		for(int n=0;n<N;n++) {
			for(int t=0;t<T;t++) {
				n_tlist.add(Integer.toString(n)+"_"+Integer.toString(t));
			}
		}
		
		this.numDone=0;
		n_tlist.parallelStream().forEach((key)->{
			//for(String key:n_tlist) {
			int n=Integer.parseInt(key.split("_")[0]);
			int t=Integer.parseInt(key.split("_")[1]);
			if(this.variogram.getSigmaMatrix().getDouble(n,t)==0) {
				//continue;
				return;
			}
			//System.out.println("loglikelihood before Training = "+n+"_"+t+" = "+this.calcCombinedLogLikelihood());
//			double initialBeta=this.beta.getDouble(n,t);
			if(this.variogram.getDistances().get(key).maxNumber().doubleValue()==0) {
				KrigingInterpolator.writeINDArray(this.variogram.getDistances().get(key),key+"distance.csv");
			}
			double initialTheta=1/this.variogram.getDistances().get(key).maxNumber().doubleValue()*10;
			//double initialTheta=.1;
			Calcfc calcfc = new Calcfc() {
					int it=0;

					@Override
					public double compute(int N, int m, double[] x, double[] con) {
  						double theta=initialTheta+initialTheta*x[0]/100;
						double nugget=KrigingInterpolator.this.getVariogram().getNugget().getDouble(n,t);
						double obj=0;
						if(theta==0) {
							obj=-10000000000000.;
						}else {
							obj=KrigingInterpolator.this.calcNtSpecificLogLikelihoodImplicit(n, t, theta, nugget, info);
						}
						con[0]=100000*(theta-0.000000001);
						con[1]=1000*(KrigingInterpolator.this.getBeta().getDouble(n,t)-.5);
						it++;

						return -1*obj;
					}
				};
				

			double[] x = {1};
 			CobylaExitStatus result = Cobyla.findMinimum(calcfc, 1, 2, x, .5, .00001, 1, 350);
			this.variogram.gettheta().putScalar(n,t,initialTheta+initialTheta*x[0]/100);
			this.preProcessNtSpecificDataImplicit(n, t, initialTheta+initialTheta*x[0]/100, this.getVariogram().getNugget().getDouble(n,t), info);
			info.getVarianceMatrixInverseAll().get(key).divi(this.variogram.getSigmaMatrix().getDouble(n,t));
			info.getLogDeterminant().put(key, info.getLogDeterminant().get(key)+Math.log(this.variogram.getSigmaMatrix().getDouble(n,t))*this.variogram.getNtSpecificOriginalIndices().get(key).size());
			this.variogram.getNugget().put(n,t,this.variogram.getNugget().getDouble(n,t)*this.variogram.getSigmaMatrix().getDouble(n,t));
			this.numDone++;
			
			//System.out.println(this.calcNtSpecificLogLikelihood(n, t, initialTheta+initialTheta*x[0]/100, beta.getDouble(n,t), this.variogram.getNugget().getDouble(n,t), info));
			//System.out.println(this.calcCombinedLogLikelihood());
			System.out.println("Finished training "+this.numDone+" out of "+n_tlist.size());

		});
		//		}
		this.trainingTime=System.currentTimeMillis()-startTime;
		KrigingModelWriter writer=new KrigingModelWriter(this);

		System.out.println("Total time for training for "+N+" links and "+T+" time setps = "+Double.toString(this.trainingTime));
	}
	
	public void placeDataImplicitNugget(double theta, double nugget, int n, int t, VarianceInfoHolder info) {
		this.variogram.gettheta().putScalar(n,t,theta);
		INDArray Z_MB=info.getZ_MB();
		INDArray varianceMatrix=this.variogram.calcVarianceMatrixWithoutSigma(n, t, theta, nugget);
		String key=Integer.toString(n)+"_"+Integer.toString(t);
		CholeskyDecomposition cd=null;
		try {
		cd= new CholeskyDecomposition(MatrixUtils.createRealMatrix(varianceMatrix.toDoubleMatrix()));
		}catch(Exception e) {
			DataIO.writeINDArray(info.getVarianceMatrixAll().get(key), "trailVar.csv");
		}
		RealMatrix inv=cd.getSolver().getInverse();
		INDArray invv=Nd4j.create(inv.getData());
		INDArray M=Nd4j.create(this.variogram.getNtSpecificTrainingSet().get(key).size(),1);
		INDArray Z=Nd4j.create(this.variogram.getNtSpecificTrainingSet().get(key).size(),1);
		for(Entry<Integer,Data>dataPoint:this.variogram.getNtSpecificTrainingSet().get(key).entrySet()) {
			Z.putScalar(dataPoint.getKey(), dataPoint.getValue().getY().getDouble(n,t));
			M.putScalar(dataPoint.getKey(), this.baseFunction.getntSpecificY(dataPoint.getValue().getX(),n,t));
			//Z_MB.putScalar(new int[] {n,t,dataPoint.getKey()},(dataPoint.getValue().getY().getDouble(n,t)-this.baseFunction.getntSpecificY(dataPoint.getValue().getX(),n,t)*beta)*this.variogram.getTtScale().getDouble(n,t));// the Y scale is directly applied on Z-MB
		}
		double beta=M.transpose().mmul(invv).mmul(M).getDouble(0);
		if(beta<0) {
			throw new IllegalArgumentException("beta cannot be less than zero");
		}
		beta=M.transpose().mmul(invv).mmul(Z).getDouble(0)/beta;
		
		if(beta<0) {
			throw new IllegalArgumentException("beta cannot be less than zero");
		}
		
		INDArray z_mb=Z.sub(M.mul(beta));
		INDArray result=z_mb.transpose().mmul(invv).mmul(z_mb);
		double sigma=1./Z.size(0)*result.getDouble(0);
		System.out.println("sigma = "+sigma);
		if(sigma<=0) {
			System.out.println();
		}
		this.beta.put(n,t,beta);
		this.variogram.getSigmaMatrix().put(n,t,sigma);
		int j=0;
		for(int i:this.variogram.getNtSpecificOriginalIndices().get(key)) {
			Z_MB.putScalar(new int[] {n,t,i},Z.getDouble(j)-M.getDouble(j)*beta);
			j++;
		}
		varianceMatrix=varianceMatrix.muli(sigma);
//		try {
//			cd= new CholeskyDecomposition(MatrixUtils.createRealMatrix(varianceMatrix.toDoubleMatrix()));
//		}catch(Exception e) {
//			DataIO.writeINDArray(info.getVarianceMatrixAll().get(key), "trailVar.csv");
//		}
//		inv=cd.getSolver().getInverse();
		invv=invv.muli(1/sigma);
		double logdet=0;
		for(int i=0;i<cd.getL().getRowDimension();i++) {
			logdet+=2*Math.log(cd.getL().getData()[i][i]);
		}
		logdet+=cd.getL().getRowDimension()*Math.log(sigma);
		info.getVarianceMatrixAll().put(key, varianceMatrix);
		info.getVarianceMatrixInverseAll().put(key, invv);
		info.getLogDeterminant().put(key, logdet);
		this.variogram.getNugget().put(n,t,nugget*sigma);
	}
	
	public INDArray postProcessWeight(INDArray weights, INDArray varianceVector) {
		int j=0;
		double absNegWeight=0;
		double absNegCov=0;
		for(int i=0;i<weights.length();i++) {
			if(weights.getDouble(i)<0) {
				j++;
				absNegWeight+=Math.abs(weights.getDouble(i));
				absNegCov+=varianceVector.getDouble(i);
			}
		}
		absNegWeight=absNegWeight/j;
		absNegCov=absNegCov/j;
		INDArray newWeights=Nd4j.create(weights.shape());
		double sumNewWeights=0;
		for(int i=0;i<weights.length();i++) {
			double w=weights.getDouble(i);
			if(w<0) {
				w=0;
			}else if(w>0 && varianceVector.getDouble(i)<absNegCov) {
				w=0;
			}
			sumNewWeights+=w;
			newWeights.putScalar(i, w);
		}
		return newWeights.div(sumNewWeights);
	}
	
	public void trainKrigingImplicitWithNugget() {
		long startTime=System.currentTimeMillis();

		List<String> n_tlist=new ArrayList<>();
		for(int n=0;n<N;n++) {
			for(int t=0;t<T;t++) {
				n_tlist.add(Integer.toString(n)+"_"+Integer.toString(t));
			}
		}
		this.numDone=0;
		
		n_tlist.parallelStream().forEach((key)->{
		//	for(String key:n_tlist) {
			int n=Integer.parseInt(key.split("_")[0]);
			int t=Integer.parseInt(key.split("_")[1]);
			//System.out.println("loglikelihood before Training = "+n+"_"+t+" = "+this.calcCombinedLogLikelihood());
			if(this.variogram.getSigmaMatrix().getDouble(n,t)==0) {
				//continue;
				return;
			}
//			double initialBeta=this.beta.getDouble(n,t);
			double initialNugget=.15;
			if(this.variogram.getDistances().get(key).maxNumber().doubleValue()==0) {
				KrigingInterpolator.writeINDArray(this.variogram.getDistances().get(key),key+"distance.csv");
			}
			double initialTheta=1/this.variogram.getDistances().get(key).maxNumber().doubleValue()*10;
			//double initialTheta=.1;
			Calcfc calcfc = new Calcfc() {
					int it=0;
					double initialObj=0;
					
					@Override
					public double compute(int N, int m, double[] x, double[] con) {
  						double theta=initialTheta+initialTheta*x[0]/100;
						double nugget=initialNugget+initialNugget*x[1]/100;
						double obj=0;
						if(theta<=0 || nugget<0) {
							obj=-10000000000000.;
						}else {
							obj=KrigingInterpolator.this.calcNtSpecificLogLikelihoodImplicit(n, t, theta, nugget, info);
						}
						con[0]=1000000*(theta-0.0000001);
						con[1]=100*(nugget-.01);
						con[2]=100*(KrigingInterpolator.this.getBeta().getDouble(n,t)-0.5);
						
						//if(it==0)System.out.println(-1*obj);
						it++;
						return -1*obj;
					}
				};
				

			double[] x = {1,1};
 			CobylaExitStatus result = Cobyla.findMinimum(calcfc, 2, 3, x, .5, .00001, 1, 1500); 	
 			double theta=initialTheta+initialTheta*x[0]/100;
			double nugget=initialNugget+initialNugget*x[1]/100;
	//		this.placeDataImplicitNugget(theta, nugget, n, t, info);
			this.variogram.gettheta().putScalar(n,t,theta);
			this.preProcessNtSpecificDataImplicit(n, t, initialTheta+initialTheta*x[0]/100, initialNugget+initialNugget*x[1]/100, info);
			//System.out.println(this.calcNtSpecificLogLikelihoodImplicit(n, t, theta, nugget, info));
			info.getVarianceMatrixInverseAll().get(key).divi(this.variogram.getSigmaMatrix().getDouble(n,t));
			info.getLogDeterminant().put(key, info.getLogDeterminant().get(key)+Math.log(this.variogram.getSigmaMatrix().getDouble(n,t))*this.variogram.getNtSpecificOriginalIndices().get(key).size());
			this.variogram.getNugget().put(n,t,(initialNugget+initialNugget*x[1]/100)*this.variogram.getSigmaMatrix().getDouble(n,t));
			//System.out.println(this.calcNtSpecificLogLikelihood(n, t, theta, beta.getDouble(n,t), this.variogram.getNugget().getDouble(n,t), info));
			//System.out.println(this.calcCombinedLogLikelihood());
			this.numDone++;
			System.out.println("Finished training "+this.numDone+" out of "+n_tlist.size());

		});
		//		}
		this.trainingTime=System.currentTimeMillis()-startTime;
		KrigingModelWriter writer=new KrigingModelWriter(this);

		System.out.println("Total time for training for "+N+" links and "+T+" time setps = "+Double.toString(this.trainingTime));
	}
	
	public void trainKrigingWithNugget() {
		long startTime=System.currentTimeMillis();

		List<String> n_tlist=new ArrayList<>();
		for(int n=0;n<N;n++) {
			for(int t=0;t<T;t++) {
				n_tlist.add(Integer.toString(n)+"_"+Integer.toString(t));
			}
		}
		this.numDone=0;
		n_tlist.parallelStream().forEach((key)->{
			//for(String key:n_tlist) {
			int n=Integer.parseInt(key.split("_")[0]);
			int t=Integer.parseInt(key.split("_")[1]);
			if(this.variogram.getSigmaMatrix().getDouble(n,t)==0) {
				//continue;
				return;
			}
			double initialBeta=this.beta.getDouble(n,t);
			final Double initialNugget=0.15*this.variogram.getSigmaMatrix().getDouble(n,t);
			
			if(this.variogram.getDistances().get(key).maxNumber().doubleValue()==0) {
				KrigingInterpolator.writeINDArray(this.variogram.getDistances().get(key),key+"distance.csv");
			}
			double initialTheta=1/this.variogram.getDistances().get(key).maxNumber().doubleValue()*100;
			//double initialTheta=.1;
			Calcfc calcfc = new Calcfc() {
					int it=0;

					@Override
					public double compute(int N, int m, double[] x, double[] con) {
  						double theta=initialTheta+initialTheta*x[0]/100;
						double beta=initialBeta+initialBeta*x[1]/100;
						double nugget=initialNugget+initialNugget*x[2]/100;
						//double beta=1;
						double obj=0;
						if(theta<=0) {
							obj=-10000000000000.;
						}else {
							obj=KrigingInterpolator.this.calcNtSpecificLogLikelihood(n, t, theta, beta,nugget, info);
						}
						con[0]=1000000*(theta-0.0000001);
						con[1]=10000*(nugget-.000001);

						it++;

						return -1*obj;
					}
				};
				
//			MatlabObj obj=new MatlabObj() {
//				int it=0;
//				@Override
//				public double evaluateFunction(double[] x) {
//					double theta=initialTheta+initialTheta*x[0]/100;
//					double beta=initialBeta+initialBeta*x[1]/100;
//					//double beta=1;
//					double obj=KrigingInterpolator.this.calcNtSpecificLogLikelihood(n, t, theta, beta,info);
//					if(theta==0) {
//						obj=-10000000000000.;
//					}
//					it++;
//
//					return -1*obj;
//				}
//
//				@Override
//				public double evaluateConstrain(double[] x) {
//					return 5;
//				}
//
//				@Override
//				public LinkedHashMap<String, Double> ScaleUp(double[] x) {
//					// TODO Auto-generated method stub
//					return null;
//				}
//
//			};

			double[] x = {1,1,1};
//			double[] xlb = {-99.999,-5};
//			double[] xup = {300,5};
 			CobylaExitStatus result = Cobyla.findMinimum(calcfc, 3, 2, x, .5, .00001, 1, 350);
			//MatlabResult result=new MatlabOptimizer(obj, x, xlb, xup).performOptimization();
//			x[0]=result.getX()[0];
//			x[1]=result.getX()[1];
			
			this.beta.putScalar(n, t,initialBeta+initialBeta*x[1]/100);
			this.variogram.gettheta().putScalar(n,t,initialTheta+initialTheta*x[0]/100);
			this.variogram.getNugget().putScalar(n,t,initialNugget+initialNugget*x[2]/100);
			this.numDone++;
			//System.out.println("current total liklihood after "+key+" = "+this.calcCombinedLogLikelihood());
			System.out.println("Finished training "+this.numDone+" out of "+n_tlist.size());

		});
		//		}
		this.trainingTime=System.currentTimeMillis()-startTime;
		KrigingModelWriter writer=new KrigingModelWriter(this);

		System.out.println("Total time for training for "+N+" links and "+T+" time setps = "+Double.toString(this.trainingTime));
	}
	
	
	public void trainKrigingWithoutbeta() {
		long startTime=System.currentTimeMillis();

		List<String> n_tlist=new ArrayList<>();
		for(int n=0;n<N;n++) {
			for(int t=0;t<T;t++) {
				n_tlist.add(Integer.toString(n)+"_"+Integer.toString(t));
				}
			}
		this.numDone=0;
		n_tlist.parallelStream().forEach((key)->{
		//for(String key:n_tlist) {
			int n=Integer.parseInt(key.split("_")[0]);
			int t=Integer.parseInt(key.split("_")[1]);
			if(this.variogram.getSigmaMatrix().getDouble(n,t)==0) {
				//continue;
				return;
			}
			//double initialBeta=this.beta.getDouble(n,t);
			if(this.variogram.getDistances().get(key).maxNumber().doubleValue()==0) {
				KrigingInterpolator.writeINDArray(this.variogram.getDistances().get(key),key+"distance.csv");
			}
			double initialTheta=1/this.variogram.getDistances().get(key).maxNumber().doubleValue()*10;
//			final Double initialNugget;
//			if(this.variogram.getNugget().getDouble(n,t)==0) {
//				initialNugget=this.variogram.getNugget().getDouble(n,t);
//			}else {
//				initialNugget=1.;
//			}
				Calcfc calcfc = new Calcfc() {
					int it=0;

					@Override
					public double compute(int N, int m, double[] x, double[] con) {
						double theta=initialTheta+initialTheta*x[0]/100;
						//double nugget=initialNugget+initialNugget*x[1]*100;
						double nugget=KrigingInterpolator.this.getVariogram().getNugget().getDouble(n,t);
						//double beta=initialBeta+initialBeta*x[1]/100;
						//double beta=1;
						double obj=KrigingInterpolator.this.calcNtSpecificLogLikelihood(n, t, theta, beta.getDouble(n,t),nugget, info);
						if(theta==0) {
							obj=10000000000000.;
						}
						con[0]=100*(theta-0.0000001);
						//con[1]=100*nugget;

						it++;

						return -1*obj;
					}
				};
				
				double[] x = {1};
				CobylaExitStatus result = Cobyla.findMinimum(calcfc, 1, 1, x, 2, .01, 1, 500);
				//this.beta.putScalar(n, t,initialBeta+initialBeta*x[1]/100);
				this.variogram.gettheta().putScalar(n,t,initialTheta+initialTheta*x[0]/100);
				//this.variogram.getNugget().putScalar(n,t,initialNugget+initialNugget*x[1]*100);
				this.numDone++;
				//System.out.println("current total liklihood after "+key+" = "+this.calcCombinedLogLikelihood());
				System.out.println("Finished training "+this.numDone+" out of "+n_tlist.size());

		});
//		}
		this.trainingTime=System.currentTimeMillis()-startTime;
		KrigingModelWriter writer=new KrigingModelWriter(this);
			
		System.out.println("Total time for training for "+N+" links and "+T+" time setps = "+Double.toString(this.trainingTime));
	}
	
	public void deepTrainKriging() {
		long startTime=System.currentTimeMillis();
		List<String> n_tlist=new ArrayList<>();
		for(int n=0;n<N;n++) {
			for(int t=0;t<T;t++) {
				n_tlist.add(Integer.toString(n)+"_"+Integer.toString(t));
				}
			}
		this.numDone=0;
		n_tlist.parallelStream().forEach((key)->{
		//for(String key:n_tlist) {
			long startTiment=System.currentTimeMillis();
			int n=Integer.parseInt(key.split("_")[0]);
			int t=Integer.parseInt(key.split("_")[1]);
			if(this.variogram.getSigmaMatrix().getDouble(n,t)==0) {
				//continue;
				return;
			}
			
			double initialBeta=this.beta.getDouble(n,t);
			double initialTheta=1/this.variogram.getDistances().get(key).maxNumber().doubleValue()*10;
//			final Double initialNugget;
//			if(this.variogram.getNugget().getDouble(n,t)==0) {
//				initialNugget=this.variogram.getNugget().getDouble(n,t);
//			}else {
//				initialNugget=1.;
//			}
			double initialCn=1;
			double initialCt=1;
				Calcfc calcfc = new Calcfc() {
					
					int it=0;

					@Override
					public double compute(int N, int m, double[] x, double[] con) {
						double theta=initialTheta+initialTheta*x[0]/100;
						double beta=initialBeta+initialBeta*x[1]/100;
						double cn=initialCn+initialCn*x[2]/100;
						double ct=initialCt+initialCt*x[3]/100;
						//double nugget=initialNugget+initialNugget*x[4]/100;
						double nugget=KrigingInterpolator.this.variogram.getNugget().getDouble(n,t);
						KrigingInterpolator.this.variogram.calcDistanceMatrix(n, t, cn, ct);
						double obj=0;
						if(theta<=0) {
							obj=10000000000000.;
						}else {
							obj=KrigingInterpolator.this.calcNtSpecificLogLikelihood(n, t, theta, beta,nugget,info);
						}
						con[0]=100*(theta-0.0000001);
						con[1]=initialCn+initialCn*x[2]/100;
						con[2]=initialCt+initialCt*x[3]/100;
						//con[3]=nugget*100;
						//con[1]=-1*info.getVarianceMatrixCondNum().get(key)+10000;
						it++;
						if(it==1) {
							System.out.println("initial obj = "+-1*obj);
						}
						return -1*obj;
					}
				};
				
				double[] x = {1,1,1,1};
				CobylaExitStatus result = Cobyla.findMinimum(calcfc, 4, 3, x, 2, .01, 1,400);
				this.beta.putScalar(n, t,initialBeta+initialBeta*x[1]/100);
				this.variogram.gettheta().putScalar(n,t,initialTheta+initialTheta*x[0]/100);
				//this.variogram.getNugget().putScalar(n, t,initialNugget+initialNugget*x[4]/100);
				double cn=initialCn+initialCn*x[2]/100;
				double ct=initialCt+initialCt*x[3]/100;
				this.variogram.calcDistanceMatrix(n,t,cn, ct);
				this.Cn.put(n,t,cn);
				this.Ct.put(n,t,ct);
				this.numDone++;
				//System.out.println("current total liklihood after "+key+" = "+this.calcCombinedLogLikelihood());
				System.out.println("optim time for case "+key+" "+Long.toString(System.currentTimeMillis()-startTiment));
				System.out.println("Finished training "+this.numDone+" out of "+n_tlist.size());
				
		});
		//}
		
		this.trainingTime=(double)(System.currentTimeMillis()-startTime);
		System.out.println("Total time for training for "+N+" links and "+T+" time setps = "+Long.toString(System.currentTimeMillis()-startTime));
	}
	
	
	public void deepTrainKrigingBPR() {
		long startTime=System.currentTimeMillis();
		List<String> n_tlist=new ArrayList<>();
		for(int n=0;n<N;n++) {
			for(int t=0;t<T;t++) {
				n_tlist.add(Integer.toString(n)+"_"+Integer.toString(t));
				}
			}
		BPRBaseFunction base=(BPRBaseFunction)this.baseFunction;
		this.numDone=0;
		n_tlist.parallelStream().forEach((key)->{
		//for(String key:n_tlist) {
			long startTiment=System.currentTimeMillis();
			int n=Integer.parseInt(key.split("_")[0]);
			int t=Integer.parseInt(key.split("_")[1]);
			if(this.variogram.getSigmaMatrix().getDouble(n,t)==0) {
				//continue;
				return;
			}
			
			double initialBeta=this.beta.getDouble(n,t);
			double initialTheta=1/this.variogram.getDistances().get(key).maxNumber().doubleValue()*10;
			double initialBPR1=.15;
			double initialBPR2=4;
//			final Double initialNugget;
//			if(this.variogram.getNugget().getDouble(n,t)==0) {
//				initialNugget=this.variogram.getNugget().getDouble(n,t);
//			}else {
//				initialNugget=1.;
//			}
				Calcfc calcfc = new Calcfc() {
					
					int it=0;

					@Override
					public double compute(int N, int m, double[] x, double[] con) {
						double theta=initialTheta+initialTheta*x[0]/100;
						double beta=initialBeta+initialBeta*x[1]/100;
						double bpr1=initialBPR1+initialBPR1*x[2]/100;
						double bpr2=initialBPR2+initialBPR2*x[3]/20;
						//double nugget=initialNugget+initialNugget*x[4]/100;
						double nugget=KrigingInterpolator.this.getVariogram().getNugget().getDouble(n,t);
						base.setAlphaBetaNTSpecific(bpr1, bpr2, n, t);
						double obj=0;
						if(theta<=0) {
							obj=10000000000000.;
						}else {
							obj=KrigingInterpolator.this.calcNtSpecificLogLikelihood(n, t, theta, beta,nugget, info);
						}
						con[0]=100*(theta-0.0000001);
						con[1]=initialBPR1+initialBPR1*x[2]/100;
						con[2]=initialBPR2+initialBPR2*x[3]/100;
						//con[3]=nugget*100;
						//con[1]=-1*info.getVarianceMatrixCondNum().get(key)+10000;
						it++;
						if(it==1) {
							System.out.println("initial obj = "+-1*obj);
						}
						return -1*obj;
					}
				};
				
				double[] x = {1,1,1,1};
				CobylaExitStatus result = Cobyla.findMinimum(calcfc, 4, 3, x, 2, .01, 1,400);
				this.beta.putScalar(n, t,initialBeta+initialBeta*x[1]/100);
				this.variogram.gettheta().putScalar(n,t,initialTheta+initialTheta*x[0]/100);
				//this.variogram.getNugget().putScalar(n,t, initialNugget+initialNugget*x[4]/100);
				double bpr1=initialBPR1+initialBPR1*x[2]/100;
				double bpr2=initialBPR2+initialBPR2*x[3]/20;
				base.setAlphaBetaNTSpecific(bpr1, bpr2, n, t);
				this.numDone++;
				//System.out.println("current total liklihood after "+key+" = "+this.calcCombinedLogLikelihood());
				System.out.println("optim time for case "+key+" "+Long.toString(System.currentTimeMillis()-startTiment));
				System.out.println("Finished training "+this.numDone+" out of "+n_tlist.size());
				
		});
		//}
		
		this.trainingTime=(double)(System.currentTimeMillis()-startTime);
		System.out.println("Total time for training for "+N+" links and "+T+" time setps = "+Long.toString(System.currentTimeMillis()-startTime));
	}
	
	private VarianceInfoHolder preProcessData() {
		return this.preProcessData(this.beta,this.variogram.gettheta(),this.variogram.getNugget());
	}


	private double[] scaleToVector(INDArray beta, INDArray theta) {
		INDArray linearBeta=beta.reshape(beta.length());
		INDArray linearTheta=theta.reshape(theta.length());
		return Nd4j.concat(0, linearBeta,linearTheta).toDoubleVector();
	}
	//beta is the first INDArray and theta is the second INDArray
	private Tuple<INDArray,INDArray> scaleFromVecor(double[] vector) {

		Nd4j.setDefaultDataTypes(DataType.DOUBLE, DataType.DOUBLE);
		INDArray rawarray=Nd4j.create(vector);
		INDArray beta=Nd4j.create(rawarray.reshape(this.N*2,this.T).get(new INDArrayIndex[] {NDArrayIndex.interval(0,this.N),NDArrayIndex.all()}).toDoubleMatrix());
		INDArray theta=Nd4j.create(rawarray.reshape(this.N*2,this.T).get(new INDArrayIndex[] {NDArrayIndex.interval(this.N,this.N*2),NDArrayIndex.all()}).toDoubleMatrix());
		return new Tuple<>(beta,theta);
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
			}
			fw.flush();
			fw.close();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}

	public INDArray getCn() {
		return Cn;
	}



	public INDArray getCt() {
		return Ct;
	}

	public double getTrainingTime() {
		return trainingTime;
	}

	public double getAveragePredictionTime() {
		return averagePredictionTime;
	}

	public void setTrainingTime(double trainingTime) {
		this.trainingTime = trainingTime;
	}
	
	public static INDArray updateX(INDArray TT,Population population,LinkToLinks l2ls) {
		INDArray X=Nd4j.create(TT.shape());
		Map<String,Double> linkToLinksDemand=new ConcurrentHashMap<>();
		population.getPersons().entrySet().forEach((e)->{
			Plan plan=e.getValue().getSelectedPlan();
			for(PlanElement pl:plan.getPlanElements()) {
				Leg l;
				ArrayList<Id<Link>> links=new ArrayList<>();

				if(pl instanceof Leg) {
					l=(Leg)pl;
					String[] part=l.getRoute().getRouteDescription().split(" ");
					for(String s:part) {
						links.add(Id.createLinkId(s.trim()));
					}
					double time=l.getDepartureTime();
					for(int i=1;i<links.size();i++) {
						Id<LinkToLink> l2lId=Id.create(links.get(i-1)+"_"+links.get(i), LinkToLink.class);
						int n=l2ls.getNumToLinkToLink().inverse().get(l2lId);
						int t=l2ls.getTimeId(time);
						String key=Integer.toString(n)+"_"+Integer.toString(t);
						if(linkToLinksDemand.containsKey(key)) {
							linkToLinksDemand.put(key, linkToLinksDemand.get(key)+1);
						}else {
							linkToLinksDemand.put(key,1.);
						}

						time+=TT.getDouble(n,t);
					}
				}else {
					continue;
				}
			}
		});

		linkToLinksDemand.entrySet().parallelStream().forEach((n_t_d)->{		
			String key=n_t_d.getKey();
			int n=Integer.parseInt(key.split("_")[0]);
			int t=Integer.parseInt(key.split("_")[1]);
			if(linkToLinksDemand.containsKey(key)) {
				if(linkToLinksDemand.get(key)==Double.NaN) {
					System.out.println();
				}

				X.putScalar(n,t,linkToLinksDemand.get(key));
			}else {
				X.putScalar(n,t,0);
			}
		});
		return X;
	}
	

	public static INDArray generateXFromPop(Population population,LinkToLinks l2ls) {
		Nd4j.setDefaultDataTypes(DataType.DOUBLE, DataType.DOUBLE);
		INDArray X=Nd4j.create(l2ls.getLinkToLinks().size(),l2ls.getTimeBean().size());
		Map<String,Double> linkToLinksDemand=new ConcurrentHashMap<>();
		population.getPersons().entrySet().forEach((e)->{
			Plan plan=e.getValue().getSelectedPlan();
			for(PlanElement pl:plan.getPlanElements()) {
				Leg l;
				ArrayList<Id<Link>> links=new ArrayList<>();
				
				if(pl instanceof Leg) {
					l=(Leg)pl;
					String[] part=l.getRoute().getRouteDescription().split(" ");
					for(String s:part) {
						links.add(Id.createLinkId(s.trim()));
					}
					double time=l.getDepartureTime();
					for(int i=1;i<links.size();i++) {
						Id<LinkToLink> l2lId=Id.create(links.get(i-1)+"_"+links.get(i), LinkToLink.class);
						int n=l2ls.getNumToLinkToLink().inverse().get(l2lId);
						int t=l2ls.getTimeId(time);
						String key=Integer.toString(n)+"_"+Integer.toString(t);
						if(linkToLinksDemand.containsKey(key)) {
							linkToLinksDemand.put(key, linkToLinksDemand.get(key)+1);
						}else {
							linkToLinksDemand.put(key,1.);
						}
						
						time+=l2ls.getLinkToLink(l2lId).getFreeFlowTT();
					}
				}else {
					continue;
				}
			}
		});
		
		IntStream.rangeClosed(0,l2ls.getLinkToLinks().size()-1).parallel().forEach((n)->
		{
			IntStream.rangeClosed(0,l2ls.getTimeBean().size()-1).parallel().forEach((t)->{
				String key=Integer.toString(n)+"_"+Integer.toString(t);
				if(linkToLinksDemand.containsKey(key)) {
					if(linkToLinksDemand.get(key)==Double.NaN) {
						System.out.println();
					}
					X.putScalar(n,t,linkToLinksDemand.get(key));
				}else {
					X.putScalar(n,t,0);
				}
			});
		});
		return X;
	}
	
}

class krigingTrainer implements Runnable{
	public static int totalNT=0;
	public static int sigmaZero=0;
	private KrigingInterpolator kriging;
	private List<String> n_tlist;
	private VarianceInfoHolder info;
	public krigingTrainer(KrigingInterpolator kriging, List<String> n_tList,VarianceInfoHolder info) {
		this.kriging=kriging;
		this.n_tlist=n_tList;
		this.info=info;
	}
	@Override
	public void run() {
		for(String key:n_tlist) {
			int n=Integer.parseInt(key.split("_")[0]);
			int t=Integer.parseInt(key.split("_")[1]);
			if(this.kriging.getVariogram().getSigmaMatrix().getDouble(n,t)==0) {
				sigmaZero++;
				continue;
			}
			double initialBeta=this.kriging.getBeta().getDouble(n,t);
			double initialTheta=1/this.kriging.getVariogram().getDistances().get(key).maxNumber().doubleValue()*10;
			final Double initialNugget;
			if(this.kriging.getVariogram().getNugget().getDouble(n,t)==0) {
				initialNugget=this.kriging.getVariogram().getNugget().getDouble(n,t);
			}else {
				initialNugget=1.;
			}
			
			Calcfc calcfc = new Calcfc() {
				int it=0;

				@Override
				public double compute(int N, int m, double[] x, double[] con) {
					double theta=initialTheta+initialTheta*x[0]/100;
					double beta=initialBeta+initialBeta*x[1]/100;
					double nugget=initialNugget+initialNugget*x[2]/100;
					double obj=kriging.calcNtSpecificLogLikelihood(n, t, theta, beta,nugget, info);
					if(theta==0) {
						obj=10000000000000.;
					}
					con[0]=100*(theta-0.0000001);

					con[1]=-1*info.getVarianceMatrixCondNum().get(key)+10000;
					con[2]=nugget*100;
					it++;

					return -1*obj;
				}
			};

			double[] x = {1,1,1};
			System.out.println("Current "+key);
			CobylaExitStatus result = Cobyla.findMinimum(calcfc, 3, 3, x, 0.5, .0001, 2, 800);
			this.kriging.getBeta().putScalar(n, t,initialBeta+initialBeta*x[1]/100);
			this.kriging.getVariogram().gettheta().putScalar(n,t,initialTheta+initialTheta*x[0]/100);
			totalNT++;
			//System.out.println("current total liklihood after "+key+" = "+this.kriging.calcCombinedLogLikelihood());
		}
	
	}
}






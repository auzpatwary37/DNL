package kriging;

import java.awt.image.DataBuffer;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.lang.reflect.InvocationTargetException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Vector;
import java.util.concurrent.ConcurrentHashMap;
import java.util.stream.IntStream;

import org.apache.commons.math3.linear.CholeskyDecomposition;
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
	private Variogram variogram;
	private INDArray beta;
	private BaseFunction baseFunction;
	private final int N;
	private final int T;
	private VarianceInfoHolder info;
	private INDArray Cn;
	private INDArray Ct;
	
	public KrigingInterpolator(Map<Integer,Tuple<INDArray,INDArray>> trainingDataSet, LinkToLinks l2ls,BaseFunction bf) {
		this.trainingDataSet=trainingDataSet;
		this.baseFunction=bf;
		this.variogram=new Variogram(trainingDataSet, l2ls);
		this.N=Math.toIntExact(trainingDataSet.get(0).getFirst().size(0));
		this.T=Math.toIntExact(trainingDataSet.get(0).getFirst().size(1));
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
		this.N=Math.toIntExact(trainingDataSet.get(0).getFirst().size(0));
		this.T=Math.toIntExact(trainingDataSet.get(0).getFirst().size(1));
		this.Cn=Cn;
		this.Ct=Ct;
		this.info=this.preProcessData();
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
		Map<String,Double> varCondNum=new ConcurrentHashMap<>();
		varianceMatrixAll.entrySet().parallelStream().forEach((n_t_K)->{
			//n_t_K.getValue()
			SingularValueDecomposition svd=new SingularValueDecomposition(MatrixUtils.createRealMatrix(n_t_K.getValue().toDoubleMatrix()));
			RealMatrix inv=svd.getSolver().getInverse();
			varCondNum.put(n_t_K.getKey(), svd.getConditionNumber());
			double[] singularValues=svd.getSingularValues();
			varianceMatrixInverseAll.put(n_t_K.getKey(),Nd4j.create(toFloatArray(inv.getData()))) ;
			singularValuesAll.put(n_t_K.getKey(), singularValues);
		});
		for(Entry<Integer,Tuple<INDArray,INDArray>>dataPoint:this.trainingDataSet.entrySet()) {
			Z_MB.put(new INDArrayIndex[] {NDArrayIndex.all(),NDArrayIndex.all(),NDArrayIndex.point(dataPoint.getKey())},dataPoint.getValue().getSecond().sub(this.baseFunction.getY(dataPoint.getValue().getFirst()).mul(beta)).mul(this.variogram.getTtScale()));// the Y scale is directly applied on Z-MB
		}
		System.out.println("Total Time for info preperation (Inversing and SVD) = "+Long.toString(System.currentTimeMillis()-startTime));
		return new VarianceInfoHolder(Z_MB,varianceMatrixAll,varianceMatrixInverseAll,singularValuesAll,varCondNum);
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
		Double varCondNum=svd.getConditionNumber();
//		if(varCondNum.isInfinite()||varCondNum.isNaN()) {
//			System.out.println("debugPoint");
//			INDArray distanceMatrix=this.variogram.getDistances().get(key);
//			KrigingInterpolator.writeINDArray(varianceMatrix, "Network/ND/"+n+"_"+t+"variances.csv");
//			KrigingInterpolator.writeINDArray(distanceMatrix, "Network/ND/"+n+"_"+t+"distances.csv");
//		}
		info.getVarianceMatrixInverseAll().put(key,Nd4j.create(toFloatArray(inv.getData()))) ;
		info.getSingularValues().put(key, singularValues);
		info.getVarianceMatrixCondNum().put(key, varCondNum);
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
						varianceVectorAll.get(key).transpose().mmul(KInverse).mmul(z_mb).getDouble(0,0)/this.variogram.getTtScale().getDouble(n,t);//Fix the Z_MB part!!!
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
				if(Z_MB.size(0)!=info.getVarianceMatrixInverseAll().get(key).size(0)) {
					int inverseSize=(int) info.getVarianceMatrixInverseAll().get(key).size(0);
					int z_mbsize=(int) Z_MB.size(0);
					int realSize=this.variogram.getNtSpecificOriginalIndices().get(key).size();
					System.out.println("Debug Point");
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
//		INDArray varianceMatrix=info.getVarianceMatrixAll().get(key);
//		INDArray distanceMatrix=this.variogram.getDistances().get(key);
//		KrigingInterpolator.writeINDArray(varianceMatrix, "Network/ND/"+n+"_"+t+"variances.csv");
//		KrigingInterpolator.writeINDArray(distanceMatrix, "Network/ND/"+n+"_"+t+"distances.csv");
		INDArray inverseMatrix=info.getVarianceMatrixInverseAll().get(key);
		//double condNum=new SingularValueDecomposition(MatrixUtils.createRealMatrix(info.getVarianceMatrixAll().get(key).toDoubleMatrix())).getConditionNumber();
//		//double sigma=this.variogram.get
		//INDArray inverseMatrix1=Nd4j.create(new CholeskyDecomposition(MatrixUtils.createRealMatrix(info.getVarianceMatrixAll().get(key).toDoubleMatrix())).getSolver().getInverse().getData());
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
		new KrigingModelWriter(kriging).writeModel("Network/ND/ModelNormal");
		//KrigingInterpolator krigingnew=new KrigingModelReader().readModel("Network/ND/Model1/modelDetails.xml");
		Map<Integer,Tuple<INDArray,INDArray>> testingData=DataIO.readDataSet("Network/ND/DataSetNDTest.txt");
		INDArray averageError=Nd4j.create(kriging.N,kriging.T);
		for(Tuple<INDArray,INDArray> testData:testingData.values()) {
			INDArray Yreal=testData.getSecond();
			INDArray y=kriging.getY(testData.getFirst());
			INDArray errorArray=Yreal.sub(y).div(Yreal).mul(100);
			for(int i=0;i<errorArray.size(0);i++) {
				for(int j=0;j<errorArray.size(1);j++) {
					errorArray.putScalar(i, j,Math.abs(errorArray.getDouble(i,j)));
				}
			}
			averageError.addi(errorArray);
		}
		averageError.div(testingData.size());
		Nd4j.writeTxt(averageError, "Network/ND/ModelNormal/averagePredictionError.txt");
		System.out.println("Model Read Succesful!!!");
		
	}
	
	
	
	public void trainKriging() {
		long startTime=System.currentTimeMillis();
//		Thread[] thread=new Thread[Runtime.getRuntime().availableProcessors()-1];
//		int k=1;
//		List<List<String>> n_tlists=new ArrayList<>();
//		for(int i=0;i<thread.length;i++) {
//			n_tlists.add(new ArrayList<>());
//		}
//		for(int n=0;n<N;n++) {
//			for(int t=0;t<T;t++) {
//				n_tlists.get(k%thread.length).add(Integer.toString(n)+"_"+Integer.toString(t));
//				k++;
//			}
//		}
//		for(int i=0;i<thread.length;i++) {
//			thread[i]=new Thread(new krigingTrainer(this, n_tlists.get(i), info));
//		}
//		for(int i=0;i<thread.length;i++) {
//			thread[i].start();
//		}
//		for(int i=0;i<thread.length;i++) {
//			try {
//				thread[i].join();
//			} catch (InterruptedException e) {
//				// TODO Auto-generated catch block
//				e.printStackTrace();
//			}
//		}
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
			if(this.variogram.getSigmaMatrix().getDouble(n,t)==0) {
				//continue;
				return;
			}
			double initialBeta=this.beta.getDouble(n,t);
			double initialTheta=1/this.variogram.getDistances().get(key).maxNumber().doubleValue()*10;
				Calcfc calcfc = new Calcfc() {
					int it=0;

					@Override
					public double compute(int N, int m, double[] x, double[] con) {
						double theta=initialTheta+initialTheta*x[0]/100;
						double beta=initialBeta+initialBeta*x[1]/100;
						double obj=KrigingInterpolator.this.calcNtSpecificLogLikelihood(n, t, theta, beta,info);
						if(theta==0) {
							obj=10000000000000.;
						}
						con[0]=100*(theta-0.0000001);
						
						con[1]=-1*info.getVarianceMatrixCondNum().get(key)+10000;
						it++;
//						if(it==1) {
//							System.out.println("initial obj = "+-1*obj);
//						}
						return -1*obj;
					}
				};
				
				double[] x = {1,1};
				CobylaExitStatus result = Cobyla.findMinimum(calcfc, 2, 2, x, 0.5, .0001, 0, 800);
				this.beta.putScalar(n, t,initialBeta+initialBeta*x[1]/100);
				this.variogram.gettheta().putScalar(n,t,initialTheta+initialTheta*x[0]/100);
				//System.out.println("current total liklihood after "+key+" = "+this.calcCombinedLogLikelihood());
		});
////		}
		KrigingModelWriter writer=new KrigingModelWriter(this);
		//writer.writeModel("Network/ND/Model1/");
		System.out.println("Total time for training for "+N+" links and "+T+" time setps = "+Long.toString(System.currentTimeMillis()-startTime));
	}
	public void deepTrainKriging() {
		long startTime=System.currentTimeMillis();
		List<String> n_tlist=new ArrayList<>();
		for(int n=0;n<N;n++) {
			for(int t=0;t<T;t++) {
				n_tlist.add(Integer.toString(n)+"_"+Integer.toString(t));
				}
			}
	
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
						KrigingInterpolator.this.variogram.calcDistanceMatrix(n, t, cn, ct);
						double obj=KrigingInterpolator.this.calcNtSpecificLogLikelihood(n, t, theta, beta,info);
						if(theta==0) {
							obj=10000000000000.;
						}
						con[0]=100*(theta-0.0000001);
						
						con[1]=-1*info.getVarianceMatrixCondNum().get(key)+10000;
						it++;
//						if(it==1) {
//							System.out.println("initial obj = "+-1*obj);
//						}
						return -1*obj;
					}
				};
				
				double[] x = {1,1,1,1};
				CobylaExitStatus result = Cobyla.findMinimum(calcfc, 4, 2, x, 10, .01, 0, 100);
				this.beta.putScalar(n, t,initialBeta+initialBeta*x[1]/100);
				this.variogram.gettheta().putScalar(n,t,initialTheta+initialTheta*x[0]/100);
				double cn=initialCn+initialCn*x[2]/100;
				double ct=initialCt+initialCt*x[3]/100;
				this.variogram.calcDistanceMatrix(n,t,cn, ct);
				this.Cn.put(n,t,cn);
				this.Ct.put(n,t,ct);
				//System.out.println("current total liklihood after "+key+" = "+this.calcCombinedLogLikelihood());
				System.out.println("optim for case "+key+Long.toString(System.currentTimeMillis()-startTiment));
		});
		//}
		KrigingModelWriter writer=new KrigingModelWriter(this);
		writer.writeModel("Network/ND/Model1/");
		System.out.println("Total time for training for "+N+" links and "+T+" time setps = "+Long.toString(System.currentTimeMillis()-startTime));
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
			Calcfc calcfc = new Calcfc() {
				int it=0;

				@Override
				public double compute(int N, int m, double[] x, double[] con) {
					double theta=initialTheta+initialTheta*x[0]/100;
					double beta=initialBeta+initialBeta*x[1]/100;
					double obj=kriging.calcNtSpecificLogLikelihood(n, t, theta, beta,info);
					if(theta==0) {
						obj=10000000000000.;
					}
					con[0]=100*(theta-0.0000001);

					con[1]=-1*info.getVarianceMatrixCondNum().get(key)+10000;
					it++;
//					if(it==1) {
//						System.out.println("initial obj = "+-1*obj);
//					}
					return -1*obj;
				}
			};

			double[] x = {1,1};
			System.out.println("Current "+key);
			CobylaExitStatus result = Cobyla.findMinimum(calcfc, 2, 2, x, 0.5, .0001, 0, 800);
			this.kriging.getBeta().putScalar(n, t,initialBeta+initialBeta*x[1]/100);
			this.kriging.getVariogram().gettheta().putScalar(n,t,initialTheta+initialTheta*x[0]/100);
			totalNT++;
			//System.out.println("current total liklihood after "+key+" = "+this.kriging.calcCombinedLogLikelihood());
		}
	
	}
}






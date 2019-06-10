package kriging;

import java.lang.reflect.InvocationTargetException;
import java.util.HashMap;
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
import org.nd4j.linalg.checkutil.CheckUtil;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.INDArrayIndex;
import org.nd4j.linalg.indexing.NDArrayIndex;
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
	private int I;
	
	
	public KrigingInterpolator(Map<Integer,Tuple<INDArray,INDArray>> trainingDataSet, LinkToLinks l2ls,BaseFunction bf) {
		this.trainingDataSet=trainingDataSet;
		this.l2ls=l2ls;
		this.baseFunction=bf;
		this.variogram=new Variogram(trainingDataSet, this.l2ls);
		this.N=Math.toIntExact(trainingDataSet.get(0).getFirst().size(0));
		this.T=Math.toIntExact(trainingDataSet.get(0).getFirst().size(1));
		this.I=trainingDataSet.size();
		this.beta=Nd4j.zeros(N,T).addi(1);
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
		Map<String,INDArray> varianceMatrixInverseAll=new ConcurrentHashMap<>();
		for(Entry<String,INDArray> n_t_K:varianceMatrixAll.entrySet()) {
			//n_t_K.getValue()
			RealMatrix inv=new SingularValueDecomposition(MatrixUtils.createRealMatrix(n_t_K.getValue().toDoubleMatrix())).getSolver().getInverse();
			varianceMatrixInverseAll.put(n_t_K.getKey(),Nd4j.create(toFloatArray(inv.getData()))) ;
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
		IntStream.rangeClosed(0,Math.toIntExact(X.size(0))-1).parallel().forEach((n)->
		{
			IntStream.rangeClosed(0,Math.toIntExact(X.size(1))-1).parallel().forEach((t)->{
				String key=Integer.toString(n)+"_"+Integer.toString(t);
				INDArray KInverse=info.getVarianceMatrixInverseAll().get(key);
				double y=Y_b.getDouble(n,t)*beta.getDouble(n,t)+
						varianceVectorAll.get(key).mmul(KInverse).mmul(Z_MB.get(new INDArrayIndex[] {NDArrayIndex.point(n),NDArrayIndex.point(t),NDArrayIndex.all()})).getDouble(0,0);
				Y.putScalar(n,t,y);
			});
		});
		return X;
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
				INDArray Z_MB=Nd4j.create(info.getZ_MB().size(2),1);
				Z_MB.put(new INDArrayIndex[] {NDArrayIndex.all(),NDArrayIndex.point(1)}, info.getZ_MB().get(new INDArrayIndex[] {NDArrayIndex.point(n),NDArrayIndex.point(t),NDArrayIndex.all()}));
				double Logdet_k=0;
				//log here for ease

				SingularValueDecomposition svd=new SingularValueDecomposition(CheckUtil.convertToApacheMatrix(info.getVarianceMatrixAll().get(key)));
				for(double dd:svd.getSingularValues()) {
					if(dd!=0) {
						Logdet_k+=Math.log(dd);
					}
				}
				INDArray secondLLTerm=Z_MB.transpose().mmul(info.getVarianceMatrixInverseAll().get(key)).mmul(Z_MB);
				double d=-1*this.I/2.0*Math.log(2*Math.PI)-
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
	
	public double calcCombinedLogLikelihood() {
		return this.calcCombinedLogLikelihood(this.variogram.gettheta(),this.getBeta());
	}
	
	public static void main(String[] args) throws NoSuchMethodException, SecurityException, ClassNotFoundException, IllegalAccessException, IllegalArgumentException, InvocationTargetException {
		DataTypeUtil.setDTypeForContext(DataType.FLOAT);
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
		//KrigingInterpolator kriging=new KrigingInterpolator(trainingData, l2ls, new FreeFlowPlusWebstarBaseFunction(l2ls));
		//System.out.println(kriging.calcCombinedLogLikelihood());
		
		
//		double[][] a=new double[][]{{1.,2,3},{4,5,6},{7,8,9}};
//		INDArray aa=Nd4j.create(a);
//		INDArray bb=Nd4j.create(a);
//		System.out.println(Arrays.deepToString(a));
//		System.out.println(Arrays.toString(Nd4j.concat(0, aa.reshape(aa.length()),bb.reshape(bb.length())).toDoubleVector()));
//		System.out.println(Arrays.deepToString(Nd4j.concat(0, aa.reshape(aa.length()),bb.reshape(bb.length())).reshape(6,3).get(new INDArrayIndex[] {NDArrayIndex.interval(0, 3)}).toDoubleMatrix()));
	}
	
	public void trainKriging() {
		FunctionSPSA function=new FunctionSPSA();
		Evaluator evaluator=new Evaluator() {
			private int n=0;
			private int t=0;

			@Override
			public double evaluate(Vector<Double> thetaa) {
				return 0;
			}

			@Override
			public double evaluate(double[] thetaa) {
				Tuple<INDArray,INDArray> betaTheta=KrigingInterpolator.this.scaleFromVecor(thetaa);
				double liklihood=KrigingInterpolator.this.calcCombinedLogLikelihood(betaTheta.getFirst(), betaTheta.getSecond());
				return liklihood;
			}
			
		};
		function.setEvaluator(evaluator);
		double[] solution=function.runSPSA(100, this.scaleToVector(this.beta, this.variogram.gettheta()), .5, 300, 10, 100, .6, .1);
		Tuple<INDArray,INDArray> betaTheta=this.scaleFromVecor(solution);
	}
	
	private double[] scaleToVector(INDArray beta, INDArray theta) {
		INDArray linearBeta=beta.reshape(beta.length());
		INDArray linearTheta=theta.reshape(theta.length());
		return Nd4j.concat(0, linearBeta,linearTheta).toDoubleVector();
	}
	//beta is the first INDArray and theta is the second INDArray
	private Tuple<INDArray,INDArray> scaleFromVecor(double[] vector) {
		INDArray rawarray=Nd4j.create(vector);
		INDArray beta=rawarray.reshape(this.N*2,this.T).get(new INDArrayIndex[] {NDArrayIndex.interval(0,this.N),NDArrayIndex.all()});
		INDArray theta=rawarray.reshape(this.N*2,this.T).get(new INDArrayIndex[] {NDArrayIndex.interval(this.N,this.N*2),NDArrayIndex.all()});
		return new Tuple<>(beta,theta);
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




package kriging;

import java.util.HashMap;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

import org.nd4j.linalg.api.ndarray.INDArray;

public class VarianceInfoHolder{
	private INDArray Z_MB;
	private	Map<String,INDArray>varianceMatrixAll; 
	private Map<String,INDArray> varianceMatrixInverseAll;
	private Map<String,double[]> singularValues;
	private Map<String,Double> varianceMatrixCondNum;
	private Map<String,Double> logDeterminant=new ConcurrentHashMap<>();;
	
	public VarianceInfoHolder(INDArray Z_MB,Map<String,INDArray>varianceMatrixAll,Map<String,INDArray> varianceMatrixInverseAll, Map<String,double[]> singularValues,Map<String,Double> varCondNum) {
		this.Z_MB=Z_MB;
		this.varianceMatrixAll=varianceMatrixAll;
		this.varianceMatrixInverseAll=varianceMatrixInverseAll;
		this.singularValues=singularValues;
		this.varianceMatrixCondNum=varCondNum;
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

	public Map<String, double[]> getSingularValues() {
		return singularValues;
	}

	public Map<String, Double> getVarianceMatrixCondNum() {
		return varianceMatrixCondNum;
	}

	public Map<String, Double> getLogDeterminant() {
		return logDeterminant;
	}

	public void setLogDeterminant(Map<String, Double> logDeterminant) {
		this.logDeterminant = logDeterminant;
	}
	
	
	
}
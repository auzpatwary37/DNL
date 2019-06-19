package kriging;

import java.util.Map;

import org.nd4j.linalg.api.ndarray.INDArray;

public class VarianceInfoHolder{
	private final INDArray Z_MB;
	private final Map<String,INDArray>varianceMatrixAll; 
	private final Map<String,INDArray> varianceMatrixInverseAll;
	private final Map<String,double[]> singularValues;
	
	public VarianceInfoHolder(INDArray Z_MB,Map<String,INDArray>varianceMatrixAll,Map<String,INDArray> varianceMatrixInverseAll, Map<String,double[]> singularValues) {
		this.Z_MB=Z_MB;
		this.varianceMatrixAll=varianceMatrixAll;
		this.varianceMatrixInverseAll=varianceMatrixInverseAll;
		this.singularValues=singularValues;
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
	
}
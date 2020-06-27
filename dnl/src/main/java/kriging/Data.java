package kriging;

import org.nd4j.linalg.api.ndarray.INDArray;

/**
 * Class to hold one data instance so that it can be recovered later. 
 * @author h
 *
 */
public class Data {
	private final INDArray X;
	private final INDArray Y;
	private INDArray R;
	private final String key;
	
	public Data(INDArray X,INDArray Y,String key) {
		this.X=X;
		this.Y=Y;
		this.key=key;
	}

	public INDArray getX() {
		return X;
	}

	public INDArray getY() {
		return Y;
	}

	public String getKey() {
		return key;
	}

	public INDArray getR() {
		return R;
	}

	public void setR(INDArray r) {
		R = r;
	}
	
	
}

package kriging;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.w3c.dom.Element;

interface BaseFunction{
	
	public INDArray getY(INDArray X);
	public void writeBaseFunctionInfo(Element baseFunction);
	
	
}

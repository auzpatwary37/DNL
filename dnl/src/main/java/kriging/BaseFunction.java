package kriging;

import org.apache.commons.math3.linear.RealMatrix;
import org.w3c.dom.Element;

interface BaseFunction{
	
	public RealMatrix getY(RealMatrix X);
	public void writeBaseFunctionInfo(Element baseFunction);
	
	
}

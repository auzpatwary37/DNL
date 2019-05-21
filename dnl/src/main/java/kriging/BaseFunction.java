package kriging;

import org.apache.commons.math3.linear.RealMatrix;

interface BaseFunction{
	
	public RealMatrix getY(RealMatrix X);
	public void writeBaseFunction(String fileLoc);
}

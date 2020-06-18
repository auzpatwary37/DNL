package training;

import org.apache.commons.math3.util.FastMath;
import org.apache.commons.math3.util.Precision;

public class SVDDet {

	 
	
	 public static double getDet(double[] singularValues) {
		 double tol = FastMath.max(singularValues.length * singularValues[0] * 0x1.0p-52,  FastMath.sqrt(Precision.SAFE_MIN));
		 double det=1;
		 for(double sv:singularValues) {
			 if(sv>tol) {
				 det=det*sv;
			 }
		 }
		 return det;
	 }
	
	
}

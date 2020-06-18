package kriging;

import java.util.ArrayList;
import java.util.List;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

//import com.mathworks.toolbox.javabuilder.MWClassID;
//import com.mathworks.toolbox.javabuilder.MWJavaObjectRef;
//import com.mathworks.toolbox.javabuilder.MWNumericArray;
//
//import CalcInverse.Inverse;


public class MatlabInverse {
/**
 * This will perform the matrix inversion in matlab
 * 
 */
	public INDArray getInverse(INDArray mat) {
		
//		Inverse inv = null;
//		MWNumericArray x0 = null;	
//		MWNumericArray x1 = null;	
//		MWJavaObjectRef origRef = null;
//		Object[] result = null;	
//		try
//		{
//
//			/* Instantiate a new Java object */
//			/* This should only be done once per application instance */
//			inv = new Inverse();
//			origRef=new MWJavaObjectRef(this);
//			try {
//				x0 = new MWNumericArray(mat.toDoubleMatrix(), MWClassID.DOUBLE);
//				result=inv.CalcInverse(1,origRef,x0);
//				x1=(MWNumericArray)result[0];
//			}catch(Exception e) {
//				System.out.println(e);
//			}
//		}catch(Exception e) {
//			System.out.println(e);
//		}
//		INDArray out=Nd4j.create((double[][])x1.toDoubleArray());
		INDArray out=Nd4j.zeros(mat.shape());
//		x0.dispose();
//		x1.dispose();
//		inv.dispose();
		return out;
	}
	
	public static void main(String[] args) {
		INDArray a=Nd4j.rand(5,5);
		INDArray b=new MatlabInverse().getInverse(a);
		System.out.println(b);
	}
	
}

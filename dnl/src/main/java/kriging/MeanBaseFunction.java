package kriging;

import java.util.Arrays;
import java.util.Map;
import java.util.stream.IntStream;

import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealMatrixFormat;
import org.matsim.api.core.v01.Id;
import org.matsim.core.utils.collections.Tuple;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.w3c.dom.Element;
import org.xml.sax.Attributes;

import linktolinkBPR.LinkToLink;

public class MeanBaseFunction implements BaseFunction{
	private INDArray mean;
	public MeanBaseFunction() {
		
	}
	public static void shout() {
		System.out.println("Shout!!!!");
	}
	
	
	public MeanBaseFunction( Map<Integer, Data> trainingData) {
		int N=Math.toIntExact(trainingData.get(0).getX().size(0));
		int T=Math.toIntExact(trainingData.get(0).getX().size(1));
		this.mean=Nd4j.create(N,T);
		IntStream.rangeClosed(0,N-1).parallel().forEach((n)->
		{
			IntStream.rangeClosed(0,T-1).parallel().forEach((t)->{
				double[] data=new double[trainingData.size()];
				for(int i=0;i<trainingData.size();i++) {
					data[i]=trainingData.get(i).getY().getDouble(n, t);
				}
				mean.putScalar(n, t, Arrays.stream(data).sum()/data.length);
			});
		});
	}
	
	public MeanBaseFunction(INDArray mean) {
		this.mean=mean;
	}
	
	@Override
	public INDArray getY(INDArray X) {
		return mean;
	}
	@Override
	public void writeBaseFunctionInfo(Element baseFunction,String fileLoc) {
		baseFunction.setAttribute("ClassName", this.getClass().getName());
		baseFunction.setAttribute("meanFileLoc", fileLoc+"/meanBase.xml");
		Nd4j.writeTxt(this.mean, fileLoc+"/meanBase.xml");
	}

	public static BaseFunction parseBaseFunction(Attributes a) {
		String fileLoc=a.getValue("meanFileLoc");
		INDArray mean=Nd4j.readTxt(a.getValue("meanFileLoc"));
		return new MeanBaseFunction(mean);
	}
	
	
	
}


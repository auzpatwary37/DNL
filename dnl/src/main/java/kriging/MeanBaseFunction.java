package kriging;

import java.util.Arrays;
import java.util.Map;
import java.util.stream.IntStream;

import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealMatrixFormat;
import org.matsim.api.core.v01.Id;
import org.matsim.core.utils.collections.Tuple;
import org.w3c.dom.Element;
import org.xml.sax.Attributes;

import linktolinkBPR.LinkToLink;

public class MeanBaseFunction implements BaseFunction{
	private RealMatrix mean;
	public MeanBaseFunction() {
		
	}
	public static void shout() {
		System.out.println("Shout!!!!");
	}
	
	
	public MeanBaseFunction( Map<Integer,Tuple<RealMatrix,RealMatrix>> trainingDataSet) {
		this.mean=new Array2DRowRealMatrix();
		IntStream.rangeClosed(0,trainingDataSet.get(0).getSecond().getRowDimension()-1).forEach((n)->
		{
			IntStream.rangeClosed(0,trainingDataSet.get(0).getSecond().getColumnDimension()-1).forEach((t)->{
				double[] data=new double[trainingDataSet.size()];
				for(int i=0;i<trainingDataSet.size();i++) {
					data[i]=trainingDataSet.get(i).getSecond().getEntry(n, t);
				}
				mean.setEntry(n, t, Arrays.stream(data).sum()/data.length);
			});
		});
	}
	
	public MeanBaseFunction(RealMatrix mean) {
		this.mean=mean;
	}
	
	@Override
	public RealMatrix getY(RealMatrix X) {
		return mean;
	}
	@Override
	public void writeBaseFunctionInfo(Element baseFunction) {
		baseFunction.setAttribute("ClassName", this.getClass().getName());
		baseFunction.setAttribute("mean", mean.toString());
	}

	public static BaseFunction parseBaseFunction(Attributes a) {
		RealMatrix mean=RealMatrixFormat.getInstance().parse(a.getValue("mean"));
		return new MeanBaseFunction(mean);
	}
	
	
	
}


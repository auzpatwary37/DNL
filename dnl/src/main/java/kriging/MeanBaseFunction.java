package kriging;

import java.util.Arrays;
import java.util.Map;
import java.util.stream.IntStream;

import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.linear.RealMatrix;
import org.matsim.core.utils.collections.Tuple;

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
	
	@Override
	public RealMatrix getY(RealMatrix X) {
		return mean;
	}

	@Override
	public void writeBaseFunction(String fileLoc) {
		// TODO Auto-generated method stub
		
	}
	
}

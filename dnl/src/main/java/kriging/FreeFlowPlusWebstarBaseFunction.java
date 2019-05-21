package kriging;

import java.util.stream.IntStream;

import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;

import linktolinkBPR.LinkToLinks;

public class FreeFlowPlusWebstarBaseFunction implements BaseFunction{
	
	private final LinkToLinks l2ls;
	
	public FreeFlowPlusWebstarBaseFunction( LinkToLinks l2ls) {
		this.l2ls=l2ls;
	}
	
	@Override
	public RealMatrix getY(RealMatrix X) {
		RealMatrix Y=MatrixUtils.createRealMatrix(X.getRowDimension(), X.getColumnDimension());
		IntStream.rangeClosed(0,X.getRowDimension()-1).forEach((n)->
		{
			IntStream.rangeClosed(0,X.getColumnDimension()-1).forEach((t)->{
				double tt=l2ls.getLinkToLinks().get(n).getLinkToLinkWebstarDelay(X.getEntry(n, t), t);
				Y.setEntry(n, t, tt);
			});
		});
		return Y;
	}

	@Override
	public void writeBaseFunction(String fileLoc) {
		
	}
	
	
}

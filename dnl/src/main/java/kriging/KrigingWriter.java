package kriging;

import java.util.Map;

import org.apache.commons.math3.linear.RealMatrix;
import org.matsim.core.utils.collections.Tuple;

public class KrigingWriter {
	private final Map<Integer,Tuple<RealMatrix,RealMatrix>> trainingDataSet;
	private final Map<String,RealMatrix> weights;
	private final RealMatrix theta;
	
	public KrigingWriter(KrigingInterpolator model) {
		this.trainingDataSet
	}
}

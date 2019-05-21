package kriging;

import java.lang.reflect.InvocationTargetException;
import java.lang.reflect.Method;
import java.util.Map;
import org.apache.commons.math3.linear.RealMatrix;
import org.matsim.core.utils.collections.Tuple;
import linktolinkBPR.LinkToLinks;

/**
 * 
 * A kriging framework
 * @author Ashraf
 *
 */
public class KrigingInterpolator{
	private final Map<Integer,Tuple<RealMatrix,RealMatrix>> trainingDataSet;
	private LinkToLinks l2ls;
	private Variogram variogram;
	private RealMatrix beta;
	private BaseFunction baseFunction;
	
	
	public KrigingInterpolator(Map<Integer,Tuple<RealMatrix,RealMatrix>> trainingDataSet, LinkToLinks l2ls) {
		this.trainingDataSet=trainingDataSet;
		this.l2ls=l2ls;
		this.variogram=new Variogram(trainingDataSet, this.l2ls);
		
	}
	
	
	//Constructor for creating in the reader
	public KrigingInterpolator(Variogram v, RealMatrix beta, BaseFunction bf) {
		this.trainingDataSet=v.getTrainingDataSet();
		this.variogram=v;
		this.beta=beta;
		this.variogram.getClass().toString();
		this.baseFunction=bf;
	}
	
	public Map<Integer, Tuple<RealMatrix, RealMatrix>> getTrainingDataSet() {
		return trainingDataSet;
	}


	public Variogram getVariogram() {
		return variogram;
	}


	public RealMatrix getBeta() {
		return beta;
	}


	public BaseFunction getBaseFunction() {
		return baseFunction;
	}
	
	public static void main(String[] args) throws NoSuchMethodException, SecurityException, ClassNotFoundException, IllegalAccessException, IllegalArgumentException, InvocationTargetException {
		BaseFunction bf=new MeanBaseFunction();
		String s=bf.getClass().getName().toString();
		System.out.println(s);
		Class.forName(s);
		Method parse=Class.forName(s).getMethod("shout");
		parse.invoke(null);
	}
	

}






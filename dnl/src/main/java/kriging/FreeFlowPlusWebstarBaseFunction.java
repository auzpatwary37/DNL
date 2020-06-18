package kriging;

import java.util.Map;
import java.util.Map.Entry;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;
import java.util.stream.IntStream;

import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealMatrixFormat;
import org.matsim.api.core.v01.Id;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.conditions.Conditions;
import org.w3c.dom.Element;
import org.xml.sax.Attributes;

import linktolinkBPR.LinkToLink;
import linktolinkBPR.LinkToLinks;

public class FreeFlowPlusWebstarBaseFunction implements BaseFunction{
	public static int no=0;
	private Map<Integer,Link2LinkInfoHolder> link2LinkInfo=new ConcurrentHashMap<>();
	
	public FreeFlowPlusWebstarBaseFunction( LinkToLinks l2ls) {
		for(int i=0;i<l2ls.getLinkToLinks().size();i++) {
			link2LinkInfo.put(i, new Link2LinkInfoHolder(l2ls.getLinkToLinks().get(i),i));
		}
	}
	
	private FreeFlowPlusWebstarBaseFunction(Map<Integer,Link2LinkInfoHolder> link2LinkInfo) {
		this.link2LinkInfo=link2LinkInfo;
	}
	@Override
	public INDArray getY(INDArray X) {
		INDArray Y=Nd4j.create(X.size(0), X.size(1));
		no++;
//		IntStream.rangeClosed(0,Math.toIntExact(X.size(0))-1).parallel().forEach((n)->
//		{
//			IntStream.rangeClosed(0,Math.toIntExact(X.size(1))-1).parallel().forEach((t)->{
		
		for(int n=0;n<X.size(0);n++) {
			for(int t=0;t<X.size(1);t++) {
				double tt=this.getLinkToLinkWebstarDelay(X.getDouble(n, t), n);
				Y.putScalar(n, t, tt);
				if(Y.cond(Conditions.isInfinite()).any()||Y.cond(Conditions.isNan()).any()) {
					System.out.println("Z is nan or inf!!!");
				}
				
			}
		}
//			});
//		});
		
		return Y;
	}


	
	public Double getLinkToLinkWebstarDelay(double demand,Integer n) {
		Link2LinkInfoHolder l2l=this.link2LinkInfo.get(n);
		Double delay=l2l.getFromLinkFreeFlowTime()+ l2l.getCycleTime()/2*(1-l2l.getG_cRatio())*(1-l2l.getG_cRatio())/(1-l2l.getG_cRatio()*(demand/1.2*l2l.getSaturationFlow()));
		if(Double.isNaN(delay)||delay==Double.POSITIVE_INFINITY) {
			System.out.println();
		}
		return delay;
	}

	@Override
	public void writeBaseFunctionInfo(Element baseFunction,String fileLoc) {
		baseFunction.setAttribute("ClassName", this.getClass().getName());
		for(Entry<Integer,Link2LinkInfoHolder>l2l:this.link2LinkInfo.entrySet()) {
			baseFunction.setAttribute(Integer.toString(l2l.getKey()), l2l.getValue().toString());
		}
	}
	
	public static BaseFunction parseBaseFunction(Attributes a) {
		Map<Integer,Link2LinkInfoHolder> link2LinkInfo=new ConcurrentHashMap<>();
		for(int i=0;i<a.getLength();i++) {
			if(!a.getQName(i).equals("ClassName")) {
				Link2LinkInfoHolder l2l=Link2LinkInfoHolder.createLinkToLinkInfo(a.getValue(i));
				link2LinkInfo.put(l2l.getN(), l2l);
			}
		}
		return new FreeFlowPlusWebstarBaseFunction(link2LinkInfo);
	}

	@Override
	public double getntSpecificY(INDArray X, int n, int t) {
		double tt=this.getLinkToLinkWebstarDelay(X.getDouble(n, t), n);
		return tt;
	}
	
}
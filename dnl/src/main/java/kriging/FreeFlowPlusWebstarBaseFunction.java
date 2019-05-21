package kriging;

import java.util.Map;
import java.util.Map.Entry;
import java.util.concurrent.ConcurrentHashMap;
import java.util.stream.IntStream;

import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealMatrixFormat;
import org.matsim.api.core.v01.Id;
import org.w3c.dom.Element;
import org.xml.sax.Attributes;

import linktolinkBPR.LinkToLink;
import linktolinkBPR.LinkToLinks;

public class FreeFlowPlusWebstarBaseFunction implements BaseFunction{
	
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
	public RealMatrix getY(RealMatrix X) {
		RealMatrix Y=MatrixUtils.createRealMatrix(X.getRowDimension(), X.getColumnDimension());
		IntStream.rangeClosed(0,X.getRowDimension()-1).forEach((n)->
		{
			IntStream.rangeClosed(0,X.getColumnDimension()-1).forEach((t)->{
				double tt=this.getLinkToLinkWebstarDelay(X.getEntry(n, t), n);
				Y.setEntry(n, t, tt);
			});
		});
		return Y;
	}


	
	public double getLinkToLinkWebstarDelay(double demand,Integer n) {
		Link2LinkInfoHolder l2l=this.link2LinkInfo.get(n);
		return l2l.getFromLinkFreeFlowTime()+ l2l.getCycleTime()/2*(1-l2l.getG_cRatio())*(1-l2l.getG_cRatio())/(1-l2l.getG_cRatio()*(demand/l2l.getSaturationFlow()));
	}

	@Override
	public void writeBaseFunctionInfo(Element baseFunction) {
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
	
}


class Link2LinkInfoHolder{
	private final Id<LinkToLink> linkToLinkId;
	private final int n;
	private final double cycleTime;
	private final double saturationFlow;
	private final double g_cRatio;
	private final double fromLinkFreeFlowTime;
	
	public Link2LinkInfoHolder(LinkToLink l2l, int n) {
		this.linkToLinkId=l2l.getLinkToLinkId();
		this.n=n;
		this.cycleTime=l2l.getCycleTime();
		this.saturationFlow=l2l.getSupply();
		this.g_cRatio=l2l.getG_cRatio();
		this.fromLinkFreeFlowTime=l2l.getFromLink().getLength()/l2l.getFromLink().getFreespeed();
	}
	
	public Link2LinkInfoHolder(String link2LinkId, int n,double cycleTime,double saturationFlow,double g_cRatio,double fromLinkFreeFlowTime) {
		this.linkToLinkId=Id.create(link2LinkId, LinkToLink.class);
		this.n=n;
		this.cycleTime=cycleTime;
		this.saturationFlow=saturationFlow;
		this.g_cRatio=g_cRatio;
		this.fromLinkFreeFlowTime=fromLinkFreeFlowTime;
	}

	public Id<LinkToLink> getLinkToLinkId() {
		return linkToLinkId;
	}

	public int getN() {
		return n;
	}

	public double getCycleTime() {
		return cycleTime;
	}

	public double getSaturationFlow() {
		return saturationFlow;
	}

	public double getG_cRatio() {
		return g_cRatio;
	}
	
	
	public double getFromLinkFreeFlowTime() {
		return fromLinkFreeFlowTime;
	}

	@Override
	public String toString() {
		String s=Integer.toString(n)+","+this.linkToLinkId.toString()+","+this.cycleTime+","+this.saturationFlow+","+this.g_cRatio+","+this.fromLinkFreeFlowTime;
		return s;
	}
	
	public static Link2LinkInfoHolder createLinkToLinkInfo(String s) {
		String[] part=s.split(",");
		return new Link2LinkInfoHolder(part[1],Integer.parseInt(part[0]),Double.parseDouble(part[2]),Double.parseDouble(part[3]),Double.parseDouble(part[4]),Double.parseDouble(part[5]));
	}
}
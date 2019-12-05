package kriging;

import java.util.Map;
import java.util.Set;

import org.matsim.api.core.v01.Id;

import linktolinkBPR.LinkToLink;

public class Link2LinkInfoHolder{
	private final Id<LinkToLink> linkToLinkId;
	private final int n;
	private final double cycleTime;
	private final double saturationFlow;
	private final double g_cRatio;
	private final double fromLinkFreeFlowTime;
	private final Map<Integer,Set<Integer>> proximityMap;
	private final Set<Integer> primaryFromLinkProximitySet;
	
	public Link2LinkInfoHolder(LinkToLink l2l, int n) {
		this.proximityMap=l2l.getProximityMap();
		this.primaryFromLinkProximitySet=l2l.getPrimaryFromLinkProximitySet();
		this.linkToLinkId=l2l.getLinkToLinkId();
		this.n=n;
		this.cycleTime=l2l.getCycleTime();
		this.saturationFlow=l2l.getSupply();
		this.g_cRatio=l2l.getG_cRatio();
		this.fromLinkFreeFlowTime=l2l.getFromLink().getLength()/l2l.getFromLink().getFreespeed();
	}
	
	public Link2LinkInfoHolder(String link2LinkId, int n,double cycleTime,double saturationFlow,double g_cRatio,double fromLinkFreeFlowTime,Map<Integer,Set<Integer>> proximityMap,Set<Integer> pflps) {
		this.proximityMap=proximityMap;
		this.primaryFromLinkProximitySet=pflps;
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

	
	public Set<Integer> getPrimaryFromLinkProximitySet() {
		return primaryFromLinkProximitySet;
	}

	public Map<Integer, Set<Integer>> getProximityMap() {
		return proximityMap;
	}

	@Override
	public String toString() {
		String s=Integer.toString(n)+","+this.linkToLinkId.toString()+","+this.cycleTime+","+this.saturationFlow+","+this.g_cRatio+","+this.fromLinkFreeFlowTime+","+LinkToLink.writeProximityMap(proximityMap)+","+LinkToLink.writePrimaryFromLinkProximitySet(primaryFromLinkProximitySet);
		return s;
	}
	
	public static Link2LinkInfoHolder createLinkToLinkInfo(String s) {
		String[] part=s.split(",");
		return new Link2LinkInfoHolder(part[1],Integer.parseInt(part[0]),Double.parseDouble(part[2]),Double.parseDouble(part[3]),Double.parseDouble(part[4]),Double.parseDouble(part[5]),LinkToLink.parseProximityMatrix(part[6]),LinkToLink.parsePrimaryFromLinkProximitySet(part[7]));
	}
}
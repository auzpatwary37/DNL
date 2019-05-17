package linktolinkBPR;

import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

import org.matsim.api.core.v01.Id;
import org.matsim.api.core.v01.network.Link;
import org.matsim.core.utils.collections.Tuple;

/**
 * This will be the basic input for the DNL loading 
 * Consider the traveling time of the from link and the waiting time before entering the two link 
 * @author h
 *
 */
public class LinkToLink {
	
	private final Map<Integer,Tuple<Double,Double>> timeBean;
	private final Link fromLink;
	private final Link toLink;
	private Map<Integer,Double> demand=new ConcurrentHashMap<>();
	private Map<Integer,Double> supply=new ConcurrentHashMap<>();
	private final Id<LinkToLink> linkToLinkId;
	private double g_cRatio=1;
	private double cycleTime=0;
	
	
	public LinkToLink(Link fromLink, Link toLink,Map<Integer, Tuple<Double, Double>> timeBean2) {
		this.fromLink=fromLink;
		this.toLink=toLink;
		this.timeBean=timeBean2;
		for(Integer timeId:this.timeBean.keySet()) {
			this.demand.put(timeId, 0.);
			this.supply.put(timeId, 1800.);
		}
		
		this.linkToLinkId=Id.create(this.fromLink.getId()+"_"+this.toLink.getId(), LinkToLink.class);
	}
	
	public void addDemand(double demand,Integer timeBeanId) {
		if(this.timeBean.containsKey(timeBeanId)) {
			this.demand.put(timeBeanId, this.demand.get(timeBeanId)+demand);
		}else {
			throw new IllegalArgumentException("timeBeanId not recognized!!!");
		}
		
	}
	
	public double getCycleTime() {
		return cycleTime;
	}

	public void setCycleTime(double cycleTime) {
		this.cycleTime = cycleTime;
	}

	/**
	 * Gives webstar() delay of the timeBeanid + free flow speed 
	 * @param timeBeanId
	 * @return
	 */
	public double getLinkToLinkWebstarDelay(String timeBeanId) {
		return this.fromLink.getLength()/this.fromLink.getFreespeed()+ cycleTime/2*(1-this.g_cRatio)*(1-this.g_cRatio)/(1-this.g_cRatio*(this.demand.get(timeBeanId)/this.supply.get(timeBeanId)));
	}

	public double getG_cRatio() {
		return g_cRatio;
	}

	public void setG_cRatio(double g_cRatio) {
		this.g_cRatio = g_cRatio;
	}

	public Map<Integer, Tuple<Double, Double>> getTimeBean() {
		return timeBean;
	}

	public Link getFromLink() {
		return fromLink;
	}

	public Link getToLink() {
		return toLink;
	}

	public Map<Integer, Double> getDemand() {
		return demand;
	}

	public Map<Integer, Double> getSupply() {
		return supply;
	}

	public Id<LinkToLink> getLinkToLinkId() {
		return linkToLinkId;
	}
	
	
	
}

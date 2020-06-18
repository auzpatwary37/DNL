package linktolinkBPR;

import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Set;
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
	private double supply=1800;
	private final Id<LinkToLink> linkToLinkId;
	private double g_cRatio=1;
	private double cycleTime=60;
	private Map<Integer,Set<Integer>> proximityMap=null;
	private Set<Integer> primaryFromLinkProximitySet=null;
	
	public LinkToLink(Link fromLink, Link toLink,Map<Integer, Tuple<Double, Double>> timeBean2) {
		this.fromLink=fromLink;
		this.toLink=toLink;
		this.timeBean=timeBean2;
		for(Integer timeId:this.timeBean.keySet()) {
			this.demand.put(timeId, 0.);
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
	
	public void setSupply(double supply) {
		this.supply = supply;
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
	public double getLinkToLinkWebstarDelay(Integer timeBeanId) {
		return this.fromLink.getLength()/this.fromLink.getFreespeed()+ cycleTime/2*(1-this.g_cRatio)*(1-this.g_cRatio)/(1-this.g_cRatio*(this.demand.get(timeBeanId)/this.supply));
	}
	
	public double getLinkToLinkWebstarDelay(double demand,Integer timeBeanId) {
		return this.fromLink.getLength()/this.fromLink.getFreespeed()+ cycleTime/2*(1-this.g_cRatio)*(1-this.g_cRatio)/(1-this.g_cRatio*(demand/this.supply));
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
	
	public void setPrimaryFromLinkProximitySet(Set<Integer>ppm) {
		this.primaryFromLinkProximitySet=ppm;
	}
	
	public Set<Integer> getPrimaryFromLinkProximitySet(){
		return this.primaryFromLinkProximitySet;
	}

	public Link getToLink() {
		return toLink;
	}

	public Map<Integer, Double> getDemand() {
		return demand;
	}

	public double getSupply() {
		return supply;
	}

	public Id<LinkToLink> getLinkToLinkId() {
		return linkToLinkId;
	}
	
	public double getFreeFlowTT() {
		return this.fromLink.getLength()/this.fromLink.getFreespeed();
	}

	public Map<Integer, Set<Integer>> getProximityMap() {
		return proximityMap;
	}

	public void setProximityMap(Map<Integer, Set<Integer>> proximityMap) {
		this.proximityMap = proximityMap;
	}
	
	public String writeProximityMap() {
		String p="";
		String entrySeperator="";
		for(Entry<Integer, Set<Integer>> e:this.proximityMap.entrySet()) {
			if(e.getValue().size()!=0) {
				p=p+entrySeperator+e.getKey();
				p=p+"_";
				String elementSeperator="";
				for(Integer n:e.getValue()) {
					p=p+elementSeperator+n;
					elementSeperator=" ";
				}
				entrySeperator=",";
		}
		}
		return p;
	}
	
	
	
	//static variant of the same code
	public static String writeProximityMap(Map<Integer, Set<Integer>> proximityMap) {
		String p="";
		String entrySeperator="";
		for(Entry<Integer, Set<Integer>> e:proximityMap.entrySet()) {
			if(e.getValue().size()!=0) {
				p=p+entrySeperator+e.getKey();
				p=p+"_";
				String elementSeperator="";
				for(Integer n:e.getValue()) {
					p=p+elementSeperator+n;
					elementSeperator=" ";
				}
				entrySeperator=",";
		}
		}
		return p;
	}
	
	public static Map<Integer,Set<Integer>> parseProximityMatrix(String p){
		Map<Integer,Set<Integer>> proximityMatrix =new HashMap<>();
		String[] entries=p.split(",");
		for(String entry:entries) {
			Integer key=Integer.parseInt(entry.split("_")[0]);
			proximityMatrix.put(key, new HashSet<>());
			
			String[] nSet=entry.split("_")[1].split(" ");
				for(String nstring:nSet) {
					int n=Integer.parseInt(nstring);
					proximityMatrix.get(key).add(n);
				}
			}
		
		
		return proximityMatrix;
	}
	
	public String writePrimaryFromLinkProximitySet() {
		String p="";
		String elementSeperator="";
		for(Integer n:this.primaryFromLinkProximitySet) {
			p=p+elementSeperator+n;
			elementSeperator=" ";
		}
		return p;
	}
	
	public static String writePrimaryFromLinkProximitySet(Set<Integer>primaryFromLinkProximitySet ) {
		String p="";
		String elementSeperator="";
		for(Integer n:primaryFromLinkProximitySet) {
			p=p+elementSeperator+n;
			elementSeperator=" ";
		}
		return p;
	}
	
	public static Set<Integer> parsePrimaryFromLinkProximitySet(String s){
		Set<Integer> set=new HashSet<>();
		for(String ss:s.split(" ")) {
			set.add(Integer.parseInt(ss));
		}
		return set;
	}
	
}

package linktolinkBPR;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import org.matsim.api.core.v01.Id;
import org.matsim.api.core.v01.network.Link;
import org.matsim.api.core.v01.network.Network;
import org.matsim.core.network.NetworkUtils;
import org.matsim.core.utils.collections.Tuple;
import com.google.common.collect.BiMap;
import com.google.common.collect.HashBiMap;


public class LinkToLinks {
	private List<LinkToLink> linkToLinks=new ArrayList<>();
	private Map<Id<Link>,List<LinkToLink>>fromLinkToLinkMap=new HashMap<>();
	private Map<Id<Link>,List<LinkToLink>>ToLinkToLinkMap=new HashMap<>();
	private BiMap<Integer,Id<LinkToLink>> numToLinkToLink=HashBiMap.create();
	private final Network network;
	private final Map<String,Tuple<Double,Double>>timeBean;
	private int l2lCounter=0;
	/**
	 * 
	 * TODO:Add signal info in this constructor as well
	 * @param network
	 */
	public LinkToLinks(Network network,Map<String,Tuple<Double,Double>>timeBean) {
		this.network=network;
		this.timeBean=timeBean;
		this.generateLinkToLinkMap();
	}
	
	/**
	 * 
	 */
	public void generateLinkToLinkMap() {
		for(Entry<Id<Link>, ? extends Link> e:network.getLinks().entrySet()){
			for(Link l:e.getValue().getToNode().getOutLinks().values()) {
				LinkToLink l2l=new LinkToLink(e.getValue(),l,timeBean);
				this.addLinkToLink(l2l);
			}
			
		}
	}
	
	public List<LinkToLink> getLinkToLinks() {
		return linkToLinks;
	}

	public Map<Id<Link>, List<LinkToLink>> getFromLinkToLinkMap() {
		return fromLinkToLinkMap;
	}

	public Map<Id<Link>, List<LinkToLink>> getToLinkToLinkMap() {
		return ToLinkToLinkMap;
	}

	public BiMap<Integer, Id<LinkToLink>> getNumToLinkToLink() {
		return numToLinkToLink;
	}

	public Network getNetwork() {
		return network;
	}

	public Map<String, Tuple<Double, Double>> getTimeBean() {
		return timeBean;
	}

	public int getL2lCounter() {
		return l2lCounter;
	}

	public void addLinkToLink(LinkToLink l2l) {
		if(this.numToLinkToLink.inverse().containsKey(l2l.getLinkToLinkId())) {
			return;
		}
		this.linkToLinks.add(l2l);
		this.numToLinkToLink.put(l2lCounter, l2l.getLinkToLinkId());
		if(this.fromLinkToLinkMap.containsKey(l2l.getFromLink().getId())) {
			this.fromLinkToLinkMap.get(l2l.getFromLink().getId()).add(l2l);
		}else {
			this.fromLinkToLinkMap.put(l2l.getFromLink().getId(),new ArrayList<LinkToLink>());
			this.fromLinkToLinkMap.get(l2l.getFromLink().getId()).add(l2l);
		}
		
		if(this.ToLinkToLinkMap.containsKey(l2l.getToLink().getId())) {
			this.ToLinkToLinkMap.get(l2l.getToLink().getId()).add(l2l);
		}else {
			this.ToLinkToLinkMap.put(l2l.getToLink().getId(),new ArrayList<LinkToLink>());
			this.ToLinkToLinkMap.get(l2l.getToLink().getId()).add(l2l);
		}
		this.l2lCounter++;
	}
	
//	public static void main(String[] args) {
//		Network network=NetworkUtils.readNetwork("Network/SiouxFalls/siouxfallsNetwork.xml");
////		/Network network=NetworkUtils.readNetwork("Network/SiouxFalls/network.xml");
//		
//		Map<String,Tuple<Double,Double>> timeBean=new HashMap<>();
//		for(int i=0;i<24;i++) {
//			timeBean.put(Integer.toString(i),new Tuple<Double,Double>(i*3600.,i*3600.+3600));
//		}
//		LinkToLinks l2ls=new LinkToLinks(network,timeBean);
//		System.out.println("Done!!! Total LinkToLink = "+l2ls.getL2lCounter());
//	}
	
}

package linktolinkBPR;

import java.util.HashMap;
import java.util.Map;

import org.matsim.api.core.v01.Id;
import org.matsim.api.core.v01.network.Link;

public class Util {

	public static Map<String,Map<Id<Link>,Double>>getLinkFlow(Map<String, Map<Id<LinkToLink>,LinkToLink>>linkToLinkVolume){
		Map<String,Map<Id<Link>,Double>> linkVolume=new HashMap<>();
		for(String timeId:linkToLinkVolume.keySet()) {
			linkVolume.put(timeId, new HashMap<Id<Link>,Double>());
			for(LinkToLink l2l:linkToLinkVolume.get(timeId).values()) {
				Id<Link> fromLinkId=l2l.getFromLink().getId();
				if(linkVolume.get(timeId).containsKey(fromLinkId)) {
					linkVolume.get(timeId).put(fromLinkId,linkVolume.get(timeId).get(fromLinkId)+l2l.getDemand().get(timeId));
				}else {
					linkVolume.get(timeId).put(fromLinkId,l2l.getDemand().get(timeId));
				}
			}
		}	
		
		return linkVolume;
		
	}
}

package linktolinkBPR;

import java.util.ArrayList;
import java.util.List;

import org.matsim.api.core.v01.Id;
import org.matsim.api.core.v01.network.Link;


public class L2LVolumeTimeMatrix {
	
	
	
	public static void main(String[] args) {
		//DataFrame<Double> df;
		List<String> timeBeans=new ArrayList<>();
		
		timeBeans.add("1");
		timeBeans.add("2");
		
		List<Id<Link>> linkIds=new ArrayList<>();
		
		Id<Link> linkId1=Id.createLinkId("l1");
		Id<Link> linkId2=Id.createLinkId("l2");
		
		linkIds.add(linkId1);
		linkIds.add(linkId2);

//		df=new DataFrame<>(linkIds,timeBeans);
//		List<Double> a=new ArrayList<>();
//		a.add(5.);
//		a.add(6.);
//		df.append("l1",a);
//		
	
		
	}
	
	
}

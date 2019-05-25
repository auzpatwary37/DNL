package matsimIntegration;

import java.util.ArrayList;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

import org.matsim.api.core.v01.Id;
import org.matsim.api.core.v01.Scenario;
import org.matsim.api.core.v01.network.Link;
import org.matsim.api.core.v01.network.Network;
import org.matsim.api.core.v01.population.Leg;
import org.matsim.api.core.v01.population.Plan;
import org.matsim.api.core.v01.population.PlanElement;
import org.matsim.api.core.v01.population.Population;
import org.matsim.core.controler.events.AfterMobsimEvent;
import org.matsim.core.controler.events.BeforeMobsimEvent;
import org.matsim.core.controler.listener.AfterMobsimListener;
import org.matsim.core.controler.listener.BeforeMobsimListener;
import org.matsim.core.utils.collections.Tuple;
import org.nd4j.linalg.api.ndarray.INDArray;

import com.google.inject.Inject;

import linktolinkBPR.LinkToLinks;
import linktolinkBPR.LinkToLink;

public class DNLDataCollectionControlerListener implements BeforeMobsimListener, AfterMobsimListener{

	private 
	private final int N;
	private final int T;
	private final Map<Integer,Tuple<Double,Double>>timeBean;
	private LinkToLinks l2ls;
	
	private ArrayList<Tuple<INDArray,INDArray>> dataset=new ArrayList<>();
	@Inject
	private Scenario scenario;
	@Inject
	private Network network;
	private INDArray X;
	
	public DNLDataCollectionControlerListener(LinkToLinks l2ls) {
		this.N=l2ls.getL2lCounter();
		this.T=l2ls.getTimeBean().size();
		this.timeBean=l2ls.getTimeBean();
		this.l2ls=l2ls;
	}
	
	@Override
	public void notifyAfterMobsim(AfterMobsimEvent event) {
		// TODO Auto-generated method stub
	}

	@Override
	public void notifyBeforeMobsim(BeforeMobsimEvent event) {
		Map<String,Double> demand=new ConcurrentHashMap<>();
		Population population=event.getServices().getScenario().getPopulation();
		population.getPersons().entrySet().forEach((e)->{
			Plan plan=e.getValue().getSelectedPlan();
			for(PlanElement pl:plan.getPlanElements()) {
				Leg l;
				ArrayList<Id<Link>> links=new ArrayList<>();
				ArrayList<Tuple<Id<LinkToLink>,Double>> linkToLinksVsTime=new ArrayList<>();
				if(pl instanceof Leg) {
					l=(Leg)pl;
					String[] part=l.getRoute().getRouteDescription().split(" ");
					for(String s:part) {
						links.add(Id.createLinkId(s.trim()));
					}
					double time=l.getDepartureTime();
					for(int i=1;i<links.size();i++) {
						Id<LinkToLink> l2lId=Id.create(links.get(i-1)+"_"+links.get(i), LinkToLink.class);
						linkToLinksVsTime.add(new Tuple<>(l2lId,time));
						time+=this.l2ls.getLinkToLink(l2lId).getFreeFlowTT();
					}
				}else {
					continue;
				}
				l.getRoute().getRouteDescription();
			}
		});
	}

}

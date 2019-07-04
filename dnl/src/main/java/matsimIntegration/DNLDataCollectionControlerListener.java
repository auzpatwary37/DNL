package matsimIntegration;

import java.util.ArrayList;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.stream.IntStream;

import org.matsim.api.core.v01.Id;
import org.matsim.api.core.v01.network.Link;
import org.matsim.api.core.v01.population.Leg;
import org.matsim.api.core.v01.population.Plan;
import org.matsim.api.core.v01.population.PlanElement;
import org.matsim.api.core.v01.population.Population;
import org.matsim.core.api.experimental.events.EventsManager;
import org.matsim.core.controler.events.AfterMobsimEvent;
import org.matsim.core.controler.events.BeforeMobsimEvent;
import org.matsim.core.controler.events.ShutdownEvent;
import org.matsim.core.controler.events.StartupEvent;
import org.matsim.core.controler.listener.AfterMobsimListener;
import org.matsim.core.controler.listener.BeforeMobsimListener;
import org.matsim.core.controler.listener.ShutdownListener;
import org.matsim.core.controler.listener.StartupListener;
import org.matsim.core.trafficmonitoring.TravelTimeCalculator;
import org.matsim.core.utils.collections.Tuple;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import com.google.inject.Inject;
import com.google.inject.name.Named;

import kriging.Data;
import linktolinkBPR.LinkToLinks;
import training.DataIO;
import linktolinkBPR.LinkToLink;

public class DNLDataCollectionControlerListener implements BeforeMobsimListener, AfterMobsimListener,ShutdownListener,StartupListener{

	@Inject
	private LinkToLinkTTRecorder TTRecorder;
	private final int N;
	private final int T;
	private LinkToLinks l2ls;
	@Inject
	private EventsManager eventManager;
	@Inject
	private TravelTimeCalculator ttCalculator;
	
	@Inject
	private @Named("fileLoc") String fileLoc;
	private @Named("keyPrefix") String keyPrefix;
	private @Named("keyFileloc") String keyFileloc;
	
	private ArrayList<Data> dataset=new ArrayList<>();
	private INDArray X;
	
	@Inject
	public DNLDataCollectionControlerListener(LinkToLinks l2ls) {
		this.l2ls=l2ls;
		this.N=l2ls.getL2lCounter();
		this.T=l2ls.getTimeBean().size();
	}
	
	@Override
	public void notifyAfterMobsim(AfterMobsimEvent event) {
		//this.dataset.add(new Tuple<>(this.X,this.TTRecorder.getTTMAP()));
		//this.TTRecorder.reset();
		INDArray Y=Nd4j.create(X.shape());
		IntStream.rangeClosed(0,N-1).parallel().forEach((n)->
		{
			IntStream.rangeClosed(0,T-1).parallel().forEach((t)->{
				LinkToLink l2l=this.l2ls.getLinkToLinks().get(n);
				Tuple<Double,Double> timeBean=l2ls.getTimeBean().get(l2ls.getNumToTimeBean().get(t));
				double tt=this.ttCalculator.getLinkToLinkTravelTimes().getLinkToLinkTravelTime(l2l.getFromLink(), l2l.getToLink(), (timeBean.getFirst()+timeBean.getSecond())*.5);
				if(tt==Double.NaN) {
					System.out.println();
				}
				Y.putScalar(n,t,tt);
			});
		});
		this.dataset.add(new Data(this.X,Y,this.keyPrefix+"_"+event.getIteration()));
	}

	@Override
	public void notifyBeforeMobsim(BeforeMobsimEvent event) {
		this.X=Nd4j.create(N,T);
		Population population=event.getServices().getScenario().getPopulation();
		Map<String,Double> linkToLinksDemand=new ConcurrentHashMap<>();
		population.getPersons().entrySet().forEach((e)->{
			Plan plan=e.getValue().getSelectedPlan();
			for(PlanElement pl:plan.getPlanElements()) {
				Leg l;
				ArrayList<Id<Link>> links=new ArrayList<>();
				
				if(pl instanceof Leg) {
					l=(Leg)pl;
					String[] part=l.getRoute().getRouteDescription().split(" ");
					for(String s:part) {
						links.add(Id.createLinkId(s.trim()));
					}
					double time=l.getDepartureTime();
					for(int i=1;i<links.size();i++) {
						Id<LinkToLink> l2lId=Id.create(links.get(i-1)+"_"+links.get(i), LinkToLink.class);
						int n=this.l2ls.getNumToLinkToLink().inverse().get(l2lId);
						int t=this.TTRecorder.getTimeId(time);
						String key=Integer.toString(n)+"_"+Integer.toString(t);
						if(linkToLinksDemand.containsKey(key)) {
							linkToLinksDemand.put(key, linkToLinksDemand.get(key)+1);
						}else {
							linkToLinksDemand.put(key,1.);
						}
						
						time+=this.l2ls.getLinkToLink(l2lId).getFreeFlowTT();
					}
				}else {
					continue;
				}
			}
		});
		
		IntStream.rangeClosed(0,N-1).parallel().forEach((n)->
		{
			IntStream.rangeClosed(0,T-1).parallel().forEach((t)->{
				String key=Integer.toString(n)+"_"+Integer.toString(t);
				if(linkToLinksDemand.containsKey(key)) {
					if(linkToLinksDemand.get(key)==Double.NaN) {
						System.out.println();
					}
					X.putScalar(n,t,linkToLinksDemand.get(key));
				}else {
					X.putScalar(n,t,0);
				}
			});
		});
	}

	@Override
	public void notifyShutdown(ShutdownEvent event) {
		DataIO.writeData(this.dataset, fileLoc, keyFileloc);
	}

	@Override
	public void notifyStartup(StartupEvent event) {
		//eventManager.addHandler(this.TTRecorder);
	}
	

}

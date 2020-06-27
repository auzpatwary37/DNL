package matsimIntegration;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
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
	@Inject
	private @Named("routeDemandFileloc") String routeFileLoc;
	@Inject
	private @Named("keyPrefix") String keyPrefix;
	@Inject
	private @Named("keyFileloc") String keyFileloc;
	
	private ArrayList<Data> dataset=new ArrayList<>();
	private INDArray X;
	@Inject
	private @Named("instantenious") boolean instantenious; 
	private Map<String,List<Integer>> routes = new HashMap<>();// save the routes in here
	private Map<Integer,Map<String,Double>> routeDemand  = new HashMap<>();//save the route demand here
	
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
				double tt=this.ttCalculator.getLinkToLinkTravelTimes().getLinkToLinkTravelTime(l2l.getFromLink(), l2l.getToLink(), (timeBean.getFirst()+timeBean.getSecond())*.5, null, null);
				if(tt==Double.NaN) {
					System.out.println();
				}
				Y.putScalar(n,t,tt);
			});
		});
		if(instantenious==true) {
			this.dataset.add(new Data(Nd4j.create(TTRecorder.getNumVehicle().toFloatMatrix()),Y,this.keyPrefix+"_"+event.getIteration()));
		}else {
			this.dataset.add(new Data(this.X,Y,this.keyPrefix+"_"+event.getIteration()));
		}
	}

	@Override
	public void notifyBeforeMobsim(BeforeMobsimEvent event) {
		this.X=Nd4j.create(N,T);
		this.routeDemand.clear();
		Population population=event.getServices().getScenario().getPopulation();
		Map<String,Double> linkToLinksDemand=new ConcurrentHashMap<>();
		population.getPersons().entrySet().forEach((e)->{
			Plan plan=e.getValue().getSelectedPlan();
			for(PlanElement pl:plan.getPlanElements()) {
				Leg l;
				ArrayList<Id<Link>> links=new ArrayList<>();
				
				if(pl instanceof Leg) {
					l=(Leg)pl;
					boolean newRoute = false;
					String routeKey = l.getRoute().getRouteDescription();
					if(!this.routes.containsKey(routeKey)) {
						this.routes.put(routeKey, new ArrayList<>());
						newRoute = true;
					}
					String[] part=l.getRoute().getRouteDescription().split(" ");
					for(String s:part) {
						links.add(Id.createLinkId(s.trim()));
					}
					double time=l.getDepartureTime().seconds();
					for(int i=1;i<links.size();i++) {
						Id<LinkToLink> l2lId=Id.create(links.get(i-1)+"_"+links.get(i), LinkToLink.class);
						int n=this.l2ls.getNumToLinkToLink().inverse().get(l2lId);
						int t=this.TTRecorder.getTimeId(time);
						if(i==1) {
							if(!this.routeDemand.containsKey(t))this.routeDemand.put(t, new HashMap<>());
							this.routeDemand.get(t).compute(routeKey, (k,v)->v==null?1:v+1);
						}
						if(newRoute)this.routes.get(routeKey).add(n);
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
		try {
			FileWriter fw = new FileWriter(new File(routeFileLoc),true);
			for(Entry<Integer, Map<String, Double>> demand:this.routeDemand.entrySet()) {
				for(Entry<String, Double> routeDemand:demand.getValue().entrySet()) {
 					fw.append(this.keyPrefix+"_"+event.getIteration()+","+routeDemand.getKey()+","+demand.getKey()+","+routeDemand.getValue());
					for(Integer i:this.routes.get(routeDemand.getKey())) {
						fw.append(","+i);
					}
					fw.append("\n");
				}
			}
			fw.flush();
			fw.close();
		} catch (IOException e1) {
			// TODO Auto-generated catch block
			e1.printStackTrace();
		}
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

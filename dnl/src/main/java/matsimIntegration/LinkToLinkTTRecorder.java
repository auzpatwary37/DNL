package matsimIntegration;

import java.util.Map;
import java.util.Map.Entry;
import java.util.concurrent.ConcurrentHashMap;

import org.matsim.api.core.v01.Id;
import org.matsim.api.core.v01.events.Event;
import org.matsim.api.core.v01.events.LinkEnterEvent;
import org.matsim.api.core.v01.events.LinkLeaveEvent;
import org.matsim.api.core.v01.events.VehicleEntersTrafficEvent;
import org.matsim.api.core.v01.events.VehicleLeavesTrafficEvent;
import org.matsim.api.core.v01.events.handler.LinkEnterEventHandler;
import org.matsim.api.core.v01.events.handler.LinkLeaveEventHandler;
import org.matsim.api.core.v01.events.handler.VehicleEntersTrafficEventHandler;
import org.matsim.api.core.v01.events.handler.VehicleLeavesTrafficEventHandler;
import org.matsim.api.core.v01.network.Link;
import org.matsim.api.core.v01.population.Population;
import org.matsim.api.core.v01.population.Route;
import org.matsim.core.api.experimental.events.VehicleArrivesAtFacilityEvent;
import org.matsim.core.api.experimental.events.handler.VehicleArrivesAtFacilityEventHandler;
import org.matsim.core.utils.collections.Tuple;
import org.matsim.vehicles.Vehicle;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import com.google.inject.Inject;

import linktolinkBPR.LinkToLink;
import linktolinkBPR.LinkToLinks;

public class LinkToLinkTTRecorder implements LinkEnterEventHandler,LinkLeaveEventHandler,VehicleLeavesTrafficEventHandler,VehicleEntersTrafficEventHandler{
	
//	@Inject
	private LinkToLinks l2ls;
	private ConcurrentHashMap<String,Double> sumTT;
	private ConcurrentHashMap<String,Integer> numVehicle;
	private Map<Id<Vehicle>,LinkEnterEvent> vehicleBuffer=new ConcurrentHashMap<>();
	
	
	@Inject
	public LinkToLinkTTRecorder(LinkToLinks l2ls) {
		this.l2ls=l2ls;
		sumTT=new ConcurrentHashMap<>();
		numVehicle=new ConcurrentHashMap<>();
	}
	
	@Override
	public void handleEvent(LinkEnterEvent event) {
		//already in some link to link buffer
		if(this.vehicleBuffer.containsKey(event.getVehicleId())) {
			//create an n_t vehicle travel time for the previous link to link
			LinkEnterEvent vinfold=this.vehicleBuffer.get(event.getVehicleId());
			Id<Link> fromLinkId=vinfold.getLinkId();
			Id<Link> toLinkId=event.getLinkId();
			double intime=vinfold.getTime();
			double tt=event.getTime()-intime;
			int n=this.getL2lNoId(fromLinkId, toLinkId);
			int t=this.getTimeId(intime);
			String key=Integer.toString(n)+"_"+Integer.toString(t);
			if(!this.sumTT.containsKey(key)) {
				this.sumTT.put(key,tt);
				this.numVehicle.put(key, 1);
			}else {
				this.sumTT.put(key,tt+this.sumTT.get(key));
				this.numVehicle.put(key, this.numVehicle.get(key)+1);
			}
			
			this.vehicleBuffer.remove(event.getVehicleId());
			//Add the vehicle for next link to link buffer
			this.vehicleBuffer.put(event.getVehicleId(), event);

		}else {
			//fresh vehicle enter
			this.vehicleBuffer.put(event.getVehicleId(), event);
		}

	}

	@Override
	public void handleEvent(LinkLeaveEvent event) {
		// TODO Auto-generated method stub
		
	}
	
	public int getTimeId(double intime) {
		if(intime==0) {
			intime=1;
		}
		for(Entry<Integer,Tuple<Double,Double>> timeBean:this.l2ls.getTimeBean().entrySet()) {
			if(intime>timeBean.getValue().getFirst() && intime<=timeBean.getValue().getSecond()) {
				return this.l2ls.getNumToTimeBean().inverse().get(timeBean.getKey());
			}
		}
		return this.l2ls.getTimeBean().size()-1;
	}
	
	public int getL2lNoId(Id<Link>fromLink,Id<Link>toLink) {
		try {
			return this.l2ls.getNumToLinkToLink().inverse().get(Id.create(fromLink+"_"+toLink, LinkToLink.class));
		} catch (Exception e) {
			System.out.println(fromLink+"_"+toLink);
			
		}
		return 0;
		//return this.l2ls.getNumToLinkToLink().inverse().get(Id.create(fromLink+"_"+toLink, LinkToLink.class));
	}
	
	public INDArray getTTMAP() {
		float[][] tt=new float[this.l2ls.getLinkToLinks().size()][this.l2ls.getTimeBean().size()];
		//int[][] numVeh=new int[this.l2ls.getLinkToLinks().size()][this.l2ls.getTimeBean().size()];
		
		for(String n_t:this.sumTT.keySet()) {
			int n=Integer.parseInt(n_t.split("_")[0]);
			int t=Integer.parseInt(n_t.split("_")[1]);
			tt[n][t]=(float) (sumTT.get(n_t)/numVehicle.get(n_t));
			//numVeh[n][t]=numVehicle.get(n_t);
		}
		return Nd4j.create(tt);
	}



	@Override
	public void handleEvent(VehicleLeavesTrafficEvent event) {
		this.vehicleBuffer.remove(event.getVehicleId());
	}
	
	public void reset() {
		this.vehicleBuffer.clear();
		sumTT.clear();
		numVehicle.clear();
		
	}

	public INDArray getNumVehicle() {
		float[][] numVeh=new float[this.l2ls.getLinkToLinks().size()][this.l2ls.getTimeBean().size()];
		for(String n_t:this.sumTT.keySet()) {
			int n=Integer.parseInt(n_t.split("_")[0]);
			int t=Integer.parseInt(n_t.split("_")[1]);
			numVeh[n][t]=numVehicle.get(n_t);
		}
		return Nd4j.create(numVeh);
	}

	@Override
	public void handleEvent(VehicleEntersTrafficEvent event) {
		
		this.vehicleBuffer.put(event.getVehicleId(), new LinkEnterEvent(event.getTime(), event.getVehicleId(), event.getLinkId()));
		
	}
	

}


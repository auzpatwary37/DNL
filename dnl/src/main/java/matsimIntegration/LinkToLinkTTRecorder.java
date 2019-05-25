package matsimIntegration;

import java.util.Map;
import java.util.Map.Entry;
import java.util.concurrent.ConcurrentHashMap;

import org.matsim.api.core.v01.Id;
import org.matsim.api.core.v01.events.LinkEnterEvent;
import org.matsim.api.core.v01.events.LinkLeaveEvent;
import org.matsim.api.core.v01.events.handler.LinkEnterEventHandler;
import org.matsim.api.core.v01.events.handler.LinkLeaveEventHandler;
import org.matsim.api.core.v01.network.Link;
import org.matsim.core.utils.collections.Tuple;
import org.matsim.vehicles.Vehicle;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import com.google.inject.Inject;

import linktolinkBPR.LinkToLink;
import linktolinkBPR.LinkToLinks;

public class LinkToLinkTTRecorder implements LinkEnterEventHandler,LinkLeaveEventHandler{
	
	@Inject
	private LinkToLinks l2ls;
	private INDArray sumTT=Nd4j.create(l2ls.getL2lCounter(),l2ls.getTimeBean().size());
	private INDArray numVehicle=Nd4j.create(sumTT.shape());
	private Map<Id<Vehicle>,VehicleInfo> vehicleBuffer=new ConcurrentHashMap<>();

	
	@Inject
	public LinkToLinkTTRecorder() {
	}
	
	@Override
	public void handleEvent(LinkEnterEvent event) {
		//already in some link to link buffer
		if(this.vehicleBuffer.containsKey(event.getVehicleId())) {
			//create an n_t vehicle travel time for the previous link to link
			VehicleInfo vinfold=this.vehicleBuffer.get(event.getVehicleId());
			Id<Link> fromLinkId=vinfold.getFromLinkId();
			Id<Link> toLinkId=event.getLinkId();
			double intime=vinfold.getEnterTime();
			double tt=event.getTime()-intime;
			int n=this.getL2lNoId(fromLinkId, toLinkId);
			int t=this.getTimeId(intime);
			this.sumTT.putScalar(n,t,this.sumTT.getDouble(n,t)+tt);
			this.numVehicle.putScalar(n,t, this.numVehicle.getDouble(n,t)+1);
			this.vehicleBuffer.remove(event.getVehicleId());
			//Add the vehicle for next link to link buffer
			this.vehicleBuffer.put(event.getVehicleId(), new VehicleInfo(event.getVehicleId(),event.getLinkId(),event.getTime()));
			
		}else {
			//fresh vehicle enter
			this.vehicleBuffer.put(event.getVehicleId(), new VehicleInfo(event.getVehicleId(),event.getLinkId(),event.getTime()));
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
				return timeBean.getKey();
			}
		}
		return this.l2ls.getTimeBean().size()-1;
	}
	
	public int getL2lNoId(Id<Link>fromLink,Id<Link>toLink) {
		return this.l2ls.getNumToLinkToLink().inverse().get(Id.create(fromLink+"_"+toLink, LinkToLink.class));
	}
	
	public INDArray getTTMAP() {
		return this.sumTT.divi(numVehicle);
	}

}

class VehicleInfo{
	
	private Id<Vehicle> vehicleId;
	private Id<Link> fromLinkId;
	private double enterTime;
	
	public VehicleInfo(Id<Vehicle> vehicleId,Id<Link> LinkId,Double time) {
		this.vehicleId=vehicleId;
		this.fromLinkId=LinkId;
		this.enterTime=time;
	}

	public Id<Vehicle> getVehicleId() {
		return vehicleId;
	}

	public Id<Link> getFromLinkId() {
		return fromLinkId;
	}

	public double getEnterTime() {
		return enterTime;
	}
	
}

package matsimIntegration;

import java.util.ArrayList;
import java.util.Map;

import org.matsim.api.core.v01.Scenario;
import org.matsim.api.core.v01.network.Network;
import org.matsim.api.core.v01.population.Population;
import org.matsim.core.controler.events.AfterMobsimEvent;
import org.matsim.core.controler.events.BeforeMobsimEvent;
import org.matsim.core.controler.listener.AfterMobsimListener;
import org.matsim.core.controler.listener.BeforeMobsimListener;
import org.matsim.core.utils.collections.Tuple;
import org.nd4j.linalg.api.ndarray.INDArray;

import com.google.inject.Inject;

import linktolinkBPR.LinkToLinks;

public class DNLDataCollectionControlerListener implements BeforeMobsimListener, AfterMobsimListener{

	private final int N;
	private final int T;
	private final Map<Integer,Tuple<Double,Double>>timeBean;
	
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
	}
	
	@Override
	public void notifyAfterMobsim(AfterMobsimEvent event) {
		// TODO Auto-generated method stub
	}

	@Override
	public void notifyBeforeMobsim(BeforeMobsimEvent event) {
		Population population=event.getServices().getScenario().getPopulation();
	}

}

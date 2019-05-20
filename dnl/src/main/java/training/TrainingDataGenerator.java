package training;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Map.Entry;
import java.util.Random;

import org.matsim.api.core.v01.Coord;
import org.matsim.api.core.v01.Id;
import org.matsim.api.core.v01.Scenario;
import org.matsim.api.core.v01.network.Link;
import org.matsim.api.core.v01.network.Network;
import org.matsim.api.core.v01.network.NetworkFactory;
import org.matsim.api.core.v01.network.NetworkWriter;
import org.matsim.api.core.v01.network.Node;
import org.matsim.api.core.v01.population.Activity;
import org.matsim.api.core.v01.population.Leg;
import org.matsim.api.core.v01.population.Person;
import org.matsim.api.core.v01.population.Plan;
import org.matsim.api.core.v01.population.Population;
import org.matsim.api.core.v01.population.PopulationFactory;
import org.matsim.api.core.v01.population.PopulationWriter;
import org.matsim.core.config.Config;
import org.matsim.core.config.ConfigUtils;
import org.matsim.core.config.ConfigWriter;
import org.matsim.core.config.groups.ControlerConfigGroup.RoutingAlgorithmType;
import org.matsim.core.config.groups.PlanCalcScoreConfigGroup.ActivityParams;
import org.matsim.core.config.groups.QSimConfigGroup.LinkDynamics;
import org.matsim.core.controler.Controler;
import org.matsim.core.controler.OutputDirectoryHierarchy;
import org.matsim.core.network.NetworkUtils;
import org.matsim.core.population.PopulationUtils;
import org.matsim.core.scenario.ScenarioUtils;
import org.matsim.vehicles.Vehicle;
import org.matsim.vehicles.VehicleType;
import org.matsim.vehicles.VehicleWriterV1;
import org.matsim.vehicles.Vehicles;

public class TrainingDataGenerator {
	public static void main(String[] args) throws IOException {
		Config config =ConfigUtils.createConfig();
		GenerateSiouxFallNetwork("Network/SiouxFalls/siouxfallsNodes.csv","Network/SiouxFalls/links.csv","Network/SiouxFalls/siouxfallsNetwork.xml");
		ConfigUtils.loadConfig(config, "Network/SiouxFalls/config.xml");
		config.network().setInputFile("Network/SiouxFalls/siouxfallsNetwork.xml");
		ArrayList<String> modes=new ArrayList<>();
		modes.add("car");
		config.plansCalcRoute().setNetworkModes(modes);
		//config.travelTimeCalculator().setCalculateLinkToLinkTravelTimes(true);
		config.travelTimeCalculator().setSeparateModes(false);
		config.plansCalcRoute().setInsertingAccessEgressWalk(false);
		config.qsim().setUsePersonIdForMissingVehicleId(true);
		config.qsim().setStartTime(15*3600);
		config.qsim().setEndTime(24*3600);
		config.qsim().setLinkDynamics(LinkDynamics.PassingQ);
		config.controler().setLinkToLinkRoutingEnabled(true);
		config.controler().setLastIteration(150);
		config.global().setCoordinateSystem("arbitrary");
		config.parallelEventHandling().setNumberOfThreads(3);
		config.controler().setWritePlansInterval(50);
		config.global().setNumberOfThreads(3);
		config.strategy().setFractionOfIterationsToDisableInnovation(0.8);
		config.controler().setWriteEventsInterval(50);
		config.strategy().addParam("ModuleProbability_1", "0.8");
		config.strategy().addParam("Module_1", "ChangeExpBeta");
		config.strategy().addParam("ModuleProbability_2", "0.2");
		config.strategy().addParam("Module_2", "ReRoute");
//		config.strategy().addParam("ModuleProbability_2", "0.1");
//		config.strategy().addParam("Module_2", "TimeAllocationMutator");
		config.controler().setRoutingAlgorithmType(RoutingAlgorithmType.AStarLandmarks);
//		config.planCalcScore().getOrCreateModeParams("car").setMarginalUtilityOfTraveling(-200);
//		config.planCalcScore().setPerforming_utils_hr(100);
		//config.qsim().setFlowCapFactor(0.5);

		
		for(int i=0;i<50;i++) {
			GenerateRandomPopulation(config,"Network/SiouxFalls/SiouxFallDemand.csv", 1, "Network/SiouxFalls",0.4);
			config.plans().setInputFile("Network/SiouxFalls/population.xml");
			config.vehicles().setVehiclesFile("Network/SiouxFalls/vehicles.xml");
			config.controler().setOutputDirectory("Network/SiouxFalls/output"+i);
			new ConfigWriter(config).write("Network/SiouxFalls/final_config.xml");
			Scenario scenario = ScenarioUtils.loadScenario(config);
			Controler controler = new Controler(scenario);
			//controler.addOverridingModule(new TravelTimeCalculatorModule());
			controler.getConfig().controler().setOverwriteFileSetting(OutputDirectoryHierarchy.OverwriteFileSetting.overwriteExistingFiles);
			controler.run();
		}
		
		
		
	}
	
	public static void GenerateSiouxFallNetwork(String nodefile,String linkFile,String netFileWriteLoc) throws IOException{
		Network network=NetworkUtils.createNetwork();
		NetworkFactory netfac=network.getFactory();
		BufferedReader bf=new BufferedReader(new FileReader(new File(nodefile)));
		bf.readLine();//get rid of the header
		String line=null;
		while((line=bf.readLine())!=null) {
			String[] part=line.split(",");
			String nodeId=part[0].trim();
			Coord coord=new Coord(Double.parseDouble(part[1].trim()),Double.parseDouble(part[2].trim()));
			network.addNode(netfac.createNode(Id.createNodeId(nodeId), coord));
		}
		bf.close();
		
		
		
		bf=new BufferedReader(new FileReader(new File(linkFile)));
		bf.readLine();//get rid of header
		line=null;
		while((line=bf.readLine())!=null) {
			String[] part=line.split(",");
			Id<Node> startNode=Id.createNodeId(part[0].trim());
			Id<Node> endNode=Id.createNodeId(part[1].trim());
			Id<Link> linkId=Id.createLinkId(startNode.toString()+"_"+endNode.toString());
			Link link=netfac.createLink(linkId, network.getNodes().get(startNode), network.getNodes().get(endNode));
			
			link.setNumberOfLanes(Double.parseDouble(part[3].trim()));
			link.setCapacity(Double.parseDouble(part[2].trim())*link.getNumberOfLanes());
			link.setLength(Double.parseDouble(part[4].trim())*1000);
			link.setFreespeed(Double.parseDouble(part[5].trim())*1000/3600);
			
			network.addLink(link);
		}
		bf.close();
		
		
		int[] O={1,2,3,13};
		for(Integer o:O) {
			Node node=network.getNodes().get(Id.createNodeId(o.toString()));
			Coord coord=new Coord(node.getCoord().getX()+100,node.getCoord().getY()+100);
			Node Onode=netfac.createNode(Id.createNodeId("O"+o.toString()), coord);
			network.addNode(Onode);
			Link oLink=netfac.createLink(Id.createLinkId(Onode.getId().toString()+"_"+node.getId().toString()), Onode, node);
			oLink.setCapacity(6000);
			oLink.setFreespeed(100);
			network.addLink(oLink);
		}
		
		int[] D={6,7,18,20};
		for(Integer d:D) {
			Node node=network.getNodes().get(Id.createNodeId(d.toString()));
			Coord coord=new Coord(node.getCoord().getX()+100,node.getCoord().getY()+100);
			Node Dnode=netfac.createNode(Id.createNodeId(d.toString()+"D"), coord);
			network.addNode(Dnode);
			Link dLink=netfac.createLink(Id.createLinkId(node.getId().toString()+"_"+Dnode.getId().toString()), node,Dnode);
			dLink.setCapacity(6000);
			dLink.setFreespeed(100);
			network.addLink(dLink);
		}
		new NetworkWriter(network).write(netFileWriteLoc);
	}
	
	public static void GenerateRandomPopulation(Config config,String demandFileLocaiton,double sdPercent,String writeLoc,double demandPercent) throws IOException {
		Population p=PopulationUtils.createPopulation(config);
		Vehicles vehicles=ScenarioUtils.loadScenario(config).getVehicles();
		
		VehicleType vt=vehicles.getFactory().createVehicleType(Id.create("car", VehicleType.class));
		vehicles.addVehicleType(vt);
		
		PopulationFactory popfac=p.getFactory();
		BufferedReader bf=new BufferedReader(new FileReader(new File(demandFileLocaiton)));
		String[] header=bf.readLine().split(",");//get rid of the header
		String line=null;
		while((line=bf.readLine())!=null) {
			HashMap<Integer,Double> demands=new HashMap<>();
			String[] part=line.split(",");
			Id<Node> originNodeId=Id.createNodeId(part[0].trim());
			Id<Node> destinationNodeId=Id.createNodeId(part[1].trim());
			int i=-1;
			for(String s:header) {
				i++;
				if(s.equalsIgnoreCase("O")||s.equalsIgnoreCase("D")){
					continue;
				}
				demands.put(Integer.parseInt(s),Double.parseDouble(part[i]));
			}
			Random random=new Random();
			for(Entry<Integer,Double> demand:demands.entrySet()) {
				double hour=demand.getKey()+12;
				double randomDemand=demand.getValue()*demandPercent+random.nextGaussian()*sdPercent/100*demand.getValue()*demandPercent;
				for(int j=0;j<=randomDemand;j++) {
					Person person =popfac.createPerson(Id.createPersonId(originNodeId.toString()+"_"+destinationNodeId.toString()+"_"+hour+"_"+j));
					Plan plan=popfac.createPlan();
					Activity act1=popfac.createActivityFromLinkId("Home1", Id.createLinkId("O"+originNodeId.toString()+"_"+originNodeId.toString()));
					double tripStartTime=hour*3600+30*60+random.nextGaussian()*30*60;
					act1.setEndTime(tripStartTime);
					Leg leg=popfac.createLeg("car");
					leg.setDepartureTime(tripStartTime);
					Activity act2=popfac.createActivityFromLinkId("Home2", Id.createLinkId(destinationNodeId.toString()+"_"+destinationNodeId.toString()+"D"));
					plan.addActivity(act1);
					plan.addLeg(leg);
					plan.addActivity(act2);
					person.addPlan(plan);
					p.addPerson(person);
					Vehicle v=vehicles.getFactory().createVehicle(Id.createVehicleId(person.getId().toString()), vt);
					vehicles.addVehicle(v);
				}
			}
		}
		ActivityParams act1 = new ActivityParams("Home1");
		act1.setTypicalDuration(8*60*60);
		config.planCalcScore().addActivityParams(act1);
		ActivityParams act2 = new ActivityParams("Home2");
		act2.setTypicalDuration(8*60*60);
		config.planCalcScore().addActivityParams(act2);
		
		new PopulationWriter(p).write(writeLoc+"/population.xml");
		new VehicleWriterV1(vehicles).writeFile(writeLoc+"/vehicles.xml");
		new ConfigWriter(config).write(writeLoc+"/config.xml");
	}
	
	
	
}
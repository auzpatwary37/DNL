package training;

import static org.junit.jupiter.api.Assertions.assertAll;

import java.io.IOException;
import java.util.HashMap;
import java.util.Map;
import java.util.Random;

import javax.xml.parsers.ParserConfigurationException;
import javax.xml.parsers.SAXParser;
import javax.xml.parsers.SAXParserFactory;


import org.matsim.api.core.v01.Id;
import org.matsim.api.core.v01.Scenario;
import org.matsim.api.core.v01.population.Activity;
import org.matsim.api.core.v01.population.Leg;
import org.matsim.api.core.v01.population.Person;
import org.matsim.api.core.v01.population.Plan;
import org.matsim.api.core.v01.population.Population;
import org.matsim.api.core.v01.population.PopulationFactory;
import org.matsim.contrib.signals.builder.Signals;
import org.matsim.contrib.signals.data.SignalsData;
import org.matsim.contrib.signals.data.SignalsDataLoader;
import org.matsim.core.config.Config;
import org.matsim.core.config.ConfigGroup;
import org.matsim.core.config.ConfigUtils;
import org.matsim.core.config.ConfigWriter;
import org.matsim.core.controler.Controler;
import org.matsim.core.controler.OutputDirectoryHierarchy;
import org.matsim.core.population.PopulationUtils;
import org.matsim.core.scenario.ScenarioUtils;
import org.matsim.core.utils.collections.Tuple;
import org.matsim.utils.objectattributes.ObjectAttributes;
import org.matsim.utils.objectattributes.ObjectAttributesXmlReader;
import org.matsim.vehicles.Vehicle;
import org.matsim.vehicles.VehicleUtils;
import org.matsim.vehicles.Vehicles;
import org.xml.sax.SAXException;

import dynamicTransitRouter.DynamicRoutingModule;
import dynamicTransitRouter.TransitRouterFareDynamicImpl;
import dynamicTransitRouter.fareCalculators.ZonalFareXMLParserV2;
import linktolinkBPR.LinkToLinks;
import linktolinkBPR.SignalFlowReductionGenerator;
import matsimIntegration.DNLDataCollectionModule;
import populationGeneration.SubPopulationTry;


public class TraninigDataGeneratorHKI {
	public static void main(String[] args) {
		Config initialConfig = RunTCS.setupConfig();
		String baseFileLoc = "Network/HKI/";
		initialConfig.plans().setInputPersonAttributeFile(baseFileLoc+"data/personAttributesHKI.xml");
		
		initialConfig.global().setInsistingOnDeprecatedConfigVersion(true);
		
		ObjectAttributes att = new ObjectAttributes();
		new ObjectAttributesXmlReader(att).readFile(baseFileLoc+"data/personAttributesHKI.xml");
		
		
		//_______________________Select the population network and vehicles file_____________________________
		initialConfig.plans().setInputFile(baseFileLoc+"cal/output_plans.xml.gz");
		initialConfig.network().setInputFile(baseFileLoc+"cal/output_network.xml.gz");
		initialConfig.vehicles().setVehiclesFile(baseFileLoc+"data/VehiclesHKI.xml");
		//________________________________________________________________________________________________
		
		initialConfig.network().setLaneDefinitionsFile(baseFileLoc+"cal/output_lanes.xml");
		ConfigGroup sscg = initialConfig.getModule("signalsystems");
		sscg.addParam("signalcontrol", baseFileLoc+"cal/output_signal_control_v2.0.xml");
		sscg.addParam("signalgroups", baseFileLoc+"cal/output_signal_groups_v2.0.xml");
		sscg.addParam("signalsystems", baseFileLoc+"cal/output_signal_systems_v2.0.xml");
		initialConfig.transit().setTransitScheduleFile(baseFileLoc+"cal/output_transitSchedule.xml.gz");
		initialConfig.transit().setVehiclesFile(baseFileLoc+"cal/output_transitVehicles.xml.gz");
		//initialConfig.plans().setHandlingOfPlansWithoutRoutingMode(HandlingOfMainModeIdentifier);
		
		initialConfig.removeModule("roadpricing");
		initialConfig.qsim().setUsePersonIdForMissingVehicleId(true);
		
		initialConfig.strategy().setFractionOfIterationsToDisableInnovation(0.85);

		Config config = initialConfig;
		config.controler().setLastIteration(6);
		config.controler().setOutputDirectory(baseFileLoc+"output");
		config.transit().setUseTransit(true);
		config.plansCalcRoute().setInsertingAccessEgressWalk(false);
		config.qsim().setUsePersonIdForMissingVehicleId(true);
		//config.controler().setLastIteration(50);
		config.parallelEventHandling().setNumberOfThreads(7);
		config.controler().setWritePlansInterval(2);
		config.qsim().setStartTime(0.0);
		config.qsim().setEndTime(28*3600);
		
		config.controler().setWriteEventsInterval(20);
		config.planCalcScore().setWriteExperiencedPlans(false);
		//config.plans().setInsistingOnUsingDeprecatedPersonAttributeFile(true);
		TransitRouterFareDynamicImpl.aStarSetting='c';
		TransitRouterFareDynamicImpl.distanceFactor=.01;
		
		//config.removeModule("signalsystems");
		
//		ConfigGeneratorLargeToy.reducePTCapacity(scenario.getTransitVehicles(),.15);
//		ConfigGeneratorLargeToy.reduceLinkCapacity(scenario.getNetwork(),.15);
//		StrategySettings stratSets = new StrategySettings();
		
		
//		config.qsim().setStorageCapFactor(.14);
		
		
		new ConfigWriter(config).write(baseFileLoc + "baseConfig.xml");
		
		new ConfigWriter(config).write("test/config.xml");
		Scenario scenario = ScenarioUtils.loadScenario(config);
		SignalFlowReductionGenerator sg = new SignalFlowReductionGenerator(scenario);
		Map<Integer,Tuple<Double,Double>> timeBean=new HashMap<>();
		for(int i=0;i<28;i++) {
			timeBean.put(i,new Tuple<Double,Double>(i*3600.,i*3600.+3600));
		}
		LinkToLinks l2ls=new LinkToLinks(scenario.getNetwork(),timeBean,3,3,sg);
		double[] ratio=new double[] {0.8,.9,1,1.1};
		
		//__________________Capacity Tweaking______________________________________-
//		config.qsim().setFlowCapFactor(1);
//		config.qsim().setStorageCapFactor(1);
//		for(LanesToLinkAssignment l2l:scenario.getLanes().getLanesToLinkAssignments().values()) {
//			for(Lane l: l2l.getLanes().values()) {
//				
//				//Why this is done? 
//				//Why .1 specifically??
//				l.setCapacityVehiclesPerHour(1800*.15);
//			}
//		}
		//__________________________________________________________________________________
		
		
		
//		for(Person p:scenario.getPopulation().getPersons().values()) {
//			VehicleUtils.insertVehicleIdIntoAttributes(p, "car", Id.createVehicleId(p.getId().toString()));
//		}
		RandomPopulationSampler popSampler = new RandomPopulationSampler(scenario.getPopulation(),scenario.getVehicles(),config);
		boolean newPop = false;
		int k=0;
		for(int i=0;i<ratio.length;i++) {
			Config configCurrent = ConfigUtils.createConfig();
			ConfigUtils.loadConfig(configCurrent,baseFileLoc+"baseConfig.xml");
			PopulationGenerationOutput output = null;
			if(newPop) {
			try {
				output = SubPopulationTry.generatePopulation(ratio[k], ratio[k], true);
			} catch (IOException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
			}else {
				output = popSampler.randomSample(ratio[k]);
			}
			output.writeFiles(baseFileLoc+"population.xml",baseFileLoc+"Vehicles.xml",baseFileLoc+"config.xml");
			
			
			
			configCurrent.vehicles().setVehiclesFile(baseFileLoc+"Vehicles.xml");
			configCurrent.plans().setInputFile(baseFileLoc+"population.xml");
			configCurrent.controler().setOutputDirectory(baseFileLoc+"output"+i);
			//configCurrent.controler().setWritePlansInterval(1);
			//configCurrent.controler().setWriteEventsInterval(1);
			configCurrent.travelTimeCalculator().setCalculateLinkToLinkTravelTimes(true);
			configCurrent.travelTimeCalculator().setTraveltimeBinSize(3600);
			configCurrent.travelTimeCalculator().setSeparateModes(false);
			configCurrent.controler().setLastIteration(400);
			//TravelTimeCalculator.Builder b;
			Scenario scenarioCurrent = ScenarioUtils.loadScenario(configCurrent);
			scenarioCurrent.addScenarioElement(SignalsData.ELEMENT_NAME, new SignalsDataLoader(config).loadSignalsData());	
			Controler controler = new Controler(scenarioCurrent);
			
			ZonalFareXMLParserV2 busFareGetter = new ZonalFareXMLParserV2(scenario.getTransitSchedule());
			SAXParser saxParser;
			
			try {
				saxParser = SAXParserFactory.newInstance().newSAXParser();
				saxParser.parse(baseFileLoc+"data/busFare.xml", busFareGetter);
				controler.addOverridingModule(new DynamicRoutingModule(busFareGetter.get(), "fare/mtr_lines_fares.csv", 
						"fare/transitDiscount.json", "fare/light_rail_fares.csv", "fare/busFareGTFS.json", "fare/ferryFareGTFS.json"));
				
//				controler.addOverridingModule(new DynamicRoutingModule(busFareGetter.get(), "fare/mtr_lines_fares.csv", 
//						"fare/GMB.csv", "fare/light_rail_fares.csv"));
				
			} catch (ParserConfigurationException e1) {
				// TODO Auto-generated catch block
				e1.printStackTrace();
			} catch (SAXException e1) {
				// TODO Auto-generated catch block
				e1.printStackTrace();
			} catch (IOException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
			Signals.configure(controler);
			controler.addOverridingModule(new DNLDataCollectionModule(l2ls,baseFileLoc+"DataSet"+i+".txt",Double.toString(ratio[k]),baseFileLoc+"KeySet"+i+".csv", false));
			controler.getConfig().controler().setOverwriteFileSetting(OutputDirectoryHierarchy.OverwriteFileSetting.overwriteExistingFiles);
			controler.run();
			if((i+1)%1==0) {
				k=k+1;
			}
		}
		 
	}
	
	
	
}


class RandomPopulationSampler{
	private final Population population;
	private final Vehicles vehicles;
	private final Config config;
	RandomPopulationSampler(Population population, Vehicles vehicles, Config config){
		this.population = population;
		this.vehicles = vehicles;
		this.config = config;
	}
	/**
	 * choose entry of the given map randomly.
	 * @param originalMap
	 * @param percentage should be less than 1 and non-negative
	 * currently work for underSampling only
	 * @return
	 */
	public PopulationGenerationOutput randomSample(double percentage){
		Population newPop = PopulationUtils.createPopulation(config);
		Vehicles vehicles = VehicleUtils.createVehiclesContainer();
		PopulationFactory popFac =  PopulationUtils.getFactory();
		int p = (int)percentage;
		if(percentage>1) {
			
			for(int i=0; i<p; i++) {
				final int j = i;
				population.getPersons().entrySet().forEach(e->{
					Person person  =  popFac.createPerson(Id.create(e.getKey().toString()+"_"+j,Person.class));
					e.getValue().getAttributes().getAsMap().entrySet().forEach(a->person.getAttributes().putAttribute(a.getKey(), a.getValue()));//We have not deepcloned the attributes. Should be okay
					e.getValue().getPlans().forEach(a->{
						Plan newPlan = popFac.createPlan();
						PopulationUtils.copyFromTo(a, newPlan);
						person.addPlan(newPlan);
					});
					newPop.addPerson(person);
					Vehicle oldVehicle = this.vehicles.getVehicles().get(Id.createVehicleId(e.getKey().toString()));
					Vehicle newVehicle = vehicles.getFactory().createVehicle(Id.createVehicleId(person.getId().toString()), oldVehicle.getType());
					vehicles.addVehicle(newVehicle);
				});
			}
		}
		Random random = new Random();
		population.getPersons().entrySet().forEach(e->{
			if(random.nextDouble()<percentage-p) {
				Person person  =  popFac.createPerson(Id.create(e.getKey().toString()+"_"+percentage,Person.class));
				e.getValue().getAttributes().getAsMap().entrySet().forEach(a->person.getAttributes().putAttribute(a.getKey(), a.getValue()));//We have not deepcloned the attributes. Should be okay
				e.getValue().getPlans().forEach(a->{
					Plan newPlan = popFac.createPlan();
					PopulationUtils.copyFromTo(a, newPlan);
					person.addPlan(newPlan);
				});
				newPop.addPerson(person);
				Vehicle oldVehicle = this.vehicles.getVehicles().get(Id.createVehicleId(e.getKey().toString()));
				Vehicle newVehicle = vehicles.getFactory().createVehicle(Id.createVehicleId(person.getId().toString()), oldVehicle.getType());
				vehicles.addVehicle(newVehicle);
			}
		});
		return new PopulationGenerationOutput(newPop,vehicles,this.config);
	}
}
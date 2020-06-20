package training;

import org.matsim.core.config.Config;
import org.matsim.core.config.ConfigGroup;

import dynamicTransitRouter.TransitRouterFareDynamicImpl;



public class RunUtils {

	public static Config provideConfig(String baseFileLoc,String writeFileLoc) {
		//Measurements calibrationMeasurements=new MeasurementsReader().readMeasurements("data\\toyScenarioLargeData\\originalMeasurements_20_11.xml");
		//calibrationMeasurements.applyFator(.1);
//		Config initialConfig=ConfigUtils.createConfig();
//		ConfigUtils.loadConfig(initialConfig, "data/toyScenarioLargeData/configToyLargeMod.xml");
		
		Config initialConfig = RunTCS.setupConfig();
		//initialConfig.plans().setInsistingOnUsingDeprecatedPersonAttributeFile(true);
		initialConfig.plans().setInputFile(baseFileLoc+"/data/populationHKI.xml");
		initialConfig.network().setInputFile(baseFileLoc+"/cal/output_network.xml.gz");
		initialConfig.vehicles().setVehiclesFile(baseFileLoc+"/data/VehiclesHKI.xml");
		
		initialConfig.network().setLaneDefinitionsFile(baseFileLoc+"/cal/output_lanes.xml");
		ConfigGroup sscg = initialConfig.getModule("signalsystems");
		sscg.addParam("signalcontrol", baseFileLoc+"/cal/output_signal_control_v2.0.xml");
		sscg.addParam("signalgroups", baseFileLoc+"/cal/output_signal_groups_v2.0.xml");
		sscg.addParam("signalsystems", baseFileLoc+"/cal/output_signal_systems_v2.0.xml");
		initialConfig.transit().setTransitScheduleFile(baseFileLoc+"/cal/output_transitSchedule.xml.gz");
		initialConfig.transit().setVehiclesFile(baseFileLoc+"/cal/output_transitVehicles.xml.gz");
		//initialConfig.plans().setHandlingOfPlansWithoutRoutingMode(HandlingOfMainModeIdentifier);
		
		initialConfig.removeModule("roadpricing");
		initialConfig.qsim().setUsePersonIdForMissingVehicleId(true);
		
		
		//VehicleUtils.insertVehicleIdIntoAttributes(person, mode, vehicleId);
		initialConfig.strategy().setFractionOfIterationsToDisableInnovation(0.85);
//		initialConfig.qsim().setFlowCapFactor(0.14);
//		initialConfig.qsim().setStorageCapFactor(0.2);

		
//		LinkedHashMap<String,Double>initialParams=loadInitialParam(pReader,new double[] {-200,-240});
//		LinkedHashMap<String,Double>params=initialParams;
//		pReader.setInitialParam(initialParams);
//		Calibrator calibrator;

		
		//SimRun simRun=new SimRunImplToyLarge(100);
		Config config = initialConfig;
		config.controler().setLastIteration(50);
		config.controler().setOutputDirectory(writeFileLoc);
		config.transit().setUseTransit(true);
		config.plansCalcRoute().setInsertingAccessEgressWalk(false);
		config.qsim().setUsePersonIdForMissingVehicleId(true);
		//config.controler().setLastIteration(50);
		config.parallelEventHandling().setNumberOfThreads(7);
		config.controler().setWritePlansInterval(50);
		config.qsim().setStartTime(0.0);
		config.qsim().setEndTime(28*3600);
		config.qsim().setStorageCapFactor(.14);
		config.controler().setWriteEventsInterval(20);
		config.planCalcScore().setWriteExperiencedPlans(false);
		//config.plans().setInsistingOnUsingDeprecatedPersonAttributeFile(true);
		TransitRouterFareDynamicImpl.aStarSetting='c';
		TransitRouterFareDynamicImpl.distanceFactor=.01;
		 return config;
	}
	
}

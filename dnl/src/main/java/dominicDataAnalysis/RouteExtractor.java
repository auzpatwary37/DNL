package dominicDataAnalysis;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.concurrent.ConcurrentHashMap;

import org.matsim.api.core.v01.Id;
import org.matsim.api.core.v01.population.Leg;
import org.matsim.api.core.v01.population.Person;
import org.matsim.api.core.v01.population.PlanElement;
import org.matsim.api.core.v01.population.Population;
import org.matsim.api.core.v01.population.Route;
import org.matsim.core.config.ConfigUtils;
import org.matsim.core.population.PopulationUtils;
import org.matsim.core.utils.collections.Tuple;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import com.google.common.collect.BiMap;
import com.google.common.collect.HashBiMap;


public class RouteExtractor {
	
	public static void main(String[] args) {
		List<Population> populations=new ArrayList<>();
		Map<String,routeDemandAndTTInfo> demandAndTTInfo=new HashMap<>();
		
		int[] names=new int[] {1,5,10,50,100};
		int[] iters=new int[] {200,500,500,500,500};
		String[] popNames=new String[5];
		for(int i=0;i<5;i++) {
			Population population=PopulationUtils.createPopulation(ConfigUtils.createConfig());
			PopulationUtils.readPopulation(population, "populations/coarse_"+names[i]+"_"+iters[i]+"it.xml.gz");
			popNames[i]="coarse_"+names[i]+"_"+iters[i]+"it";
			populations.add(population);
		}
		Map<Id<Route>,Route> routes=extractCarRoutes(populations);
		BiMap<Integer,Id<Route>> numToRoute=HashBiMap.create();
		int i=0;
		for(Id<Route>routeId:routes.keySet()) {
			numToRoute.put(i, routeId);
			i++;
		}
		Map<Integer,Tuple<Double,Double>>timeBeans=new HashMap<>();
		i=0;		
		for(double j=0;j<24;j=j+.5) {
			timeBeans.put(i, new Tuple<>(j*3600,j+0.5*3600));
			i++;
		}
		
		writeRouteMetaData(routes,numToRoute,"populations/");
		
		BiMap<Integer,Integer> numToTimeBean=HashBiMap.create();
		i=0;
		for(Integer timeId:timeBeans.keySet()) {
			numToTimeBean.put(i, timeId);
			i++;
		}
		
		writeTimeIdMetaData(timeBeans,numToTimeBean,"populations/");
		i=0;
		for(Population population:populations) {
			routeDemandAndTTInfo info=new routeDemandAndTTInfo(routes,timeBeans,population,numToRoute,numToTimeBean);
			info.setPopulationDetails(popNames[i]);
			demandAndTTInfo.put(popNames[i],info);
			info.writeInfo("populations/");
			i++;
		}
		
	}
	
	public static Map<Id<Route>,Route> extractCarRoutes(List<Population> populations){
		Map<Id<Route>,Route> routes =new ConcurrentHashMap<>();
		for(Population population:populations) {
			population.getPersons().entrySet().parallelStream().forEach((p)->{
				for(PlanElement pe:p.getValue().getSelectedPlan().getPlanElements()) {
					if(pe instanceof Leg) {
						Leg leg=(Leg)pe;
						Id<Route> routeid= Id.create(leg.getRoute().getRouteDescription(),Route.class);
						if(!routes.containsKey(routeid)) {
							routes.put(routeid,leg.getRoute());
						}
					}
				}
			});
		}
		return routes;
	}
	public static void writeRouteMetaData( Map<Id<Route>,Route> routes,BiMap<Integer,Id<Route>> numToRoute, String fileLoc) {
		try {
			FileWriter fw= new FileWriter(new File(fileLoc+"_RoutesMetaData.csv"));
			fw.append("RouteNumber,RouteDescription,NumberOfLinks\n");//header
			for(Entry<Integer, Id<Route>> routeno:numToRoute.entrySet()) {
				fw.append(routeno.getKey()+","+routeno.getValue().toString()+","+routes.get(routeno.getValue()).getRouteDescription().split(" ").length+"\n");
			}
			fw.flush();
			fw.close();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}
	
	public static void writeTimeIdMetaData( Map<Integer,Tuple<Double,Double>> timeBean,BiMap<Integer,Integer> numTotimeBean, String fileLoc) {
		try {
			FileWriter fw= new FileWriter(new File(fileLoc+"_timeBeanMetaData.csv"));
			fw.append("timeBeanNoCode,timeBeanId,fromTime,ToTime\n");//header
			for(Entry<Integer, Integer> timeBeanNo:numTotimeBean.entrySet()) {
				fw.append(timeBeanNo.getKey()+","+timeBeanNo.getValue().toString()+","+timeBean.get(timeBeanNo.getValue()).getFirst()+","+timeBeanNo.getValue().toString()+","+timeBean.get(timeBeanNo.getValue()).getSecond()+"\n");
			}
			fw.flush();
			fw.close();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}
}

class routeDemandAndTTInfo{
	
	private BiMap<Integer,Id<Route>> numToRoute=HashBiMap.create();
	private BiMap<Integer,Integer> numToTimeBean=HashBiMap.create();
	private Map<Id<Route>,Route> routes;
	private Map<Integer,Tuple<Double,Double>>timeBeans;
	private final Population population;
	private INDArray demand;
	private INDArray averageTT;
	private String populationDetails=null;
	
	public routeDemandAndTTInfo(Map<Id<Route>,Route> routes,Map<Integer,Tuple<Double,Double>>timeBeans, Population population) {
		this.routes=routes;
		this.timeBeans=timeBeans;
		int i=0;
		for(Id<Route>routeId:routes.keySet()) {
			this.numToRoute.put(i, routeId);
			i++;
		}
		i=0;
		for(Integer timeId:timeBeans.keySet()) {
			this.numToTimeBean.put(i, timeId);
			i++;
		}
		this.population=population;
		this.demand=Nd4j.create(this.routes.size(),this.timeBeans.size());
		this.averageTT=Nd4j.create(this.routes.size(),this.timeBeans.size());
		this.generateDemandAndAverageTT();
	}
	public routeDemandAndTTInfo(Map<Id<Route>,Route> routes,Map<Integer,Tuple<Double,Double>>timeBeans, Population population,BiMap<Integer,Id<Route>> numToRoute,BiMap<Integer,Integer> numToTimeBean) {
		this.routes=routes;
		this.timeBeans=timeBeans;
		this.numToRoute=numToRoute;
		this.numToTimeBean=numToTimeBean;
		this.population=population;
		this.demand=Nd4j.create(this.routes.size(),this.timeBeans.size());
		this.averageTT=Nd4j.create(this.routes.size(),this.timeBeans.size());
		this.generateDemandAndAverageTT();
	}
	private void generateDemandAndAverageTT() {
		for(Person p:population.getPersons().values()){
			for(PlanElement pe:p.getSelectedPlan().getPlanElements()) {
				if(pe instanceof Leg) {
					Leg leg=(Leg)pe;
					Id<Route> routeid= Id.create(leg.getRoute().getRouteDescription(),Route.class);
					int routeNo=this.numToRoute.inverse().get(routeid);
					int timeId=this.getTimeId(leg.getDepartureTime().seconds());
					this.demand.putScalar(routeNo, timeId,this.demand.getDouble(routeNo,timeId)+1);
					this.averageTT.putScalar(routeNo, timeId,this.averageTT.getDouble(routeNo,timeId)+leg.getTravelTime().seconds());
				}
			}
		}
		this.averageTT=this.averageTT.div(this.demand);
	}
	private int getTimeId(double time) {
		if(time==0) {
			time=1;
		}
		for(Entry<Integer, Tuple<Double, Double>> timeBean:this.timeBeans.entrySet()) {
			if(time>timeBean.getValue().getFirst() && time<=timeBean.getValue().getSecond()) {
				return this.numToTimeBean.inverse().get(timeBean.getKey());
			}
		}
		
		return this.timeBeans.size()-1;
	}
	//Give folder location +/
	// there will be four files written Demand, averageTT, RouteMetaData, TimeBeanMetaData;
	public void writeInfo(String fileloc) {
		try {
			FileWriter fw= new FileWriter(new File(fileloc+this.populationDetails+"_Demand.csv"));
			for(int i=0; i<this.routes.size();i++) {
				String s="";
				for(int j=0;j<this.timeBeans.size();j++) {
					fw.append(s+this.demand.getScalar(i,j));
					s=",";
				}
				fw.append("\n");
			}
			fw.flush();
			fw.close();
			
			fw= new FileWriter(new File(fileloc+this.populationDetails+"_AverageTT.csv"));
			for(int i=0; i<this.routes.size();i++) {
				String s="";
				for(int j=0;j<this.timeBeans.size();j++) {
					fw.append(s+this.averageTT.getScalar(i,j));
					s=",";
				}
				fw.append("\n");
			}
			fw.flush();
			fw.close();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}
	
	public BiMap<Integer, Id<Route>> getNumToRoute() {
		return numToRoute;
	}
	public BiMap<Integer, Integer> getNumToTimeBean() {
		return numToTimeBean;
	}
	public Map<Id<Route>, Route> getRoutes() {
		return routes;
	}
	public Map<Integer, Tuple<Double, Double>> getTimeBeans() {
		return timeBeans;
	}
	public Population getPopulation() {
		return population;
	}
	public INDArray getDemand() {
		return demand;
	}
	public INDArray getAverageTT() {
		return averageTT;
	}
	public String getPopulationDetails() {
		return populationDetails;
	}
	public void setPopulationDetails(String populationDetails) {
		this.populationDetails = populationDetails;
	}
	
}
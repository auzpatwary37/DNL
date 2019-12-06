package training;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.concurrent.ConcurrentHashMap;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

import org.matsim.api.core.v01.Id;
import org.matsim.api.core.v01.network.Link;
import org.matsim.api.core.v01.network.Network;
import org.matsim.api.core.v01.population.Leg;
import org.matsim.api.core.v01.population.Plan;
import org.matsim.api.core.v01.population.PlanElement;
import org.matsim.api.core.v01.population.Population;
import org.matsim.core.config.Config;
import org.matsim.core.config.ConfigUtils;
import org.matsim.core.network.NetworkUtils;
import org.matsim.core.population.PopulationUtils;
import org.matsim.core.utils.collections.Tuple;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.buffer.util.DataTypeUtil;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.ops.transforms.Transforms;

import kriging.BPRBaseFunction;
import kriging.Data;
import kriging.KrigingInterpolator;
import kriging.KrigingModelReader;
import kriging.KrigingModelWriter;
import kriging.MeanBaseFunction;
import linktolinkBPR.LinkToLink;
import linktolinkBPR.LinkToLinks;
import linktolinkBPR.SignalFlowReductionGenerator;

public class trainingAndTiming {
	public static void main(String[] args) {
		
		
		String modelFolderName="newLargeDataSet";
		String modelName="MeanModelFS_Dec";
		String modelOtherFolderName="largeDataset";
		int N=33;
		int T=9;
		String logfileloc="Network/ND/"+modelFolderName+"/ExperimentLog.csv";
		FileWriter logfw=null;
		try {
			logfw = new FileWriter(new File(logfileloc),true);
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		try {
			logfw.append("ModelName,Network,Proportion of trainingData,I,T,ModelLocation,InitialLogLiklihood,FinalLogLiklihood,TrainingTime,AverageError,averagePredictiontime,maxError\n");
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}//header
		
		Network network=NetworkUtils.readNetwork("Network/ND/ndNetwork.xml");
		//network=NetworkUtils.readNetwork("Network/SiouxFalls/network.xml");
		SignalFlowReductionGenerator sg = null;
		Map<Integer,Tuple<Double,Double>> timeBean=new HashMap<>();
		
		for(int i=15;i<24;i++) {
			timeBean.put(i,new Tuple<Double,Double>(i*3600.,i*3600.+3600));
		}
		long time=System.currentTimeMillis();
		LinkToLinks l2ls=new LinkToLinks(network,timeBean,3,3,sg);
		
		System.out.println(System.currentTimeMillis()-time);
		
		DataTypeUtil.setDTypeForContext(DataType.DOUBLE);
		Nd4j.setDefaultDataTypes(DataType.DOUBLE, DataType.DOUBLE);
		
		Map<Integer,Data> trainingData=DataIO.readDataSet("Network/ND/"+modelFolderName+"/DataSetTrain"+800+".txt","Network/ND/"+modelFolderName+"/KeySetTrain"+800+".csv");
		Map<Integer,Data> testingData=DataIO.readDataSet("Network/ND/"+modelFolderName+"/DataSetTest"+800+".txt","Network/ND/"+modelFolderName+"/KeySetTest"+800+".csv");
	//	Map<Integer,Data> testingDataConst=DataIO.readDataSet("Network/ND/"+modelOtherFolderName+"/DataSetTest"+800+".txt","Network/ND/"+modelOtherFolderName+"/KeySetTest"+800+".csv");
		
		TrainingController tc=new TrainingController(l2ls, trainingData);
		Map<String,List<Integer>> n_tindex=null;
		Map<String,Map<Integer,Double>> errorMap=new HashMap<>();
		
		for(int i=100;i<=800;i=i+50) {
		String fullModelName=modelName+"_"+i+"_"+T;
		File file =new File("Network/ND/"+modelFolderName+"/"+fullModelName);
		KrigingInterpolator kriging=null;
		double initialLL=0;
		
		if(i==100) {
			n_tindex=tc.createN_TSpecificTrainingSet(100);
		}else {
			tc.createAdditionalN_TSpecificTrainingSetWithErrorWeight(n_tindex, 50, errorMap);
		}
		
		if(file.exists() && file.list().length!=0) {
			KrigingInterpolator kg=new KrigingInterpolator(trainingData, l2ls, new MeanBaseFunction(trainingData),n_tindex);//change if u want to change the base function
			kriging=new KrigingModelReader().readModel(file.getPath()+"/modelDetails.xml");
			initialLL=kg.calcCombinedLogLikelihood();
			
		}else {
			file.mkdir();
			kriging=new KrigingInterpolator(trainingData, l2ls, new MeanBaseFunction(trainingData),n_tindex);//change of u want to change the base function
			initialLL=kriging.calcCombinedLogLikelihood();
			//kriging.trainKrigingWithoutbeta();
			kriging.trainKriging();
			new KrigingModelWriter(kriging).writeModel(file.getPath());
		}
		
		try {
			logfw.append(fullModelName+","+"ND"+","+i+","+trainingData.size()+","+T+","+file.getPath()+",");
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		double finalLL=kriging.calcCombinedLogLikelihood();
		double trainingTime=kriging.getTrainingTime();
		try {
			logfw.append(Double.toString(initialLL)+","+finalLL+","+trainingTime+",");
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		INDArray averageError=Nd4j.create(N,T);
		double totalTime=0;
		FileWriter fw=null;
		
		try {
			fw = new FileWriter(new File(file.getPath()+"/errorLogger.csv"));
			fw.append("key,errorNorm\n");
			
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		
		
		realAndModelOutput rm=new realAndModelOutput();
		INDArray errorscore=Nd4j.create(testingData.size(),1);
		int k=0;
		for(Entry<Integer, Data> testDataEntry:testingData.entrySet()) {
			Data testData=testDataEntry.getValue();
			INDArray Yreal=testData.getY();
			long startTime=System.currentTimeMillis();
			
			INDArray Y=kriging.getY(testData.getX());
			totalTime+=System.currentTimeMillis()-startTime;
			rm.addDatapair(Yreal, Y);
			INDArray errorArray=Yreal.sub(Y).div(Yreal).mul(100);
			double errorScore=Yreal.sub(Y).norm2Number().doubleValue()/Yreal.norm2Number().doubleValue()*100;
			try {
				fw.append(testData.getKey()+","+errorArray.norm2Number().floatValue()+"\n");
				//errorMap.put(testDataEntry.getKey(), errorScore);
			} catch (IOException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
			errorscore.put(k,1, errorScore);
			errorArray=Transforms.abs(errorArray);
			averageError.addi(errorArray);
			k++;
		}
		averageError.divi(testingData.size());
		Nd4j.writeTxt(averageError, file.getPath()+"/averagePredictionError.txt");
		rm.writeCsv(file.getPath()+"/yrealvsymodel.csv");
		totalTime=totalTime/testingData.size();
		try {
			fw.flush();
			fw.close();
			//logfw.append(Double.toString(averageError.sumNumber().doubleValue()/(N*T))+","+totalTime+","+averageError.maxNumber().doubleValue());
			logfw.append(Double.toString(errorscore.sumNumber().doubleValue()/(testingData.size()))+","+totalTime+","+errorscore.maxNumber().doubleValue());
			logfw.append("\n");
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		
		try {
			logfw.flush();

		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		System.out.println("fisnished training and testing of "+fullModelName);
		//This part now will calculate nt specific error for the training database.
		INDArray errorArrayAverage=Nd4j.create(N,T);
		INDArray errorArrayMax=Nd4j.create(N,T);
		
		realAndModelOutput rmNew=new realAndModelOutput();
		final KrigingInterpolator kriging1=kriging;
		IntStream.rangeClosed(0,N-1).parallel().forEach((n)->
		{
			IntStream.rangeClosed(0,T-1).parallel().forEach((t)->{
				double totalError=0;
				double maxError=0;
				
				String key = Integer.toString(n)+"_"+Integer.toString(t);
				errorMap.put(key,new HashMap<>());
				Map<Integer,Double> errornt=errorMap.get(key);
				List<Integer> indices=kriging1.getVariogram().getNtSpecificOriginalIndices().get(key);
				for(int ii=0;ii<trainingData.size();ii++) {
					if(!indices.contains(ii)) {
						double yReal=trainingData.get(ii).getY().getDouble(n,t);
						double yKriging=kriging1.getY(trainingData.get(ii).getX(), n, t);
						double error=(yReal-yKriging)/yReal*100;
						errornt.put(ii, error);
						rmNew.addDatapair(yReal,yKriging);
						totalError+=error;
						if(maxError<error) {
							maxError=error;
						}
					}
				}
				errorArrayAverage.putScalar(n, t,totalError/(trainingData.size()-indices.size()));
				errorArrayMax.putScalar(n,t,maxError);
				
			});
			
		});
		rmNew.writeCsv(file.getPath()+"/yrealvsymodelTrainingData.csv");
		KrigingInterpolator.writeINDArray(errorArrayAverage, file.getPath()+"/trainingDataErrorAverage.csv");
		KrigingInterpolator.writeINDArray(errorArrayAverage, file.getPath()+"/trainingDataErrorMax.csv");
		
//		 //This is for testing purpose only
//		 k=0;
//		 realAndModelOutput rmNew=new realAndModelOutput();
//		 INDArray averageErrorNew=Nd4j.create(N,T);
//		 INDArray errorscoreNew=Nd4j.create(testingData.size(),1);
//		 for(Entry<Integer, Data> testDataEntry:testingData.entrySet()) {
//			 Data testData=testDataEntry.getValue();
//			 INDArray Yreal=testData.getY();
//			 //long startTime=System.currentTimeMillis();
//
//			 INDArray Y=kriging.getY(testData.getX());
//			 //totalTime+=System.currentTimeMillis()-startTime;
//			 rmNew.addDatapair(Yreal, Y);
//			 INDArray errorArray=Yreal.sub(Y).div(Yreal).mul(100);
//			 double errorScore=Yreal.sub(Y).norm2Number().doubleValue()/Yreal.norm2Number().doubleValue()*100;
//			 errorscoreNew.put(k,1, errorScore);
//			 errorArray=Transforms.abs(errorArray);
//			 averageErrorNew.addi(errorArray);
//			 k++;
//		 }
//		 averageErrorNew.divi(testingData.size());
//		 Nd4j.writeTxt(averageErrorNew, file.getPath()+"/averagePredictionErrorConst.txt");
//		 rmNew.writeCsv(file.getPath()+"/yrealvsymodelConst.csv");
//		 KrigingInterpolator.writeINDArray(errorscoreNew, file.getPath()+"/errorScoreConst.csv");
//		 //Test End
		 
		 System.out.println("trainingDataSize = "+trainingData.size()+" and testing data size = "+testingData.size());
		}
		try {
			logfw.close();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
//		for(double d=0.3;d<=.9;d=d+.1) {
//			Map<Integer,Data> trainingData=DataIO.readDataSet("Network/ND/"+modelFolderName+"/DataSetNDTrain"+d+".txt","Network/ND/"+modelFolderName+"/KeySetNDTrain"+d+".csv");
//			Map<Integer,Data> testingData=DataIO.readDataSet("Network/ND/"+modelFolderName+"/DataSetNDTest"+d+".txt","Network/ND/"+modelFolderName+"/KeySetNDTest"+d+".csv");
//			String fullModelName=modelName+"_"+d+"_"+T;
//			File file =new File("Network/ND/"+modelFolderName+"/"+fullModelName);
//			KrigingInterpolator kriging=null;
//			double initialLL=0;
//			if(file.exists() && file.list().length!=0) {
//				KrigingInterpolator kg=new KrigingInterpolator(trainingData, l2ls, new BPRBaseFunction(l2ls));//change of u want to change the base function
//				kriging=new KrigingModelReader().readModel(file.getPath()+"/modelDetails.xml");
//				initialLL=kg.calcCombinedLogLikelihood();
//				
//			}else {
//				file.mkdir();
//				kriging=new KrigingInterpolator(trainingData, l2ls, new BPRBaseFunction(l2ls));//change of u want to change the base function
//				initialLL=kriging.calcCombinedLogLikelihood();
//				kriging.deepTrainKriging();
//				new KrigingModelWriter(kriging).writeModel(file.getPath());
//			}
//			
//			try {
//				logfw.append(fullModelName+","+"ND"+","+d+","+trainingData.size()+","+T+","+file.getPath()+",");
//			} catch (IOException e) {
//				// TODO Auto-generated catch block
//				e.printStackTrace();
//			}
//			//make model
//			
//			
//			double finalLL=kriging.calcCombinedLogLikelihood();
//			double trainingTime=kriging.getTrainingTime();
//			try {
//				logfw.append(Double.toString(initialLL)+","+finalLL+","+trainingTime+",");
//			} catch (IOException e) {
//				// TODO Auto-generated catch block
//				e.printStackTrace();
//			}
//			
//			INDArray averageError=Nd4j.create(N,T);
//			double totalTime=0;
//			Config configcurrent=ConfigUtils.createConfig();
//			ConfigUtils.loadConfig(configcurrent, "Network/ND/final_config.xml");
//			double[] ratios=new double[] {0.5,.625,0.75,.875,1,1.125,1.25,1.375,1.5};
//			int testDataNum=0;
//			for(Data testData:testingData.values()) {
//				INDArray Yreal=testData.getY();
//				long startTime=System.currentTimeMillis();
//				Population population=getPopulation(testData.getKey(),"Network/ND/"+modelFolderName,configcurrent,ratios);
//				
//				Tuple<INDArray,INDArray> xy=kriging.getXYIterative(population);
//				totalTime+=System.currentTimeMillis()-startTime;
//				INDArray errorArrayY=Yreal.sub(xy.getSecond()).div(Yreal).mul(100);
//				
//				errorArrayY=Transforms.abs(errorArrayY);
//				averageError.addi(errorArrayY);
//				testDataNum++;
//				
//			}
//			averageError.divi(testingData.size());
//			Nd4j.writeTxt(averageError, file.getPath()+"/averagePredictionError.txt");
//			totalTime=totalTime/testingData.size();
//			try {
//				logfw.append(Double.toString(averageError.sumNumber().doubleValue()/(N*T))+","+totalTime+","+averageError.maxNumber().doubleValue());
//				logfw.append("\n");
//			} catch (IOException e) {
//				// TODO Auto-generated catch block
//				e.printStackTrace();
//			}
//			
//			try {
//				logfw.flush();
//
//			} catch (IOException e) {
//				// TODO Auto-generated catch block
//				e.printStackTrace();
//			}
//			System.out.println("fisnished training and testing of "+fullModelName);
//			
//		}
//		try {
//			logfw.close();
//		} catch (IOException e) {
//			// TODO Auto-generated catch block
//			e.printStackTrace();
//		}
		
	}
	
	public static INDArray updateX(INDArray TT,Population population,LinkToLinks l2ls) {
		INDArray X=Nd4j.create(TT.shape());
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
						int n=l2ls.getNumToLinkToLink().inverse().get(l2lId);
						int t=l2ls.getTimeId(time);
						String key=Integer.toString(n)+"_"+Integer.toString(t);
						if(linkToLinksDemand.containsKey(key)) {
							linkToLinksDemand.put(key, linkToLinksDemand.get(key)+1);
						}else {
							linkToLinksDemand.put(key,1.);
						}

						time+=TT.getDouble(n,t);
					}
				}else {
					continue;
				}
			}
		});

		linkToLinksDemand.entrySet().parallelStream().forEach((n_t_d)->{		
			String key=n_t_d.getKey();
			int n=Integer.parseInt(key.split("_")[0]);
			int t=Integer.parseInt(key.split("_")[1]);
			if(linkToLinksDemand.containsKey(key)) {
				if(linkToLinksDemand.get(key)==Double.NaN) {
					System.out.println();
				}

				X.putScalar(n,t,linkToLinksDemand.get(key));
			}else {
				X.putScalar(n,t,0);
			}
		});
		return X;
	}
	
	public static Population getPopulation(String key,String baseFolderLoc,Config config,double[] ratios) {
		int simIter=0;
		int iter=Integer.parseInt(key.split("_")[1]);
		double ratio=Double.parseDouble(key.split("_")[0]);
		for(int i=0;i<ratios.length;i++) {
			if(ratio==ratios[i]) {
				simIter=i;
				break;
			}
		}
		Population population=PopulationUtils.createPopulation(config);
		PopulationUtils.readPopulation(population, baseFolderLoc+"/output"+simIter+"/Iters/it."+iter+"/"+iter+".plans.xml.gz");
		return population;
	}
	
	
}


class realAndModelOutput{
	private INDArray realOutputVsModelOutput;
	private INDArray realOutput=null;
	private INDArray modelOutput=null;
	private int pointRecorded=0;
	public realAndModelOutput() {
		this.realOutputVsModelOutput=null;
	}
	
	public void addDatapair(double yReal, double yKriging) {
		if(pointRecorded==0) {
			this.realOutput=Nd4j.create(new double[] {yReal});
			this.modelOutput=Nd4j.create(new double[] {yKriging});
		}else {
			this.realOutput=Nd4j.concat(0, this.realOutput,Nd4j.create(new double[] {yReal}));
			this.modelOutput=Nd4j.concat(0,this.modelOutput,Nd4j.create(new double[] {yKriging}));
		}
		this.pointRecorded++;
	}

	public void addDatapair(INDArray real,INDArray model) {
		INDArray realvector=real.reshape(real.length());
		INDArray modelvector=model.reshape(model.length());
		INDArray ar=Nd4j.create(realvector.length(),2);
		ar.putColumn(0, realvector);
		ar.putColumn(1, modelvector);
		if(pointRecorded==0) {
			this.realOutputVsModelOutput=ar;
		}else {
			this.realOutputVsModelOutput=Nd4j.concat(0, this.realOutputVsModelOutput,ar);
		}
		this.pointRecorded++;
	}
	
	
	public void writeCsv(String fileLoc) {
		if(this.realOutputVsModelOutput==null) {
			this.realOutputVsModelOutput=Nd4j.create(this.realOutput.size(0),2);
			this.realOutputVsModelOutput.putColumn(0, this.realOutput);
			this.realOutputVsModelOutput.putColumn(1, this.modelOutput);
		}
		KrigingInterpolator.writeINDArray(this.realOutputVsModelOutput, fileLoc);
	}
	
}
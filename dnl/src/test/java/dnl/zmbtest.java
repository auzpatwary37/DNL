package dnl;

import static org.junit.jupiter.api.Assertions.*;

import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;

import org.junit.Assert;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.matsim.api.core.v01.network.Network;
import org.matsim.core.network.NetworkUtils;
import org.matsim.core.utils.collections.Tuple;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.buffer.util.DataTypeUtil;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import kriging.Data;
import kriging.KrigingInterpolator;
import kriging.MeanBaseFunction;
import kriging.VarianceInfoHolder;
import linktolinkBPR.LinkToLinks;
import linktolinkBPR.SignalFlowReductionGenerator;
import training.DataIO;

class zmbtest {
	KrigingInterpolator kriging;
	@BeforeEach
	void setUp() throws Exception {
		DataTypeUtil.setDTypeForContext(DataType.FLOAT);
		Nd4j.setDefaultDataTypes(DataType.FLOAT, DataType.FLOAT);
		Map<Integer,Data> trainingData=DataIO.readDataSet("Network/ND/DataSetNDTrain.txt","Network/ND/KeySetNDTrain.csv");
		Network network=NetworkUtils.readNetwork("Network/ND/ndNetwork.xml");
		//Network network=NetworkUtils.readNetwork("Network/SiouxFalls/network.xml");
		SignalFlowReductionGenerator sg = null;
		//config.network().setInputFile("Network/SiouxFalls/network.xml");
		Map<Integer,Tuple<Double,Double>> timeBean=new HashMap<>();
		for(int i=15;i<24;i++) {
			timeBean.put(i,new Tuple<Double,Double>(i*3600.,i*3600.+3600));
		}
		LinkToLinks l2ls=new LinkToLinks(network,timeBean,3,3,sg);
		this.kriging=new KrigingInterpolator(trainingData, l2ls, new MeanBaseFunction(trainingData));
	
	}

	@Test
	void test() {
		int n=2;
		int t=1;
		String key=Integer.toString(n)+"_"+Integer.toString(t);
		VarianceInfoHolder infoAll=kriging.preProcessData(kriging.getBeta(), kriging.getVariogram().gettheta());
		INDArray Z_MBall=Nd4j.create(kriging.getVariogram().getNtSpecificOriginalIndices().get(key).size(),1);
		for(int j=0;j<Z_MBall.size(0);j++) {
			Z_MBall.putScalar(j,0,infoAll.getZ_MB().getDouble(n,t,kriging.getVariogram().getNtSpecificOriginalIndices().get(key).get(j)));
		}
		
		INDArray Z_MBnt=Nd4j.create(Z_MBall.shape());
		VarianceInfoHolder infont=kriging.preProcessNtSpecificData(n, t, kriging.getBeta().getDouble(n,t), kriging.getVariogram().gettheta().getDouble(n,t), infoAll);
		for(int j=0;j<Z_MBnt.size(0);j++) {
			Z_MBnt.putScalar(j,0,infont.getZ_MB().getDouble(n,t,kriging.getVariogram().getNtSpecificOriginalIndices().get(key).get(j)));
		}
		System.out.println(Arrays.toString(Z_MBall.toDoubleVector()));
		System.out.println(Arrays.toString(Z_MBnt.toDoubleVector()));
		assertTrue(Arrays.equals(Z_MBall.toDoubleVector(), Z_MBnt.toDoubleVector()));
	}
	
	

}

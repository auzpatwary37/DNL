package kriging;

import java.util.HashMap;
import java.util.Map;
import java.util.Map.Entry;
import java.util.concurrent.ConcurrentHashMap;
import java.util.stream.IntStream;

import org.matsim.core.utils.collections.Tuple;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.w3c.dom.Element;
import org.xml.sax.Attributes;

import linktolinkBPR.LinkToLinks;

public class FreeFlowPlusAkcelikDelayBaseFunction implements BaseFunction{

	public static int no=0;
	private Map<Integer,Link2LinkInfoHolder> link2LinkInfo=new ConcurrentHashMap<>();
	private Map<Integer,Double> timeBeanLength=new HashMap<>();
	
	public FreeFlowPlusAkcelikDelayBaseFunction( LinkToLinks l2ls) {
		for(int i=0;i<l2ls.getLinkToLinks().size();i++) {
			link2LinkInfo.put(i, new Link2LinkInfoHolder(l2ls.getLinkToLinks().get(i),i));
		}
		for(Entry<Integer,Integer> timeMap:l2ls.getNumToTimeBean().entrySet()) {
			Tuple<Double,Double>tb=l2ls.getTimeBean().get(timeMap.getValue());
			this.timeBeanLength.put(timeMap.getKey(),tb.getSecond()-tb.getFirst());
		}
	}
	
	public FreeFlowPlusAkcelikDelayBaseFunction(Map<Integer,Link2LinkInfoHolder> link2LinkInfo,Map<Integer,Double>timeBeanLength) {
		this.link2LinkInfo=link2LinkInfo;
		this.timeBeanLength=timeBeanLength;
	}
	
	@Override
	public INDArray getY(INDArray X) {
		INDArray Y=Nd4j.create(X.size(0), X.size(1));
		no++;
		IntStream.rangeClosed(0,Math.toIntExact(X.size(0))-1).parallel().forEach((n)->
		{
			IntStream.rangeClosed(0,Math.toIntExact(X.size(1))-1).parallel().forEach((t)->{
		
//		for(int n=0;n<X.size(0);n++) {
//			for(int t=0;t<X.size(1);t++) {
				double tt=this.getLinkToLinkAkcelikDelay(X.getDouble(n, t), n,this.timeBeanLength.get(t));
				Y.putScalar(n, t, tt);
//			}
//		}
			});
		});
		
		return Y;
	}
	
	public Double getLinkToLinkAkcelikDelay(double demand,Integer n,double time) {
		Link2LinkInfoHolder l2l=this.link2LinkInfo.get(n);
		if(demand>=l2l.getSaturationFlow()) {
			demand =l2l.getSaturationFlow()-1;
		}
		Double delay=l2l.getFromLinkFreeFlowTime();//Add the free flow travel time of the link
		
		double overFlowQueue=0;
		double x=demand/(l2l.getG_cRatio()*l2l.getSaturationFlow());
		double xPrime=0.67+l2l.getSaturationFlow()*l2l.getG_cRatio()*l2l.getCycleTime()/(3600*600);
		//Calculate the overflow if x > xPrime; For details please see section 2.5.3 in TPDM 
		if(x>xPrime) {
			overFlowQueue=l2l.getG_cRatio()*l2l.getSaturationFlow()*time/4*(x-1+Math.sqrt((x-1)*(x-1)+12*(x-xPrime)/(l2l.getG_cRatio()*l2l.getSaturationFlow()*time)));
		}
		if(demand>0 && l2l.getG_cRatio()<1) {
			delay+=l2l.getCycleTime()/2.*Math.pow(1-l2l.getG_cRatio(), 2)/(2*(1-demand/l2l.getSaturationFlow()))+overFlowQueue*x/demand;
		}
		
		if(Double.isNaN(delay)||delay==Double.POSITIVE_INFINITY) {
			System.out.println();
		}
		//System.out.println();
		return delay;
	}
	

	@Override
	public void writeBaseFunctionInfo(Element baseFunction,String fileLoc) {
		baseFunction.setAttribute("ClassName", this.getClass().getName());
		for(Entry<Integer,Link2LinkInfoHolder>l2l:this.link2LinkInfo.entrySet()) {
			baseFunction.setAttribute(Integer.toString(l2l.getKey()), l2l.getValue().toString());
		}
		StringBuilder sb = new StringBuilder();
		String prefix = "";
        for (Entry<Integer,Double>timeLength: this.timeBeanLength.entrySet()) {
            sb.append(prefix);
            prefix=",";
        	sb.append(timeLength.getValue());
        }
        String s = sb.toString();
		
		baseFunction.setAttribute("timeLength", s);
	}
	
	public static BaseFunction parseBaseFunction(Attributes a) {
		Map<Integer,Link2LinkInfoHolder> link2LinkInfo=new ConcurrentHashMap<>();
		Map<Integer,Double> timeLength=new HashMap<>();
		for(int i=0;i<a.getLength();i++) {
			if(!a.getQName(i).equals("ClassName") && !a.getQName(i).equals("timeLength")) {
				Link2LinkInfoHolder l2l=Link2LinkInfoHolder.createLinkToLinkInfo(a.getValue(i));
				link2LinkInfo.put(l2l.getN(), l2l);
			}else if(a.getQName(i).equals("timeLength")) {
				String[] part=a.getValue(i).split(",");
				for(int j=0;j<part.length;j++) {
					timeLength.put(j, Double.parseDouble(part[j]));
				}
			}
		}
		return new FreeFlowPlusAkcelikDelayBaseFunction(link2LinkInfo,timeLength);
	}

	@Override
	public double getntSpecificY(INDArray X, int n, int t) {
		double tt=this.getLinkToLinkAkcelikDelay(X.getDouble(n, t), n,this.timeBeanLength.get(t));
		return tt;
	}

}

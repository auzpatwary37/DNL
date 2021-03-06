package kriging;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.HashMap;
import java.util.Map;
import java.util.Map.Entry;
import java.util.concurrent.ConcurrentHashMap;

import org.matsim.core.utils.collections.Tuple;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.conditions.Conditions;
import org.w3c.dom.Element;
import org.xml.sax.Attributes;

import linktolinkBPR.LinkToLinks;

public class BPRBaseFunction implements BaseFunction{
	private INDArray alpha;
	private INDArray beta;
	public static int no=0;
	private Map<Integer,Link2LinkInfoHolder> link2LinkInfo=new ConcurrentHashMap<>();
	private Map<Integer,Double> timeBeanLength=new HashMap<>();
	
	public BPRBaseFunction( LinkToLinks l2ls) {
		for(int i=0;i<l2ls.getLinkToLinks().size();i++) {
			link2LinkInfo.put(i, new Link2LinkInfoHolder(l2ls.getLinkToLinks().get(i),i));
		}
		for(Entry<Integer,Integer> timeMap:l2ls.getNumToTimeBean().entrySet()) {
			Tuple<Double,Double>tb=l2ls.getTimeBean().get(timeMap.getValue());
			this.timeBeanLength.put(timeMap.getKey(),tb.getSecond()-tb.getFirst());
		}
		alpha=Nd4j.ones(this.link2LinkInfo.size(),this.timeBeanLength.size()).muli(.15);
		beta=Nd4j.ones(alpha.shape()).muli(4);
	}
	
	private BPRBaseFunction(Map<Integer,Link2LinkInfoHolder> link2LinkInfo,Map<Integer,Double>timeBeanLength,INDArray alpha,INDArray beta) {
		this.link2LinkInfo=link2LinkInfo;
		this.timeBeanLength=timeBeanLength;
		this.alpha=alpha;
		this.beta=beta;
	}
	
	@Override
	public INDArray getY(INDArray X) {
		INDArray Y=Nd4j.create(X.size(0), X.size(1));
		no++;
//		IntStream.rangeClosed(0,Math.toIntExact(X.size(0))-1).parallel().forEach((n)->
//		{
//			IntStream.rangeClosed(0,Math.toIntExact(X.size(1))-1).parallel().forEach((t)->{
		
		for(int n=0;n<X.size(0);n++) {
			for(int t=0;t<X.size(1);t++) {
				double linkFlow=0;
				double averageGC=0;
				for(int nn:this.link2LinkInfo.get(n).getPrimaryFromLinkProximitySet()) {
					linkFlow+=X.getDouble(nn,t);
					averageGC+=this.link2LinkInfo.get(nn).getG_cRatio();
				}
				averageGC=averageGC/this.link2LinkInfo.get(n).getPrimaryFromLinkProximitySet().size();
				double tt=this.getLinkToLinkBPRDelay(linkFlow,averageGC, n,this.timeBeanLength.get(t),alpha.getDouble(n,t),beta.getDouble(n,t));
				Y.putScalar(n, t, tt);
				if(Y.cond(Conditions.isInfinite()).any()||Y.cond(Conditions.isNan()).any()) {
					System.out.println("Z is nan or inf!!!");
				}
				
			}
		}
//			});
//		});
		
		//System.out.println();
		return Y;
	}
	

	private double getLinkToLinkBPRDelay(double demand, int n,double timeLength) {
		Link2LinkInfoHolder l2l=this.link2LinkInfo.get(n);
		Double delay=l2l.getFromLinkFreeFlowTime()*(1+0.15*Math.pow((demand/(l2l.getSaturationFlow()*l2l.getG_cRatio()*timeLength)),4));
		
		return delay;
	}
	
	private double getLinkToLinkBPRDelay(double demand,double averageGC, int n,double timeLength,double alpha,double beta) {
		Link2LinkInfoHolder l2l=this.link2LinkInfo.get(n);
		Double delay=l2l.getFromLinkFreeFlowTime()*(1+alpha*Math.pow((demand/(l2l.getSaturationFlow()*averageGC*timeLength/3600)),beta));
		
		return delay;
	}

	@Override
	public void writeBaseFunctionInfo(Element baseFunction, String fileLoc) {
		baseFunction.setAttribute("ClassName", this.getClass().getName());
		for(Entry<Integer,Link2LinkInfoHolder>l2l:this.link2LinkInfo.entrySet()) {
			try {
			baseFunction.setAttribute("a_"+l2l.getKey(), l2l.getValue().toString());
			}catch(Exception e) {
				System.out.println(e);
			}
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
		
		Nd4j.writeTxt(this.alpha,  fileLoc+"/bprAlpha.txt");
		baseFunction.setAttribute("alphaLocation", fileLoc+"/bprAlpha.txt");
		
		Nd4j.writeTxt(this.beta,  fileLoc+"/bprBeta.txt");
		baseFunction.setAttribute("betaLocation", fileLoc+"/bprBeta.txt");
	}
	
	
	public void writecsvLinktoLinkinfo(String fileLoc) {
		try {
			FileWriter fw = new FileWriter(new File(fileLoc));
			for(Link2LinkInfoHolder l2l2:this.link2LinkInfo.values()) {
				fw.append(l2l2.toString()+"\n");
				fw.flush();
			}
			fw.close();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}
	
	public static BaseFunction parseBaseFunction(Attributes a) {
		Map<Integer,Link2LinkInfoHolder> link2LinkInfo=new ConcurrentHashMap<>();
		Map<Integer,Double> timeLength=new HashMap<>();
		for(int i=0;i<a.getLength();i++) {
			if(!a.getQName(i).equals("ClassName") && !a.getQName(i).equals("timeLength")&& !a.getQName(i).equals("alphaLocation")&& !a.getQName(i).equals("betaLocation")) {
				Link2LinkInfoHolder l2l=Link2LinkInfoHolder.createLinkToLinkInfo(a.getValue(i));
				link2LinkInfo.put(l2l.getN(), l2l);
			}else if(a.getQName(i).equals("timeLength")) {
				String[] part=a.getValue(i).split(",");
				for(int j=0;j<part.length;j++) {
					timeLength.put(j, Double.parseDouble(part[j]));
				}
			}
		}
		INDArray alpha;
		INDArray beta;
		if(a.getValue("alphaLocation")!=null) {
			alpha=Nd4j.readTxt(a.getValue("alphaLocation"));
			beta=Nd4j.readTxt(a.getValue("betaLocation"));
		}else {
			alpha=Nd4j.ones(link2LinkInfo.size(),timeLength.size()).muli(.15);
			beta=Nd4j.ones(alpha.shape()).muli(4);
		}
		return new BPRBaseFunction(link2LinkInfo,timeLength,alpha,beta);
	}

	@Override
	public double getntSpecificY(INDArray X, int n, int t) {
		double linkFlow=0;
		double averageGC=0;
		for(int nn:this.link2LinkInfo.get(n).getPrimaryFromLinkProximitySet()) {
			linkFlow+=X.getDouble(nn,t);
			averageGC+=this.link2LinkInfo.get(nn).getG_cRatio();
		}
		averageGC=averageGC/this.link2LinkInfo.get(n).getPrimaryFromLinkProximitySet().size();
		double tt=this.getLinkToLinkBPRDelay(linkFlow,averageGC, n,this.timeBeanLength.get(t),alpha.getDouble(n,t),beta.getDouble(n,t));
		return tt;
	}

	public INDArray getAlpha() {
		return alpha;
	}

	public void setAlpha(INDArray alpha) {
		this.alpha = alpha;
	}

	public INDArray getBeta() {
		return beta;
	}

	public void setBeta(INDArray beta) {
		this.beta = beta;
	}
	
	public void setAlphaBetaNTSpecific(double alpha, double beta,int n,int t) {
		this.alpha.put(n, t,alpha);
		this.beta.put(n,t, beta);
	}
}

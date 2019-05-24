package linktolinkBPR;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Set;

import org.apache.commons.math3.linear.OpenMapRealMatrix;
import org.apache.commons.math3.linear.RealMatrix;
import org.matsim.api.core.v01.Id;
import org.matsim.api.core.v01.network.Link;
import org.matsim.api.core.v01.network.Network;
import org.matsim.core.config.Config;
import org.matsim.core.config.ConfigUtils;
import org.matsim.core.network.NetworkUtils;
import org.matsim.core.utils.collections.Tuple;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import com.google.common.collect.BiMap;
import com.google.common.collect.HashBiMap;


public class LinkToLinks {
	private List<LinkToLink> linkToLinks=new ArrayList<>();
	private Map<Id<Link>,List<LinkToLink>>fromLinkToLinkMap=new HashMap<>();
	private Map<Id<Link>,List<LinkToLink>>ToLinkToLinkMap=new HashMap<>();
	private BiMap<Integer,Id<LinkToLink>> numToLinkToLink=HashBiMap.create();
	private final Network network;
	private final Map<Integer,Tuple<Double,Double>>timeBean;
	private int l2lCounter=0;
	private Map<String,INDArray>weights=new HashMap<>();
	/**
	 * 
	 * TODO:Add signal info in this constructor as well
	 * @param network
	 */
	public LinkToLinks(Network network,Map<Integer,Tuple<Double,Double>>timeBean,int kn,int kt,SignalFlowReductionGenerator sg) {
		this.network=network;
		//Time bean has to continuous, there cannot be any gap between. Should be homogeneous as well. Should we make it endogenous? and take the 
		//number of time bean as input instead? What will happen to the input demand of link to link?
		this.timeBean=timeBean;
		this.generateLinkToLinkMap();
		for(int n=0;n<this.linkToLinks.size();n++) {
			Link fromLink=this.linkToLinks.get(n).getFromLink();
			Link toLink=this.linkToLinks.get(n).getToLink();
			if(sg!=null) {
				this.linkToLinks.get(n).setG_cRatio(sg.getGCratio(fromLink, toLink.getId())[0]);
				this.linkToLinks.get(n).setCycleTime(sg.getGCratio(fromLink, toLink.getId())[1]);
			}
			for(int t=0;t<timeBean.size();t++) {
				this.weights.put(Integer.toString(n)+"_"+Integer.toString(t), this.generateWeightMatrix(n, t, kn, kt));
			}
		}
	}
	
	/**
	 * 
	 */
	public void generateLinkToLinkMap() {
		for(Entry<Id<Link>, ? extends Link> e:network.getLinks().entrySet()){
			for(Link l:e.getValue().getToNode().getOutLinks().values()) {
				LinkToLink l2l=new LinkToLink(e.getValue(),l,timeBean);
				this.addLinkToLink(l2l);
			}
			
		}
	}
	
	public List<LinkToLink> getLinkToLinks() {
		return linkToLinks;
	}

	public Map<Id<Link>, List<LinkToLink>> getFromLinkToLinkMap() {
		return fromLinkToLinkMap;
	}

	public Map<Id<Link>, List<LinkToLink>> getToLinkToLinkMap() {
		return ToLinkToLinkMap;
	}

	public BiMap<Integer, Id<LinkToLink>> getNumToLinkToLink() {
		return numToLinkToLink;
	}

	public Network getNetwork() {
		return network;
	}

	public Map<Integer, Tuple<Double, Double>> getTimeBean() {
		return timeBean;
	}

	public int getL2lCounter() {
		return l2lCounter;
	}

	public void addLinkToLink(LinkToLink l2l) {
		if(this.numToLinkToLink.inverse().containsKey(l2l.getLinkToLinkId())) {
			return;
		}
		this.linkToLinks.add(l2l);
		this.numToLinkToLink.put(l2lCounter, l2l.getLinkToLinkId());
		if(this.fromLinkToLinkMap.containsKey(l2l.getFromLink().getId())) {
			this.fromLinkToLinkMap.get(l2l.getFromLink().getId()).add(l2l);
		}else {
			this.fromLinkToLinkMap.put(l2l.getFromLink().getId(),new ArrayList<LinkToLink>());
			this.fromLinkToLinkMap.get(l2l.getFromLink().getId()).add(l2l);
		}
		
		if(this.ToLinkToLinkMap.containsKey(l2l.getToLink().getId())) {
			this.ToLinkToLinkMap.get(l2l.getToLink().getId()).add(l2l);
		}else {
			this.ToLinkToLinkMap.put(l2l.getToLink().getId(),new ArrayList<LinkToLink>());
			this.ToLinkToLinkMap.get(l2l.getToLink().getId()).add(l2l);
		}
		this.l2lCounter++;
	}
	
	
	private INDArray generateWeightMatrix(int n,int t,int kn,int kt){
		LinkToLink l2l=this.linkToLinks.get(n);
		INDArray we=Nd4j.create(this.linkToLinks.size(), this.timeBean.size());
		//double weight[][]=new double[this.linkToLinks.size()][this.timeBean.size()];
		Map<Integer,Set<LinkToLink>>linkToLinkMap=new HashMap<>();
		linkToLinkMap.put(0, new HashSet<>());
		linkToLinkMap.get(0).add(l2l);
		linkToLinkMap.put(1, new HashSet<>());
		this.fetchLinkToLinkWithFromLink(l2l.getToLink(), 1, kn, linkToLinkMap);
		this.fetchLinkToLinkWithToLink(l2l.getFromLink(), 1, kn, linkToLinkMap);
		for(Entry<Integer,Set<LinkToLink>> linkToLinks:linkToLinkMap.entrySet()) {
			double wk;
			if(linkToLinks.getKey()==0) {
				wk=3;
			}else {
				wk=1+1/linkToLinks.getKey();
			}
			for(LinkToLink ll2ll:linkToLinks.getValue()) {
				int l2lIndex=this.numToLinkToLink.inverse().get(ll2ll.getLinkToLinkId()); 
				for(int tt=Math.max(t-kt,0);tt<=Math.min(this.timeBean.size()-1,t+kt);tt++) {
					float wt=0;
					if(tt-t==0) {
						wt=3;
					}else {
						wt=1+1/Math.abs(tt-t);
					}
					//weight[l2lIndex][tt]=wk*wt;
					we.putScalar(l2lIndex, tt, wk*wt);
				}
			}
		}
		//double[][] a =we.getData();
		return we;
	}
	
	private Map<Integer,Set<LinkToLink>>fetchLinkToLinkWithFromLink(Link toLink,int k,int kn,Map<Integer,Set<LinkToLink>> linkToLinks){
		if(this.fromLinkToLinkMap.get(toLink.getId())!=null) {
		for(LinkToLink l2l:this.fromLinkToLinkMap.get(toLink.getId())) {
			boolean linkExist=false;
			for(Set<LinkToLink> l2ls:linkToLinks.values()) {
				if(l2ls.contains(l2l)) {
					linkExist=true;
				}
			}
			if(linkExist==true) {
				continue;
			}
			linkToLinks.get(k).add(l2l);
			k=k+1;
			if(k>kn) {
				return linkToLinks;
			}
			if(!linkToLinks.containsKey(k)) {
				linkToLinks.put(k, new HashSet<>());
			}
			fetchLinkToLinkWithFromLink(l2l.getToLink(),k,kn,linkToLinks);
		}
		}
		return linkToLinks;
	}
	
	private Map<Integer,Set<LinkToLink>>fetchLinkToLinkWithToLink(Link fromLink,int k,int kn,Map<Integer,Set<LinkToLink>> linkToLinks){
		if(this.ToLinkToLinkMap.get(fromLink.getId())!=null) {
		for(LinkToLink l2l:this.ToLinkToLinkMap.get(fromLink.getId())) {
			boolean linkExist=false;
			for(Set<LinkToLink> l2ls:linkToLinks.values()) {
				if(l2ls.contains(l2l)) {
					linkExist=true;
				}
			}
			if(linkExist==true) {
				continue;
			}
			linkToLinks.get(k).add(l2l);
			k=k+1;
			if(k>kn) {
				return linkToLinks;
			}
			if(!linkToLinks.containsKey(k)) {
				linkToLinks.put(k, new HashSet<>());
			}
			fetchLinkToLinkWithToLink(l2l.getFromLink(),k,kn,linkToLinks);
		}
		}
		return linkToLinks;
	}
	
	public INDArray getWeightMatrix(int n, int t){
		return this.weights.get(Integer.toString(n)+"_"+Integer.toString(t));
	}
	
	public Map<String,INDArray> getWeightMatrices(){
		return this.weights;
	}
	
	public static void main(String[] args) {
		Network network=NetworkUtils.readNetwork("Network/SiouxFalls/siouxfallsNetwork.xml");
		//Network network=NetworkUtils.readNetwork("Network/SiouxFalls/network.xml");
		Config config =ConfigUtils.createConfig();
		SignalFlowReductionGenerator sg = null;
		//config.network().setInputFile("Network/SiouxFalls/network.xml");
		Map<Integer,Tuple<Double,Double>> timeBean=new HashMap<>();
		for(int i=15;i<18;i++) {
			timeBean.put(i,new Tuple<Double,Double>(i*3600.,i*3600.+3600));
		}
		LinkToLinks l2ls=new LinkToLinks(network,timeBean,3,3,sg);
		System.out.println("Done!!! Total LinkToLink = "+l2ls.getL2lCounter());
	}
	
	
}

package linktolinkBPR;

import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;

import javax.xml.parsers.ParserConfigurationException;
import javax.xml.parsers.SAXParserFactory;

import org.matsim.api.core.v01.Id;
import org.matsim.api.core.v01.network.Link;
import org.matsim.api.core.v01.network.Network;
import org.matsim.core.network.NetworkUtils;
import org.matsim.core.utils.collections.Tuple;
import org.xml.sax.Attributes;
import org.xml.sax.SAXException;
import org.xml.sax.helpers.DefaultHandler;

import com.google.common.collect.BiMap;
import com.google.common.collect.HashBiMap;

public class LinkToLinksReader extends DefaultHandler{
	private Map<Id<LinkToLink>,LinkToLink> linkToLinkMap=new HashMap<>();
	private int kn=0;
	private int kt=0;
	private BiMap<Integer,Id<LinkToLink>> numToLinkToLink=HashBiMap.create();
	private BiMap<Integer,Integer>numToTimeBean=HashBiMap.create();
	private Network network;
	private Map<Integer,Tuple<Double,Double>>timeBean;

	@Override 
	public void startElement(String uri, String localName, String qName, Attributes attributes) {
		if(qName.equalsIgnoreCase("metaData")) {
			this.kn=Integer.parseInt(attributes.getValue("Kn"));
			this.kt=Integer.parseInt(attributes.getValue("Kt"));
			//this.network=NetworkUtils.readNetwork(attributes.getValue("networkFileLoc"));
			String numToLinkToLink=attributes.getValue("NumToL2l");
			for(String entry:numToLinkToLink.split(",")) {
				int key=Integer.parseInt(entry.split("\t")[0]);
				Id<LinkToLink> l2lId=Id.create(entry.split("\t")[1], LinkToLink.class);
				this.numToLinkToLink.put(key, l2lId);
			}

			String numTotimeBean=attributes.getValue("NumToTimeBean");
			for(String entry:numTotimeBean.split(",")) {
				int key=Integer.parseInt(entry.split("_")[0]);
				int value=Integer.parseInt(entry.split("_")[1]);
				this.numToTimeBean.put(key, value);
			}

//			String timeBean=attributes.getValue("timeBean");
//			for(String entry:timeBean.split(",")) {
//				int key=Integer.parseInt(entry.split("_")[0]);
//				Tuple<Double, Double> value=new Tuple<Double,Double>(Double.parseDouble(entry.split("_")[1].split(" ")[0]),Double.parseDouble(entry.split("_")[1].split(" ")[1]));
//				this.timeBean.put(key, value);
//			}
		}
		if(qName.equalsIgnoreCase("LinkToLink")) {
			Link fromLink=this.network.getLinks().get(Id.createLinkId(attributes.getValue("fromLink")));
			Link toLink=this.network.getLinks().get(Id.createLinkId(attributes.getValue("toLink")));
			LinkToLink l2l=new LinkToLink(fromLink, toLink, timeBean);
			Map<Integer,Set<Integer>> proximityMap=LinkToLink.parseProximityMatrix(attributes.getValue("proximityMap"));
			l2l.setProximityMap(proximityMap);
			l2l.setG_cRatio(Double.parseDouble(attributes.getValue("g_cRatio")));
			l2l.setCycleTime(Double.parseDouble(attributes.getValue("cycleTime")));
			l2l.setSupply(Double.parseDouble(attributes.getValue("Supply")));
			this.linkToLinkMap.put(l2l.getLinkToLinkId(), l2l);
		}
	}

	@Override 
	public void endElement(String uri, String localName, String qName) {

	}

	public void setNetwork(Network network) {
		this.network = network;
	}

	public void setTimeBean(Map<Integer, Tuple<Double, Double>> timeBean) {
		this.timeBean = timeBean;
	}

	public LinkToLinks readLinkToLinks(String fileLoc) {

		LinkToLinksMetaDatahandler mh=new LinkToLinksMetaDatahandler();
		try {
			SAXParserFactory.newInstance().newSAXParser().parse(fileLoc+"/LinkToLinks.xml",mh);
		} catch (SAXException | IOException | ParserConfigurationException e1) {
			// TODO Auto-generated catch block
			e1.printStackTrace();
		}
		this.setTimeBean(mh.getTimeBean());
		this.setNetwork(mh.getNetwork());
		
		try {
			SAXParserFactory.newInstance().newSAXParser().parse(fileLoc+"/LinkToLinks.xml",this);
		} catch (SAXException e) {
			e.printStackTrace();
		} catch (IOException e) {
			e.printStackTrace();
		} catch (ParserConfigurationException e) {
			e.printStackTrace();
		}
		List<LinkToLink> link2Links=new ArrayList<>();
		for(int i=0;i<this.numToLinkToLink.size();i++) {
			link2Links.add(this.linkToLinkMap.get(this.numToLinkToLink.get(i)));
		}

		LinkToLinks linkToLinks=new LinkToLinks(this.network,this.timeBean,this.numToLinkToLink,this.numToTimeBean,link2Links,this.kn,this.kt);
		linkToLinks.setL2lCounter(this.linkToLinkMap.size());
		return linkToLinks;
	}
}

class LinkToLinksMetaDatahandler extends DefaultHandler {
	private Network network;
	private Map<Integer,Tuple<Double,Double>>timeBean=new HashMap<>();
	@Override 
	public void startElement(String uri, String localName, String qName, Attributes attributes) {
		if(qName.equalsIgnoreCase("metaData")) {
			this.network=NetworkUtils.readNetwork(attributes.getValue("networkFileLoc"));
			String timeBean=attributes.getValue("timeBeans");
			for(String entry:timeBean.split(",")) {
				int key=Integer.parseInt(entry.split("_")[0]);
				Tuple<Double, Double> value=new Tuple<Double,Double>(Double.parseDouble(entry.split("_")[1].split(" ")[0]),Double.parseDouble(entry.split("_")[1].split(" ")[1]));
				this.timeBean.put(key, value);
			}
		}
	}
	public Network getNetwork() {
		return network;
	}
	public Map<Integer, Tuple<Double, Double>> getTimeBean() {
		return timeBean;
	}
	
}
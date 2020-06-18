package linktolinkBPR;

import java.io.FileOutputStream;
import java.util.Map.Entry;

import javax.xml.parsers.DocumentBuilder;
import javax.xml.parsers.DocumentBuilderFactory;
import javax.xml.transform.OutputKeys;
import javax.xml.transform.Transformer;
import javax.xml.transform.TransformerFactory;
import javax.xml.transform.dom.DOMSource;
import javax.xml.transform.stream.StreamResult;

import org.matsim.api.core.v01.Id;
import org.matsim.api.core.v01.network.NetworkWriter;
import org.matsim.core.utils.collections.Tuple;
import org.w3c.dom.Document;
import org.w3c.dom.Element;

public class LinkToLinksWriter {
	private LinkToLinks l2ls;
	
	public LinkToLinksWriter(LinkToLinks l2ls){
		this.l2ls=l2ls;
	}
	
	public void write(String fileLoc) {
		try {
			DocumentBuilderFactory documentBuilderFactory = DocumentBuilderFactory.newInstance();
			DocumentBuilder documentBuilder = documentBuilderFactory.newDocumentBuilder();

			Document document = documentBuilder.newDocument();

			Element rootEle = document.createElement("LinkToLinks");
			
			
			//Store the metaData here 
			Element metaData=document.createElement("metaData");
			metaData.setAttribute("Kn", Integer.toString(l2ls.getKn()));
			metaData.setAttribute("Kt", Integer.toString(l2ls.getKt()));
			metaData.setAttribute("networkFileLoc",fileLoc+"/network.xml");
			new NetworkWriter(l2ls.getNetwork()).writeV2(fileLoc+"/network.xml");
			String numTol2l="";
			String entrySeperator="";
			for(Entry<Integer, Id<LinkToLink>> e:l2ls.getNumToLinkToLink().entrySet()) {
				numTol2l=numTol2l+entrySeperator+e.getKey()+"\t"+e.getValue().toString();
				entrySeperator=",";
			}
			metaData.setAttribute("NumToL2l", numTol2l);
			
			entrySeperator="";
			String numTotimeBean="";
			for(Entry<Integer, Integer> e:l2ls.getNumToTimeBean().entrySet()) {
				numTotimeBean=numTotimeBean+entrySeperator+e.getKey()+"_"+e.getValue();
				entrySeperator=",";
			}
			metaData.setAttribute("NumToTimeBean", numTotimeBean);
			
			String timeBeans="";
			entrySeperator="";
			for(Entry<Integer, Tuple<Double,Double>> e:l2ls.getTimeBean().entrySet()) {
				timeBeans=timeBeans+entrySeperator+e.getKey()+"_"+e.getValue().getFirst()+" "+e.getValue().getSecond();
				entrySeperator=",";
			}
			metaData.setAttribute("timeBeans", timeBeans);
			
			rootEle.appendChild(metaData);
			
			for(LinkToLink l2l:this.l2ls.getLinkToLinks()) {
				Element l2lelement=document.createElement("LinkToLink");
				l2lelement.setAttribute("proximityMap", l2l.writeProximityMap());
				l2lelement.setAttribute("pflps", l2l.writePrimaryFromLinkProximitySet());
				l2lelement.setAttribute("fromLink", l2l.getFromLink().getId().toString());
				l2lelement.setAttribute("toLink", l2l.getToLink().getId().toString());
				l2lelement.setAttribute("Supply",Double.toString(l2l.getSupply()));
				l2lelement.setAttribute("Id", l2l.getLinkToLinkId().toString());
				l2lelement.setAttribute("g_cRatio",Double.toString(l2l.getG_cRatio()));
				l2lelement.setAttribute("cycleTime",Double.toString(l2l.getCycleTime()));
				rootEle.appendChild(l2lelement);
			}
			
			document.appendChild(rootEle);
			

			Transformer tr = TransformerFactory.newInstance().newTransformer();
			tr.setOutputProperty(OutputKeys.INDENT, "yes");
			tr.setOutputProperty(OutputKeys.METHOD, "xml");
			tr.setOutputProperty(OutputKeys.ENCODING, "UTF-8");
			tr.setOutputProperty("{http://xml.apache.org/xslt}indent-amount", "4");
			tr.transform(new DOMSource(document), new StreamResult(new FileOutputStream(fileLoc+"/LinkToLinks.xml")));


		}catch(Exception e) {
			
		}
	}
	

}

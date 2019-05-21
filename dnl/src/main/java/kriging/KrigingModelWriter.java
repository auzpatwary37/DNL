package kriging;

import java.io.FileOutputStream;
import java.text.SimpleDateFormat;
import java.util.Calendar;
import java.util.Map;
import java.util.Map.Entry;

import javax.xml.parsers.DocumentBuilder;
import javax.xml.parsers.DocumentBuilderFactory;
import javax.xml.transform.OutputKeys;
import javax.xml.transform.Transformer;
import javax.xml.transform.TransformerFactory;
import javax.xml.transform.dom.DOMSource;
import javax.xml.transform.stream.StreamResult;

import org.apache.commons.math3.linear.RealMatrix;
import org.matsim.core.utils.collections.Tuple;
import org.w3c.dom.Document;
import org.w3c.dom.Element;

public class KrigingModelWriter {
	private final Map<Integer,Tuple<RealMatrix,RealMatrix>> trainingDataSet;
	private final Map<String,RealMatrix> weights;
	private final RealMatrix theta;
	private final RealMatrix beta;
	private final BaseFunction baseFunction;
	
	public KrigingModelWriter(KrigingInterpolator model) {
		this.trainingDataSet=model.getTrainingDataSet();
		this.weights=model.getVariogram().getWeights();
		this.theta=model.getVariogram().gettheta();
		this.beta=model.getBeta();
		this.baseFunction=model.getBaseFunction();
	}
	
	public void writeModel(String fileLoc) {
		try {
			DocumentBuilderFactory documentBuilderFactory = DocumentBuilderFactory.newInstance();
			DocumentBuilder documentBuilder = documentBuilderFactory.newDocumentBuilder();

			Document document = documentBuilder.newDocument();

			Element rootEle = document.createElement("Kriging_DNL");
			
			//Store the metaData here 
			Element metaData=document.createElement("meataData");
			metaData.setAttribute("N", Integer.toString(this.trainingDataSet.get(0).getFirst().getRowDimension()));
			metaData.setAttribute("T", Integer.toString(this.trainingDataSet.get(0).getFirst().getColumnDimension()));
			metaData.setAttribute("I", Integer.toString(this.trainingDataSet.size()));
			metaData.setAttribute("DateAndTime", new SimpleDateFormat("yyyy/MM/dd HH:mm:ss").format(Calendar.getInstance()));
			rootEle.appendChild(metaData);
			
			Element trainingDataSet=document.createElement("trainingDataSet");
			
			for(int i=0;i<this.trainingDataSet.size();i++) {
				Element trainingData=document.createElement("trainingData");
				trainingData.setAttribute("Id", Integer.toString(i));
				trainingData.setAttribute("X",this.trainingDataSet.get(i).getFirst().toString());
				trainingData.setAttribute("Y",this.trainingDataSet.get(i).getSecond().toString());
				trainingDataSet.appendChild(trainingData);
			}
			rootEle.appendChild(trainingDataSet);
			
			Element weights=document.createElement("Weights");
			for(Entry<String,RealMatrix> weight:this.weights.entrySet()) {
				weights.setAttribute(weight.getKey(), weight.getValue().toString());
			}
			rootEle.appendChild(weights);
			
			Element theta=document.createElement("theta");
			theta.setAttribute("theta", this.theta.toString());
			
			rootEle.appendChild(theta);
			
			Element betaEle=document.createElement("beta");
			betaEle.setAttribute("beta", beta.toString());
		
			rootEle.appendChild(betaEle);
			
			Element baseFunction=document.createElement("baseFunction");
			this.baseFunction.writeBaseFunctionInfo(baseFunction);
			document.appendChild(baseFunction);
			
			document.appendChild(rootEle);
			

			Transformer tr = TransformerFactory.newInstance().newTransformer();
			tr.setOutputProperty(OutputKeys.INDENT, "yes");
			tr.setOutputProperty(OutputKeys.METHOD, "xml");
			tr.setOutputProperty(OutputKeys.ENCODING, "UTF-8");
			tr.setOutputProperty("{http://xml.apache.org/xslt}indent-amount", "4");
			tr.transform(new DOMSource(document), new StreamResult(new FileOutputStream(fileLoc)));


		}catch(Exception e) {
			
		}
	}
}

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

import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;
import org.matsim.core.utils.collections.Tuple;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.w3c.dom.Document;
import org.w3c.dom.Element;

import training.DataIO;

public class KrigingModelWriter {
	private final Map<Integer,Tuple<INDArray,INDArray>> trainingDataSet;
	private final Map<String,INDArray> weights;
	private final INDArray theta;
	private final INDArray beta;
	private final BaseFunction baseFunction;
	
	
	public KrigingModelWriter(KrigingInterpolator model) {
		this.trainingDataSet=model.getTrainingDataSet();
		this.weights=model.getVariogram().getWeights();
		this.theta=model.getVariogram().gettheta();
		this.beta=model.getBeta();
		this.baseFunction=model.getBaseFunction();
		
	}
	
	/**
	 * Provide folder location as multiple files will be created.
	 * @param fileLoc
	 */
	public void writeModel(String fileLoc) {
		try {
			DocumentBuilderFactory documentBuilderFactory = DocumentBuilderFactory.newInstance();
			DocumentBuilder documentBuilder = documentBuilderFactory.newDocumentBuilder();

			Document document = documentBuilder.newDocument();

			Element rootEle = document.createElement("Kriging_DNL");
			
			//Store the metaData here 
			Element metaData=document.createElement("meataData");
			metaData.setAttribute("N", Long.toString(this.trainingDataSet.get(0).getFirst().size(0)));
			metaData.setAttribute("T", Long.toString(this.trainingDataSet.get(0).getFirst().size(1)));
			metaData.setAttribute("I", Integer.toString(this.trainingDataSet.size()));
			//metaData.setAttribute("DateAndTime", new SimpleDateFormat("yyyy/MM/dd HH:mm:ss").format(Calendar.getInstance()));
			rootEle.appendChild(metaData);
			
			Element trainingDataSet=document.createElement("trainingDataSet");	
			DataIO.writeData(this.trainingDataSet,fileLoc+"/dataSet.txt");
			trainingDataSet.setAttribute("FileLocation", fileLoc+"/dataSet.txt");
			rootEle.appendChild(trainingDataSet);
			
			Element weights=document.createElement("Weights");
			DataIO.writeWeights(this.weights, fileLoc+"/weights.txt");
			weights.setAttribute("Filelocation", fileLoc+"/weights.txt");
			rootEle.appendChild(weights);
			
			Element theta=document.createElement("theta");
			Nd4j.writeTxt(this.theta, fileLoc+"/theta.txt");
			theta.setAttribute("FilelOcation", fileLoc+"/theta.txt");
			rootEle.appendChild(theta);
			
			Element betaEle=document.createElement("beta");
			Nd4j.writeTxt(this.beta, fileLoc+"/beta.txt");
			theta.setAttribute("FilelOcation", fileLoc+"/beta.txt");
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
			tr.transform(new DOMSource(document), new StreamResult(new FileOutputStream(fileLoc+".xml")));


		}catch(Exception e) {
			
		}
	}
}

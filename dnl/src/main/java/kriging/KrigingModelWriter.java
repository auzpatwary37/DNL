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

import linktolinkBPR.LinkToLinks;
import linktolinkBPR.LinkToLinksWriter;
import training.DataIO;

public class KrigingModelWriter {
	private final KrigingInterpolator krigingModel;
	private final Map<Integer,Data> trainingDataSet;
	private final INDArray theta;
	private final INDArray beta;
	private final BaseFunction baseFunction;
	private INDArray Cn;
	private INDArray Ct;
	private LinkToLinks l2ls;
	private final String n_tSpecificIndices;
	
	public KrigingModelWriter(KrigingInterpolator model) {
		this.krigingModel=model;
		this.trainingDataSet=model.getTrainingDataSet();
		this.theta=model.getVariogram().gettheta();
		this.beta=model.getBeta();
		this.l2ls=model.getVariogram().getL2ls();
		this.baseFunction=model.getBaseFunction();
		this.Cn=model.getCn();
		this.Ct=model.getCt();
		this.n_tSpecificIndices=model.getVariogram().writeN_T_SpecificIndices();
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
			metaData.setAttribute("N", Long.toString(this.trainingDataSet.get(0).getX().size(0)));
			metaData.setAttribute("T", Long.toString(this.trainingDataSet.get(0).getX().size(1)));
			metaData.setAttribute("I", Integer.toString(this.trainingDataSet.size()));
			metaData.setAttribute("TrainingTime", Double.toString(this.krigingModel.getTrainingTime()));
			metaData.setAttribute("n_tSpecificTraningIndices", this.n_tSpecificIndices);
			rootEle.appendChild(metaData);
			
			Element trainingDataSet=document.createElement("trainingDataSet");	
			DataIO.writeData(this.trainingDataSet,fileLoc+"/dataSet.txt",fileLoc+"/keySet.csv");
			trainingDataSet.setAttribute("FileLocation", fileLoc+"/dataSet.txt");
			trainingDataSet.setAttribute("KeyFileLocation", fileLoc+"/keySet.csv");
			rootEle.appendChild(trainingDataSet);
			
			Element Cn=document.createElement("Cn");
			Nd4j.writeTxt(this.Cn, fileLoc+"/Cn.txt");
			Cn.setAttribute("Filelocation", fileLoc+"/Cn.txt");
			rootEle.appendChild(Cn);

			Element Ct=document.createElement("Ct");
			Nd4j.writeTxt(this.Ct, fileLoc+"/Ct.txt");
			Ct.setAttribute("Filelocation", fileLoc+"/Ct.txt");
			rootEle.appendChild(Ct);
			
			Element theta=document.createElement("theta");
			Nd4j.writeTxt(this.theta, fileLoc+"/theta.txt");
			theta.setAttribute("Filelocation", fileLoc+"/theta.txt");
			rootEle.appendChild(theta);
			
			Element betaEle=document.createElement("beta");
			Nd4j.writeTxt(this.beta, fileLoc+"/beta.txt");
			betaEle.setAttribute("Filelocation", fileLoc+"/beta.txt");
			rootEle.appendChild(betaEle);
			
			Element l2ls=document.createElement("LinkToLinks");
			new LinkToLinksWriter(this.l2ls).write(fileLoc);
			l2ls.setAttribute("FileLocation", fileLoc);
			
			Element baseFunction=document.createElement("baseFunction");
			this.baseFunction.writeBaseFunctionInfo(baseFunction,fileLoc);
			//System.out.println();
			rootEle.appendChild(baseFunction);
			
			
			
			rootEle.appendChild(l2ls);
			
			document.appendChild(rootEle);
			

			Transformer tr = TransformerFactory.newInstance().newTransformer();
			tr.setOutputProperty(OutputKeys.INDENT, "yes");
			tr.setOutputProperty(OutputKeys.METHOD, "xml");
			tr.setOutputProperty(OutputKeys.ENCODING, "UTF-8");
			tr.setOutputProperty("{http://xml.apache.org/xslt}indent-amount", "4");
			tr.transform(new DOMSource(document), new StreamResult(new FileOutputStream(fileLoc+"/modelDetails.xml")));


		}catch(Exception e) {
			System.out.println(e);
		}
	}
}

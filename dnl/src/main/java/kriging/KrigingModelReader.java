package kriging;

import java.io.IOException;
import java.lang.reflect.InvocationTargetException;
import java.lang.reflect.Method;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

import javax.xml.parsers.ParserConfigurationException;
import javax.xml.parsers.SAXParserFactory;

import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealMatrixFormat;
import org.matsim.core.utils.collections.Tuple;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.xml.sax.Attributes;
import org.xml.sax.SAXException;
import org.xml.sax.helpers.DefaultHandler;

import linktolinkBPR.LinkToLinks;
import linktolinkBPR.LinkToLinksReader;
import training.DataIO;

public class KrigingModelReader extends DefaultHandler {
	
	private INDArray theta;
	private INDArray beta;
	private INDArray nugget;
	private INDArray Cn;
	private INDArray Ct;
	private Map<Integer,Data> trainingDataSet;
	private LinkToLinks l2ls;
	private int N;
	private int T;
	private int I;
	private BaseFunction bf;
	private double trainingTime=0;
	private Map<String,List<Integer>> n_tSpecificTrainingIndices;

	@Override 
	public void startElement(String uri, String localName, String qName, Attributes attributes) {
		
		if(qName.equalsIgnoreCase("theta")) {
			theta=Nd4j.readTxt(attributes.getValue("Filelocation"));
		}
		
		if(qName.equalsIgnoreCase("nugget")) {
			nugget=Nd4j.readTxt(attributes.getValue("Filelocation"));
		}
		
		if(qName.equalsIgnoreCase("beta")) {
			beta=Nd4j.readTxt(attributes.getValue("Filelocation"));
		}
		
		if(qName.equalsIgnoreCase("meatadata")) {
			N=Integer.parseInt(attributes.getValue("N"));
			T=Integer.parseInt(attributes.getValue("T"));
			I=Integer.parseInt(attributes.getValue("I"));
			this.trainingTime=Double.parseDouble(attributes.getValue("TrainingTime"));
			this.n_tSpecificTrainingIndices=Variogram.parseN_T_SpecificIndicies(attributes.getValue("n_tSpecificTraningIndices"));
		}
		
		if(qName.equalsIgnoreCase("Cn")) {
			String f=attributes.getValue("Filelocation");
			Cn=Nd4j.readTxt(f);
		}
		if(qName.equalsIgnoreCase("Ct")) {
			Ct=Nd4j.readTxt(attributes.getValue("Filelocation"));
		}
		
		if(qName.equalsIgnoreCase("trainingDataSet")) {
			this.trainingDataSet=DataIO.readDataSet(attributes.getValue("FileLocation"),attributes.getValue("KeyFileLocation"));
			
		}
		if(qName.equalsIgnoreCase("LinkToLinks")) {
			this.l2ls=new LinkToLinksReader().readLinkToLinks(attributes.getValue("FileLocation"));
			
		}
		
		
		if(qName.equalsIgnoreCase("baseFunction")) {
			try {
				Method parseBaseFunction=Class.forName(attributes.getValue("ClassName")).getMethod("parseBaseFunction", Attributes.class);
				this.bf=(BaseFunction)parseBaseFunction.invoke(null, attributes);
				
			} catch (NoSuchMethodException | SecurityException | ClassNotFoundException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			} catch (IllegalAccessException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			} catch (IllegalArgumentException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			} catch (InvocationTargetException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
		}
		
		
	}
	
	@Override 
	public void endElement(String uri, String localName, String qName) {
		
	}
	
	public KrigingInterpolator readModel(String fileLoc) {
		
		try {
			SAXParserFactory.newInstance().newSAXParser().parse(fileLoc,this);
		} catch (SAXException e) {
			e.printStackTrace();
		} catch (IOException e) {
			e.printStackTrace();
		} catch (ParserConfigurationException e) {
			e.printStackTrace();
		}
		
		Variogram v=new Variogram(trainingDataSet,this.l2ls,this.theta,this.nugget,this.Cn,this.Ct,this.n_tSpecificTrainingIndices);
		
		KrigingInterpolator kriging=new KrigingInterpolator(v,beta,this.bf,Cn,Ct);
		kriging.setTrainingTime(trainingTime);
		
		return kriging;
	}

	
}

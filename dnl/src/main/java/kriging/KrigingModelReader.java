package kriging;

import java.io.IOException;
import java.lang.reflect.InvocationTargetException;
import java.lang.reflect.Method;
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

import training.DataIO;

public class KrigingModelReader extends DefaultHandler {
	
	private INDArray theta;
	private INDArray beta;
	private Map<String,INDArray> weights;
	private Map<Integer,Tuple<INDArray,INDArray>> trainingDataSet;
	private int N;
	private int T;
	private int I;
	private BaseFunction bf;
	String dateAndTime;
	@Override 
	public void startElement(String uri, String localName, String qName, Attributes attributes) {
		
		if(qName.equalsIgnoreCase("theta")) {
			theta=Nd4j.readTxt(attributes.getValue("FileLocation"));
		}
		
		if(qName.equalsIgnoreCase("beta")) {
			beta=Nd4j.readTxt(attributes.getValue("FileLocation"));
		}
		
		if(qName.equalsIgnoreCase("metadata")) {
			N=Integer.parseInt(attributes.getValue("N"));
			T=Integer.parseInt(attributes.getValue("T"));
			I=Integer.parseInt(attributes.getValue("I"));
		}
		
		if(qName.equalsIgnoreCase("weights")) {
			this.weights=DataIO.readWeight(attributes.getValue("FileLocation"));
		}
		
		if(qName.equalsIgnoreCase("trainingData")) {
			this.trainingDataSet=DataIO.readDataSet(attributes.getValue("FileLocation"));
			
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
	
	public KrigingInterpolator readMeasurements(String fileLoc) {
		
		try {
			SAXParserFactory.newInstance().newSAXParser().parse(fileLoc,this);
		} catch (SAXException e) {
			e.printStackTrace();
		} catch (IOException e) {
			e.printStackTrace();
		} catch (ParserConfigurationException e) {
			e.printStackTrace();
		}
		
		Variogram v=new Variogram(trainingDataSet, this.weights,this.theta);
		
		return new KrigingInterpolator(v,beta,this.bf);
	}

	
}

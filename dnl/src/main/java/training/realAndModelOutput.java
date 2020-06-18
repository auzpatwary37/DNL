package training;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import kriging.KrigingInterpolator;

public class realAndModelOutput{
	private INDArray realOutputVsModelOutput;
	private INDArray realOutput=null;
	private INDArray modelOutput=null;
	private int pointRecorded=0;
	public realAndModelOutput() {
		this.realOutputVsModelOutput=null;
	}
	
	public synchronized void addDatapair(double yReal, double yKriging) {
		if(pointRecorded==0) {
			this.realOutput=Nd4j.create(new double[] {yReal});
			this.modelOutput=Nd4j.create(new double[] {yKriging});
		}else {
			this.realOutput=Nd4j.concat(0, this.realOutput,Nd4j.create(new double[] {yReal}));
			this.modelOutput=Nd4j.concat(0,this.modelOutput,Nd4j.create(new double[] {yKriging}));
		}
		
		this.pointRecorded++;
		
		if(this.modelOutput.size(0)!=this.realOutput.size(0) || this.modelOutput.size(0)!=this.pointRecorded ) {
			throw new IllegalArgumentException("Debug Point!!!");
		}
	}

	public synchronized void addDatapair(INDArray real,INDArray model) {
		INDArray realvector=real.reshape(real.length());
		INDArray modelvector=model.reshape(model.length());
		INDArray ar=Nd4j.create(realvector.length(),2);
		ar.putColumn(0, realvector);
		ar.putColumn(1, modelvector);
		if(pointRecorded==0) {
			this.realOutputVsModelOutput=ar;
		}else {
			this.realOutputVsModelOutput=Nd4j.concat(0, this.realOutputVsModelOutput,ar);
		}
		this.pointRecorded++;
	}
	
	
	public void writeCsv(String fileLoc) {
		if(this.realOutputVsModelOutput==null) {
			this.realOutputVsModelOutput=Nd4j.create(this.realOutput.size(0),2);
			this.realOutputVsModelOutput.putColumn(0, this.realOutput);
			this.realOutputVsModelOutput.putColumn(1, this.modelOutput);
		}
		KrigingInterpolator.writeINDArray(this.realOutputVsModelOutput, fileLoc);
	}
	
}
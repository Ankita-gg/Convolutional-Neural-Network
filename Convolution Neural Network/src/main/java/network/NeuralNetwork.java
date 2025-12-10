package network;

import java.util.ArrayList;
import java.util.List;

import dataForJava.Image;
import dataForJava.MatrixUtility;
import layer.Layers;

public class NeuralNetwork {
	List<Layers> _layer;
	double scalefactor;
	
	 public NeuralNetwork(List<Layers> _layer,double scalefactor) {
			super();
			this._layer = _layer;
			this.scalefactor=scalefactor;
			linklayers();
		 }
	private void linklayers()
	{
		if(_layer.size()<=1)
			return;
		for(int i=0;i<_layer.size();i++)
		{
			if(i==0)
			{
				_layer.get(i).setNextlayer(_layer.get(i+1));
			}
			else if(i==_layer.size()-1)
			{
				_layer.get(i).setPreviouslayer(_layer.get(i-1));
			}
			else
			{
				_layer.get(i).setPreviouslayer(_layer.get(i-1));
				_layer.get(i).setNextlayer(_layer.get(i+1));
			}
		}
	}
	
	public double[] getError(double[]networkoutput,int correctanswer)
	{
		int numclasses=networkoutput.length;
		double expected[]=new double[numclasses];
		expected[correctanswer]=1;
		return MatrixUtility.add(networkoutput,MatrixUtility.multiply(expected,-1));
	}
	
	 private int getMaxIndex(double[] in){

	        double max = 0;
	        int index = 0;

	        for(int i = 0; i < in.length; i++){
	            if(in[i] >= max){
	                max = in[i];
	                index = i;
	            }

	        }

	        return index;
	    }
	 
	 public int guess(Image image){
	        List<double[][]> inList = new ArrayList<>();
	        inList.add(MatrixUtility.multiply(image.getData(), (1.0/scalefactor)));

	        double[] out = _layer.get(0).getoutput(inList);
	        int guess = getMaxIndex(out);

	        return guess;
	    }
	 
	 public float test (List<Image> images){
	        int correct = 0;

	        for(Image img: images){
	            int guess = guess(img);

	            if(guess == img.getLabel()){
	                correct++;
	            }
	        }

	        return((float)correct/images.size());
	    }
	 
	 public void train (List<Image> images){

	        for(Image img:images){
	            List<double[][]> inList = new ArrayList<>();
	            inList.add(MatrixUtility.multiply(img.getData(), (1.0/scalefactor)));

	            double[] out = _layer.get(0).getoutput(inList);
	            double[] dldO = getError(out, img.getLabel());

	            _layer.get((_layer.size()-1)).backpropagation(dldO);
	        }

	    }

	
	
	

}

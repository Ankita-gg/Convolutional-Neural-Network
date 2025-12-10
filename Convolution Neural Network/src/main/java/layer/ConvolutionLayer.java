package layer;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import dataForJava.MatrixUtility;

public class ConvolutionLayer extends Layers {
	private long SEED;
	
	private List<double[][]> filters;
	private int filtersize;
	private int stepsize;
	
	private int  inlength;
	private int inrows;
	private int incols;
	private double learningrate;
	
	
	private List<double[][]> lastinput;

	public ConvolutionLayer( int filtersize, int stepsize, int inlength, int inrows,int intcols,long sEED,int numFilters,double learningrate) {
		SEED = sEED;
		this.filtersize = filtersize;
		this.stepsize = stepsize;
		this.inlength = inlength;
		this.inrows = inrows;
		this.incols = intcols;
		this.learningrate=learningrate;
		
		generateRandomFilters(numFilters);
	}
	private void generateRandomFilters(int numFilters)
	{
		List<double[][]> filters1=new ArrayList<>();
		Random random=new Random(SEED);
		
		for(int n=0;n<numFilters;n++)
		{
			double[][] newFilter=new double[filtersize][filtersize];
			for(int i=0;i<filtersize;i++)
			{
				for(int j=0;j<filtersize;j++)
				{
					double value=random.nextGaussian();
					newFilter[i][j]=value;
				}
			}
			filters1.add(newFilter);
		}
		filters=filters1;
	}
	public List<double[][]> convolutionForwardPass(List<double[][]> list)
	{
		List<double[][]> output=new ArrayList<>();
		lastinput=list;
		for(int m=0;m<list.size();m++)
		{
			for(double[][] filter:filters)
			{
				output.add(convolve(list.get(m),filter,stepsize));
			}
		}
		return output;
	}
	private double[][] convolve(double[][]input,double[][]filter,int stepsize)
	{
		int outrows=(input.length-filter.length)/stepsize+1;
		int outCols=(input[0].length-filter[0].length)/stepsize+1;
		
		int inrows=input.length;
		int incols=input[0].length;
		
		int frows=filter.length;
		int fcols=filter[0].length;
		
		double[][]output =new double[outrows][outCols];
		
		int outRow=0;
		int outcol;
		
		for(int i=0;i<=inrows-frows;i+=stepsize)
		{
			outcol=0;
			for(int j=0;j<=incols-fcols;j+=stepsize)
			{
				double sum=0.0;
				for(int x=0;x<frows;x++)
				{
					for(int y=0;y<fcols;y++)
					{
						int inputrowindex=i+x;
						int inputcolindex=j+y;
						
						double value=filter[x][y]*input[inputrowindex][inputcolindex];
						sum+=value;
					}
				}
				output[outRow][outcol]=sum;
				outcol++;
			}
			outRow++;
		}
		return output;
	}
	
	public double[][] spacearray(double[][]input)
	{
		if(stepsize==1)
			return input;
		int outrows=(input.length-1)*stepsize+1;
		int outcols=(input[0].length-1)*stepsize+1;
		double output[][]=new double[outrows][outcols];
		
		for(int i=0;i<input.length;i++)
		{
			for(int j=0;j<input[0].length;j++)
			{
				output[i*stepsize][j*stepsize]=input[i][j];
			}
		}
		return output;
			
	}
	
	private double[][] fullconvolve(double[][]input,double[][]filter)
	{
		int outrows=(input.length+filter.length)+1;
		int outCols=(input[0].length+filter[0].length)+1;
		
		int inrows=input.length;
		int incols=input[0].length;
		
		int frows=filter.length;
		int fcols=filter[0].length;
		
		double[][]output =new double[outrows][outCols];
		
		int outRow=0;
		int outcol;
		
		for(int i=-frows+1;i<=inrows;i++)
		{
			outcol=0;
			for(int j=-fcols+1;j<=incols;j++)
			{
				double sum=0.0;
				for(int x=0;x<frows;x++)
				{
					for(int y=0;y<fcols;y++)
					{
						int inputrowindex=i+x;
						int inputcolindex=j+y;
						
						if(inputrowindex>=0 && inputcolindex >=0 && inputrowindex<inrows && inputcolindex<incols)
						{
							double value=filter[x][y]*input[inputrowindex][inputcolindex];
							sum+=value;	
						}
						
						
					}
				}
				output[outRow][outcol]=sum;
				outcol++;
			}
			outRow++;
		}
		return output;
	}

	@Override
	public double[] getoutput(List<double[][]> input) {
		List<double[][]>output=convolutionForwardPass(input);
		
		return nextlayer.getoutput(output);
	}

	@Override
	public double[] getoutput(double[] input) {
		// TODO Auto-generated method stub
		List<double[][]>matrixInput=VectorToMatrix(input,inlength,inrows,incols);
		return getoutput(matrixInput);
	}

	@Override
	public void backpropagation(double[] dldo) {
		// TODO Auto-generated method stub
		List<double[][]> matrixinput=VectorToMatrix(dldo,inlength,inrows,incols);
		backpropagation(matrixinput);
		
	}

	@Override
	public void backpropagation(List<double[][]> dldo) {
		List<double[][]> filtersdelta=new ArrayList<>();
		List<double[][]>dldopreviouslayer=new ArrayList<>();
		for(int f=0;f<filters.size();f++)
		{
			filtersdelta.add(new double[filtersize][filtersize]);
		}
		
		for(int i=0;i<lastinput.size();i++)
		{
			double[][]errorforinput=new double[inrows][incols];
			for(int f=0;f<filters.size();f++)
			{
				double[][] currFilter=filters.get(f);
				double[][] error=dldo.get(i*filters.size()+f);
				double spacedError[][]=spacearray(error);
				double[][]dldf=convolve(lastinput.get(i),spacedError,1);
				double[][]delta=MatrixUtility.multiply(dldf,learningrate*-1);
				double[][]newTotalDelta=MatrixUtility.add(filtersdelta.get(f), delta);
				filtersdelta.set(f, newTotalDelta);
				double[][]flippederror=flipArrayHorizontal(flipArrayVertical(spacedError));
				errorforinput=MatrixUtility.add(errorforinput,fullconvolve(currFilter,flippederror));
			}
			dldopreviouslayer.add(errorforinput);
		}
		for(int f=0;f<filters.size();f++)
		{
			double modified[][]=MatrixUtility.add(filtersdelta.get(f),filters.get(f));
			filters.set(f, modified);
		}
		if(previouslayer!=null)
		{
			previouslayer.backpropagation(dldopreviouslayer);
		}
		
	}
	
	 public double[][] flipArrayHorizontal(double[][] array){
	        int rows = array.length;
	        int cols = array[0].length;

	        double[][] output = new double[rows][cols];

	        for(int i = 0; i < rows; i++){
	            for(int j = 0; j < cols; j++){
	                output[rows-i-1][j] = array[i][j];
	            }
	        }
	        return output;
	    }

	    public double[][] flipArrayVertical(double[][] array){
	        int rows = array.length;
	        int cols = array[0].length;

	        double[][] output = new double[rows][cols];

	        for(int i = 0; i < rows; i++){
	            for(int j = 0; j < cols; j++){
	                output[i][cols-j-1] = array[i][j];
	            }
	        }
	        return output;
	    }
	    
	    
	   

	@Override
	public int getoutputlength() {
		// TODO Auto-generated method stub
		return filters.size()*inlength;
	}

	@Override
	public int getoutputrows() {
		// TODO Auto-generated method stub
		return (inrows-filtersize)/stepsize+1;
	}

	@Override
	public int getoutputcolumns() {
		// TODO Auto-generated method stub
		return (incols-filtersize)/stepsize+1;
	}

	@Override
	public int getoutputelements() {
		// TODO Auto-generated method stub
		return getoutputcolumns()*getoutputrows()*getoutputlength();
	}

}

package layer;

import java.util.ArrayList;
import java.util.List;

public class MaxPoolLayer extends Layers{
	private int stepsize;
	private int windowsize;
	
	private int inlength;
	private int inrows;
	private int inCols;
	
	List <int[][]> lastMaxRows;
	List <int[][]> lastMaxColumns;
	

	

	public MaxPoolLayer(int stepsize, int windowsize, int inlength, int inrows, int inCols) {
		super();
		this.stepsize = stepsize;
		this.windowsize = windowsize;
		this.inlength = inlength;
		this.inrows = inrows;
		this.inCols = inCols;
	}
	public List<double[][]> maxPoolForwardPass(List<double[][]> input)
	{
		List<double[][]> output=new ArrayList<>();
		lastMaxRows=new ArrayList<>();
		lastMaxColumns=new ArrayList<>();
		for(int l=0;l<input.size();l++)
		{
			output.add(pool(input.get(l)));
			
		}
		return output;
	}
  
	public double[][] pool(double input[][])
	{
		double[][] output=new double[getoutputrows()][getoutputcolumns()];
		int maxrows[][]=new int[getoutputrows()][getoutputcolumns()];
		int maxcolumns[][]=new int[getoutputrows()][getoutputcolumns()];
		for(int r=0;r<getoutputrows();r+=stepsize)
		{
			for(int c=0;c<getoutputcolumns();c+=stepsize)
			{
				double max=0.0;
				maxrows[r][c]=-1;
				maxcolumns[r][c]=-1;
				
				for(int x=0;x<windowsize;x++)
				{
					for(int y=0;y<windowsize;y++)
					{
						if(max<input[r+x][c+y])
						{
							max=input[r+x][c+y];
							maxrows[r][c]=r+x;
							maxcolumns[r][c]=c+y;
						}
					}
				}
				output[r][c]=max;
			}
		}
		lastMaxRows.add(maxrows);
		lastMaxColumns.add(maxcolumns);
		return output;
	}
	
	@Override
	public double[] getoutput(List<double[][]> input) {
		// TODO Auto-generated method stub
		List<double[][]> outputPool=maxPoolForwardPass(input);
		return nextlayer.getoutput(outputPool);
	}
	@Override
	public double[] getoutput(double[] input) {
		List<double[][]> matrixList=VectorToMatrix(input,inlength,inrows,inCols);
		return getoutput(matrixList);
	}

	@Override
	public void backpropagation(double[] dldo) {
		// TODO Auto-generated method stub
		List<double[][]> matrixList=VectorToMatrix(dldo,getoutputlength(),getoutputrows(),getoutputcolumns());
		backpropagation(matrixList);
		
	}

	@Override
	public void backpropagation(List<double[][]> dldo) {
		List<double[][]>dxdl =new ArrayList<>();
		int l=0;
		for(double[][] array:dldo)
		{
			double error[][]=new double[inrows][inCols];
			for(int r=0;r<getoutputrows();r++)
			{
				for(int c=0;c<getoutputcolumns();c++)
				{
					int max_i=lastMaxRows.get(l)[r][c];
					int max_j=lastMaxColumns.get(l)[r][c];
					
					if(max_i!=-1)
					{
						error[max_i][max_j]+=array[r][c];
					}
					
				}
			}
			dxdl.add(error);
			l++;
		
		}
		if(previouslayer!=null)
		{
			previouslayer.backpropagation(dxdl);
		}
		
	}

	@Override
	public int getoutputlength() {
		// TODO Auto-generated method stub
		return inlength;
	}

	@Override
	public int getoutputrows() {
		// TODO Auto-generated method stub
		return (inrows-windowsize)/stepsize+1;
	}

	@Override
	public int getoutputcolumns() {
		// TODO Auto-generated method stub
		return (inCols-windowsize)/stepsize+1;
	}

	@Override
	public int getoutputelements() {
		// TODO Auto-generated method stub
		return inlength*getoutputcolumns()*getoutputrows();
	}

}

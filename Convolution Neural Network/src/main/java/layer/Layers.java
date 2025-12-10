package layer;

import java.util.ArrayList;
import java.util.List;

public abstract class Layers {
protected Layers nextlayer;
protected Layers previouslayer;

public abstract double[] getoutput(List<double[][]> input);
public abstract double[] getoutput(double[] input);

public abstract void backpropagation(double[] dldo);
public abstract void backpropagation(List<double[][]> dldo);

public abstract int getoutputlength();
public abstract int getoutputrows();
public abstract int getoutputcolumns();
public abstract int getoutputelements();

public Layers getNextlayer() {
	return nextlayer;
}
public void setNextlayer(Layers nextlayer) {
	this.nextlayer = nextlayer;
}
public Layers getPreviouslayer() {
	return previouslayer;
}
public void setPreviouslayer(Layers previouslayer) {
	this.previouslayer = previouslayer;
}
public double[] MatrixToVector(List<double[][]> input)
{
	int length=input.size();
	int rows=input.get(0).length;
	int columns=input.get(0)[0].length;
	double vector[]=new double[length*rows*columns];
	int i=0;
	
	for(int l=0;l<length;l++)
	{
		for(int r=0;r<rows;r++)
		{
			for(int c=0;c<columns;c++)
			{
				vector[i]=input.get(l)[r][c];
				i++;
			}
		}
	}
	return vector;
}

public List<double[][]> VectorToMatrix(double[]a,int length,int rows,int columns)
{
	List<double[][]> output =new ArrayList<>();
	int i=0;
	for(int l=0;l<length;l++)
	{
		double matrix[][]=new double[rows][columns];
		for(int r=0;r<rows;r++)
		{
			for(int c=0;c<columns;c++)
			{
				matrix[r][c]=a[i];
				i++;
			}
		}
		output.add(matrix);
	}
	return output;
}
}

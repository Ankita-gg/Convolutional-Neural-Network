package layer;

import java.util.List;
import java.util.Random;

public class FullyConnectedLayers extends Layers {

	private double[][] weights;
	private int inlength;
	private int outlength;
	private long SEED;
	private double[] lastZ;
	private double[] lastX;
	private final double leak=0.01;
	private double learningrate;
	
	public FullyConnectedLayers(int inlength, int outlength,long s,double learningrate) {
		super();
		this.inlength = inlength;
		this.outlength = outlength;
		this.SEED=s;
		this.learningrate=learningrate;
		weights=new double[inlength][outlength];
		
		setRandomWeights();
	}
	
	public double[] fullyConnectedForwardPass(double[] input) {
		
		lastX=input;
		
		double[] Z =new double[outlength];
		double[] out=new double[outlength];
		
		for(int i=0;i<inlength;i++)
		{
			for(int j=0;j<outlength;j++)
			{
				Z[j]+=input[i]*weights[i][j];
			}
		}
		lastZ=Z;
		
		
			for(int j=0;j<outlength;j++)
			{
				out[j]=relu(Z[j]);
			}
			
			return out;
		
	}

	@Override
	public double[] getoutput(List<double[][]> input) {
		double[] vector=MatrixToVector(input);
		return getoutput(vector);
	}

	@Override
	public double[] getoutput(double[] input) {
		double[] forwardPass = fullyConnectedForwardPass(input);
		if(nextlayer!=null)
		{
			return nextlayer.getoutput(forwardPass);		
		}
		else
			return forwardPass;
	    }

	@Override
	public void backpropagation(double[] dldo) {
		double dldx[]=new double[inlength];
		double dodz;
		double dzdw;
		double dldw;
		double dzdx;
		for (int k=0;k<inlength;k++)
		{
			double dldx_sum=0;
			for(int j=0;j<outlength;j++)
			{
				dodz=derivativerelu(lastZ[j]);
				dzdw=lastX[k];
				dldw=dldo[j]*dodz*dzdw;
				dzdx=weights[k][j];
				weights[k][j]-=dldw*learningrate;
				dldx_sum+=dldo[j]*dodz*dzdx;
			}
			dldx[k]=dldx_sum;
		}
		if(previouslayer!=null)
		{
			previouslayer.backpropagation(dldx);
		}
		
	}

	@Override
	public void backpropagation(List<double[][]> dldo) {
		double vector[]=MatrixToVector(dldo);
		backpropagation(vector);
		
	}

	@Override
	public int getoutputlength() {
		// TODO Auto-generated method stub
		return 0;
	}

	@Override
	public int getoutputrows() {
		// TODO Auto-generated method stub
		return 0;
	}

	@Override
	public int getoutputcolumns() {
		// TODO Auto-generated method stub
		return 0;
	}

	@Override
	public int getoutputelements() {
		// TODO Auto-generated method stub
		return outlength;
	}
	
	public void setRandomWeights()
	{
		Random random=new Random(SEED);
		
		for(int i=0;i<inlength;i++)
		{
			for(int j=0;j<outlength;j++)
			{
				weights[i][j]=random.nextGaussian();
			}
		}
	}
	
	public double relu(double input)
	{
		if(input<=0)
			return 0;
		else
			return input;
	}
	
	public double derivativerelu(double input)
	{
		if(input<=0)
			return leak;
		else
			return 1;
	}

}

package dataForJava;

public class Image {
	private double[][]data;
	private int label;
	
	public double[][] getData() {
		return data;
	}

	public void setData(double[][] data) {
		this.data = data;
	}

	public int getLabel() {
		return label;
	}

	public void setLabel(int label) {
		this.label = label;
	}

	public Image(double[][]data,int label)
	{
		this.data=data;
		this.label=label;
	}
  
	public String toString()
	{
		String s= label + ", \n";
		for(int i=0;i<data.length;i++)
		{
			for(int j=0;j<data[0].length;j++) {
				if(data[i][j]==0)
					s+="\t";
				else
				  s+=data[i][j]+ ",";
			}
			s+="\n";
		}
		return s;
	}
}

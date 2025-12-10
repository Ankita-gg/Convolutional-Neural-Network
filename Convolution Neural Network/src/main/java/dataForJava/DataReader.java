package dataForJava;

import java.io.BufferedReader;
import java.io.FileReader;
import java.util.ArrayList;
import java.util.List;

public class DataReader {
private final int rows=28;
private final int columns=28;
 
public List<Image> readData(String path)
{
	List<Image> images=new ArrayList<>();
	try(BufferedReader dataReader = new BufferedReader(new FileReader(path))){
		String line;
		while((line =dataReader.readLine())!=null){
			String[] lineitems = line.split(",");
			double[][]data =new double[rows][columns];
			int  label=Integer.parseInt(lineitems[0]);
			
			int i=1;
			
			for(int j=0;j<rows;j++)
			{
				for(int k=0;k<columns;k++)
				{
					data[j][k]=(double)Integer.parseInt(lineitems[i]);
					i++;
				}
			}
			images.add(new Image(data,label));
		}
		}
		
	catch (Exception e) {
		   System.out.println(e);
		
	}
	return images;
}
}

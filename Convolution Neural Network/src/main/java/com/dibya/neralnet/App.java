package com.dibya.neralnet;

import java.net.URLDecoder;
import java.util.*;
import java.nio.charset.StandardCharsets;
import java.util.List;

import dataForJava.DataReader;
import dataForJava.Image;
import network.NetworkBuilder;
import network.NeuralNetwork;

public class App {
	public static void main(String[] args) {
		long seed = 123;
		System.out.println("Stsring data Loading....");
		String path_train = URLDecoder.decode(App.class.getResource("/mnist_train.csv").getPath(),
				StandardCharsets.UTF_8);
		String path_test = URLDecoder.decode(App.class.getResource("/mnist_test.csv").getPath(),
				StandardCharsets.UTF_8);

		System.out.println(path_train);
		List<Image> images_train = new DataReader().readData(path_train);
		List<Image> images_test = new DataReader().readData(path_test);

		System.out.println("Image Train Size: " + images_train.size());
		System.out.println("Image Test Size: " + images_test.size());
		/*
		 * System.out.println(images.get(0).toString());
		 * System.out.println(images2.get(0).toString());
		 * 
		 * 
		 */

		NetworkBuilder builder = new NetworkBuilder(28, 28, 256 * 100);
		builder.addConvolutionLayer(8, 5, 1, 0.1, seed);
		builder.addMaxPoolLayer(3, 2);
		builder.addFullyConnectedLayer(10, 0.1, seed);

		NeuralNetwork net = builder.build();

		
		float rate = net.test(images_test);
		System.out.println("Pre training success rate: " + rate);

		int epochs = 5;

		for (int i = 0; i < epochs; i++) {
			Collections.shuffle(images_train);
			net.train(images_train);
			rate = net.test(images_test);
			System.out.println("Success rate after round " + i + ": " + rate);
		}
	}
}

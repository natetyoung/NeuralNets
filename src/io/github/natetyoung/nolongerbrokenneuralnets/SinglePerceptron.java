package io.github.natetyoung.nolongerbrokenneuralnets;

import java.util.Arrays;

public class SinglePerceptron {
	public double[] weights;
	public double c;
	public SinglePerceptron(int n){
		weights = new double[n];
		for(int i=0;i<n;i++){
			weights[i]=Math.random()-.5;
		}
		this.c=.01;
	}
	public double train(double[] inputs, double expected){
		double output = output(inputs);
		if(output==expected) return output;
		for(int i=0;i<weights.length;i++){
			weights[i]+=c*(expected-output)*(i<inputs.length?inputs[i]:1);
		}
		return output;
	}
	public double output(double[] inputs){
		double s=0;
		for(int i=0;i<weights.length;i++){
			s+=weights[i]*(i<inputs.length?inputs[i]:1);
		}
		return s>0 ? 1 : 0;
	}
	public double sigmoid(double z){
		return 1/(1+Math.exp(z));
	}
	public static void main(String[] args) {
		SinglePerceptron p = new SinglePerceptron(2);
		for(int i=0;i<1000;i++){
			double[] in = {Math.random(),Math.random()};
			int exp = ((in[1]>in[0])? 1:0);
			System.out.println(exp==(int)p.train(in, exp));
			System.out.println(Arrays.toString(p.weights));
		}
	}
}

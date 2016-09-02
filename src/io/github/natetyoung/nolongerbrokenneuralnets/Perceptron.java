package io.github.natetyoung.nolongerbrokenneuralnets;

import java.util.Arrays;
/**
 * Neuron / Node class for use with <code>Network</code>
 */
public class Perceptron {
	/**
	 * The most recent error of the neuron, computed during backpropagation
	 */
	public double err;
	/**
	 * The most recent output of the neuron, computed during feedforward
	 */
	public double a;
	/**
	 * An enum of the possible activation functions for a <code>Perceptron</code>
	 */
	public static enum Function {LIN,SIG};
	/**
	 * The activation function used by this <code>Perceptron</code>, defaults to the sigmoid function
	 */
	public Function fn;
	public Perceptron(){
		fn = Function.SIG;
	}
	/**
	 * Compute, store and return the output of this neuron with the given inputs and weights feeding into it
	 * @param inputs The inputs to this neuron - usually the outputs of the previous layer in the network
	 * @param weights The weights of the connections that feed into this neuron
	 */
	public double output(double[] inputs, double[] weights){
		double z=0;
		for(int i=0;i<weights.length;i++){
			z+=weights[i]*(i<inputs.length?inputs[i]:1);
		}
		a=(fn==Function.SIG?sigmoid(z):z);
		return a;
	}
	/**
	 * Compute, store and return the error of this neuron with the given errors and weights of the layer after it
	 * @param errors The errors of the neurons in the next layer of the network
	 * @param weights The weights of the connections that feed out of this neuron
	 */
	public double error(double[] errors, double[] weights){
		double e=0;
		for(int i=0;i<errors.length; i++){
			e+=errors[i]*weights[i];
		}
		e*=(fn==Function.SIG? a*(1-a) : 1);
		err=e;
		return e;
	}
	/**
	 * The sigmoid function
	 */
	public double sigmoid(double z){
		return 1/(1+Math.exp(-.05*z));
	}
	/**
	 * The derivative of the sigmoid function
	 */
	public double dsigmoid(double z)
	{
		return sigmoid(z)*(1-sigmoid(z));
	}
}

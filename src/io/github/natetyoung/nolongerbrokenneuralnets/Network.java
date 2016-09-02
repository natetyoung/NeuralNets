package io.github.natetyoung.nolongerbrokenneuralnets;

import java.util.Arrays;

public class Network {
	/**
	 * Weights organized as [layer to][node to][node from]
	 */
	public double[][][] weights;
	/**
	 * Most recent delta-weights organized as [layer to][node to][node from]
	 */
	public double[][][] dweights;
	/**
	 * Neurons in network, arranged as [layer][node]
	 */
	public Perceptron[][] nodes;
	/**
	 * Learning constant, some small number.
	 * Smaller means the network will be slower to train
	 * Larger means the network may overshoot the correct parameter values
	 */
	public double lc=.1;
	/**
	 * Momentum term; fraction of the previous delta-weight incorporated into the new one
	 * If high, weights will tend to move in one direction, and change direction slowly
	 * Used to prevent the network from getting stuck in local minima
	 */
	public double momentum = 0.9;
	/**
	 * Create a network with the given topology
	 * @param numNodes An array indicating the number of nodes in each layer; should start with number of inputs and end with number of outputs
	 */
	public Network(int[] numNodes){
		weights = new double[numNodes.length][][];
		dweights = new double[numNodes.length][][];
		nodes = new Perceptron[numNodes.length][];
		for(int i=0; i<numNodes.length; i++){
			nodes[i] = new Perceptron[numNodes[i]];
			for(int j=0; j<nodes[i].length; j++){
				nodes[i][j] = new Perceptron();
				if(j==nodes[i].length-1){
					nodes[i][j].fn = Perceptron.Function.LIN;
				}
			}
			weights[i] = new double[numNodes[i]][i>=1? numNodes[i-1] : 1];
			dweights[i] = new double[numNodes[i]][i>=1? numNodes[i-1] : 1];
			for(int j=0; j<weights[i].length; j++){
				for(int k=0; k<weights[i][j].length; k++){
					weights[i][j][k] = (i==0)? 1 : Math.random()-.5;
				}
			}
		}
	}
	/**
	 * Computes the output of the network on a given input
	 * @param inputs The inputs to be fed to the network
	 * @return The output of the network
	 */
	public double[] feedForward(double[] inputs){
		double[] outputs = inputs;
		for(int i=0; i<inputs.length; i++){
			nodes[0][i].a = inputs[i];
		}
		for(int i=1; i<nodes.length; i++){
			outputs = new double[nodes[i].length];
			for(int j=0; j<nodes[i].length; j++){
				outputs[j] = nodes[i][j].output(inputs, weights[i][j]);
			}
			inputs = outputs;
		}
		return outputs;
	}
	/**
	 * Computes the error at each node without actually changing any weights
	 * @param expected The expected result ("real" output) for the most recent feedforward cycle
	 */
	public void backpropagate(double[] expected){
		for(int i=0; i<nodes[nodes.length-1].length; i++){
			nodes[nodes.length-1][i].error(new double[]{expected[i]-nodes[nodes.length-1][i].a}, new double[]{1});
		}
		for(int l=nodes.length-2; l>=0; l--){
			double[] err = new double[nodes[l+1].length];
			for(int j=0; j<err.length; j++){
				err[j] = .5*nodes[l+1][j].err;
			}
			for(int j=0; j<nodes[l].length; j++){
				double[] w = new double[err.length];
				for(int k=0; k<w.length; k++){
					w[k] = weights[l+1][k][j];
				}
				nodes[l][j].error(err, w);
			}
		}
	}
	/**
	 * Changes the weights of the network based on the learning constant and the latest outputs and errors of the nodes in the network
	 * 
	 */
	public void changeWeights(){
		for(int l=1; l<weights.length; l++){
			for(int i=0; i<weights[l].length; i++){
				for(int j=0; j<weights[l][i].length; j++){
					dweights[l][i][j]=(lc*(nodes[l-1][j].a*nodes[l][i].err)+dweights[l][i][j]*momentum);
					weights[l][i][j] += dweights[l][i][j];
				}
			}
		}
	}
	/**
	 * Convenience method to train the network once on the given training example - feedforward, backpropagate, change weights
	 * @param inputs The input values of the training example, to be passed to <code>feedforward</code>
	 * @param expected The expected output value of the network, to be passed to <code>backpropagate</code>
	 * @return The output of the network on the inputs
	 */
	public double[] train(double[] inputs, double[] expected){
		double[] res = feedForward(inputs);
		backpropagate(expected);
		changeWeights();
		return res;
	}
	/**
	 * An example main method that trains the network on XOR over 10k iterations and prints ~10% of the results - whether or not the network was correct
	 */
	public static void main(String[] args) {
		Network n = new Network(new int[]{2,3,1});
		System.out.println(Arrays.deepToString(n.weights));
		for(int i=2; i<10000; i++){
			double[] in = {Math.round(Math.random()), Math.round(Math.random())};
			int exp = (((int)(in[0]+in[1]))==1? 1:0);
			String toPrint = (exp==(int)(Math.round(n.train(in, new double[]{exp})[0])))? "YES": "NO";
			if(Math.random()>.9) System.out.println(toPrint);
		}
		System.out.println(Arrays.deepToString(n.weights));
	}
}

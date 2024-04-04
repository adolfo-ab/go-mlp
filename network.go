package main

import "gonum.org/v1/gonum/mat"

type NeuralNetwork struct {
	config  NeuralNetworkConfig
	wHidden *mat.Dense
	bHidden *mat.Dense
	wOutput *mat.Dense
	bOutput *mat.Dense
}

type NeuralNetworkConfig struct {
	numInputNeurons  int
	numHiddenNeurons int
	numOutputNeurons int
	numEpochs        int
	learningRate     float64
}

package main

import (
	"gonum.org/v1/gonum/mat"
	"math"
)

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

func NewNeuralNetwork(config NeuralNetworkConfig) *NeuralNetwork {
	return &NeuralNetwork{config: config}
}

func relu(x float64) float64 {
	return math.Max(0, x)
}

func reluDerivative(x float64) float64 {
	if x > 0 {
		return 1
	}
	return 0
}

func softmax(x []float64) []float64 {
	sumExp := 0.0
	out := make([]float64, len(x))

	for _, value := range x {
		sumExp += math.Exp(value)
	}

	for i, value := range x {
		out[i] = math.Exp(value) / sumExp
	}
	return out
}

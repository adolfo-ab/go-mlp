package main

import (
	"gonum.org/v1/gonum/mat"
	"math"
	"math/rand"
	"time"
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
	nn := &NeuralNetwork{config: config}
	nn.wHidden = mat.NewDense(config.numInputNeurons, config.numHiddenNeurons, nil)
	nn.bHidden = mat.NewDense(1, config.numHiddenNeurons, nil)
	nn.wOutput = mat.NewDense(config.numHiddenNeurons, config.numOutputNeurons, nil)
	nn.bOutput = mat.NewDense(1, config.numOutputNeurons, nil)
	nn.initializeWeightsAndBiases()
	return nn
}

func (nn *NeuralNetwork) initializeWeightsAndBiases() {
	randSource := rand.NewSource(time.Now().UnixNano())
	randGen := rand.New(randSource)

	wHiddenRaw := nn.wHidden.RawMatrix().Data
	bHiddenRaw := nn.bHidden.RawMatrix().Data
	wOutputRaw := nn.wOutput.RawMatrix().Data
	bOutputRaw := nn.bOutput.RawMatrix().Data
	for _, value := range [][]float64{wHiddenRaw, bHiddenRaw, wOutputRaw, bOutputRaw} {
		for i := range value {
			value[i] = randGen.Float64()
		}
	}
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

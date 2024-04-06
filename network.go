package main

import (
	"errors"
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

func (nn *NeuralNetwork) forwardPass(input *mat.Dense, yTrue *mat.Dense) (*mat.Dense, float64, error) {
	// Compute the input to the hidden layer
	hiddenInput := mat.NewDense(input.RawMatrix().Rows, nn.config.numHiddenNeurons, nil)
	hiddenInput.Mul(input, nn.wHidden)
	hiddenInput.Add(hiddenInput, nn.bHidden)

	// Apply the ReLU activation function to the output of the hidden layer
	hiddenOutput := mat.NewDense(hiddenInput.RawMatrix().Rows, nn.config.numHiddenNeurons, nil)
	applyFunc(hiddenInput, hiddenOutput, relu)

	// Compute the input to the output layer
	finalInput := mat.NewDense(hiddenOutput.RawMatrix().Rows, nn.config.numOutputNeurons, nil)
	finalInput.Mul(hiddenOutput, nn.wOutput)
	finalInput.Add(finalInput, nn.bOutput)

	// Apply the softmax activation function to the output of the output layer
	finalOutput := applySoftmaxToDense(finalInput)

	// Calculate the cross-entropy loss
	loss, err := crossEntropyLoss(yTrue, finalOutput)
	if err != nil {
		return nil, 0, err
	}

	return finalOutput, loss, nil
}

func applySoftmaxToDense(input *mat.Dense) *mat.Dense {
	r, c := input.Dims()
	output := mat.NewDense(r, c, nil)

	for i := 0; i < r; i++ {
		row := mat.Row(nil, i, input)
		softmaxRow := softmax(row)
		for j, prob := range softmaxRow {
			output.Set(i, j, prob)
		}
	}

	return output
}

func applyFunc(input, output *mat.Dense, f func(float64) float64) {
	r, c := input.Dims()
	for i := 0; i < r; i++ {
		for j := 0; j < c; j++ {
			output.Set(i, j, f(input.At(i, j)))
		}
	}
}

func crossEntropyLoss(yTrue *mat.Dense, yPred *mat.Dense) (float64, error) {
	if yTrue.Dims() != yPred.Dims() {
		return 0, errors.New("Dimensions of yTrue and yPred do not match")
	}

	r, c := yTrue.Dims()
	totalLoss := 0.0

	for i := 0; i < r; i++ {
		for j := 0; j < c; j++ {
			trueVal := yTrue.At(i, j)
			predVal := yPred.At(i, j)

			// Clip the predicted value to avoid log(0)
			predVal = math.Max(1e-10, math.Min(1-1e-10, predVal))

			// If trueVal is zero, ignore this term
			if trueVal != 0 {
				totalLoss += trueVal * math.Log(predVal)
			}
		}
	}

	return -totalLoss, nil
}

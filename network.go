package main

import (
	"errors"
	"gonum.org/v1/gonum/floats"
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

func (nn *NeuralNetwork) train(features, labels *mat.Dense) error {
	nn.initializeWeightsAndBiases()

	output := new(mat.Dense)

	if err := nn.backpropagation(features, labels, output); err != nil {
		return err
	}

	return nil
}

func (nn *NeuralNetwork) backpropagation(features, labels, output *mat.Dense) error {
	for i := 0; i < nn.config.numEpochs; i++ {
		// Forward pass
		hiddenLayerInput := new(mat.Dense)
		hiddenLayerInput.Mul(features, nn.wHidden)
		addBHidden := func(_, col int, v float64) float64 { return v + nn.bHidden.At(0, col) }
		hiddenLayerInput.Apply(addBHidden, hiddenLayerInput)

		hiddenLayerActivations := new(mat.Dense)
		applyReLU := func(_, _ int, v float64) float64 { return relu(v) }
		hiddenLayerActivations.Apply(applyReLU, hiddenLayerInput)

		outputLayerInput := new(mat.Dense)
		outputLayerInput.Mul(hiddenLayerActivations, nn.wOutput)
		addBOut := func(_, col int, v float64) float64 { return v + nn.bOutput.At(0, col) }
		outputLayerInput.Apply(addBOut, outputLayerInput)
		output.Apply(applyReLU, outputLayerInput)

		// Backpropagation
		networkError := new(mat.Dense)
		networkError.Sub(labels, output)

		slopeOutputLayer := new(mat.Dense)
		applyReLUDerivative := func(_, _ int, v float64) float64 { return reluDerivative(v) }
		slopeOutputLayer.Apply(applyReLUDerivative, output)
		slopeHiddenLayer := new(mat.Dense)
		slopeHiddenLayer.Apply(applyReLUDerivative, hiddenLayerActivations)

		dOutput := new(mat.Dense)
		dOutput.MulElem(networkError, slopeOutputLayer)
		errorAtHiddenLayer := new(mat.Dense)
		errorAtHiddenLayer.Mul(dOutput, nn.wOutput.T())

		dHiddenLayer := new(mat.Dense)
		dHiddenLayer.MulElem(errorAtHiddenLayer, slopeHiddenLayer)

		// Adjust parameters
		wOutAdjusted := new(mat.Dense)
		wOutAdjusted.Mul(hiddenLayerActivations.T(), dOutput)
		wOutAdjusted.Scale(nn.config.learningRate, wOutAdjusted)
		nn.wOutput.Add(nn.wOutput, wOutAdjusted)

		bOutAdjusted, err := sumAlongAxis(0, dOutput)
		if err != nil {
			return err
		}
		bOutAdjusted.Scale(nn.config.learningRate, bOutAdjusted)
		nn.bOutput.Add(nn.bOutput, bOutAdjusted)

		wHiddenAdjusted := new(mat.Dense)
		wHiddenAdjusted.Mul(features.T(), dHiddenLayer)
		wHiddenAdjusted.Scale(nn.config.learningRate, wHiddenAdjusted)
		wHiddenAdjusted.Add(nn.wHidden, wHiddenAdjusted)

		bHiddenAdjusted, err := sumAlongAxis(0, dHiddenLayer)
		if err != nil {
			return err
		}
		bHiddenAdjusted.Scale(nn.config.learningRate, bHiddenAdjusted)
		bHiddenAdjusted.Add(nn.bHidden, bHiddenAdjusted)
	}
	return nil
}

func sumAlongAxis(axis int, m *mat.Dense) (*mat.Dense, error) {
	numRows, numCols := m.Dims()
	var output *mat.Dense

	switch axis {
	case 0:
		data := make([]float64, numCols)
		for i := 0; i < numCols; i++ {
			col := mat.Col(nil, i, m)
			data[i] = floats.Sum(col)
		}
		output = mat.NewDense(1, numCols, data)
	case 1:
		data := make([]float64, numRows)
		for i := 0; i < numRows; i++ {
			row := mat.Row(nil, i, m)
			data[i] = floats.Sum(row)
		}
		output = mat.NewDense(numRows, 1, data)
	default:
		return nil, errors.New("Invalid axis, must be 0 or 1.")
	}

	return output, nil
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

func (nn *NeuralNetwork) forwardPass(input *mat.Dense) (*mat.Dense, error) {
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
	finalOutput := applySoftmax(finalInput)

	return finalOutput, nil
}

func applySoftmax(input *mat.Dense) *mat.Dense {
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

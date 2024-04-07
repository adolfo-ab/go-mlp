package main

import "fmt"

func main() {
	data, err := loadData()
	if err != nil {
		panic("Oh no!!")
	}

	features, labels, err := parseData(data)
	if err != nil {
		panic("Failed when parsing data.")
	}

	trainFeatures, trainLabels, _, _ := splitData(features, labels)

	nnConfig := &NeuralNetworkConfig{
		numInputNeurons:  4,
		numHiddenNeurons: 3,
		numOutputNeurons: 3,
		numEpochs:        5000,
		learningRate:     0.1,
	}

	nn := NewNeuralNetwork(*nnConfig)
	fmt.Println("Before training: ")
	fmt.Println(nn.wHidden.RawMatrix().Data)

	nn.train(trainFeatures, trainLabels)
	fmt.Println("After training: ")
	fmt.Println(nn.wHidden.RawMatrix().Data)

}

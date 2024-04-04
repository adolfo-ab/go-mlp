package main

import "fmt"

func main() {
	/*
		data, err := loadData()
		if err != nil {
			panic("Oh no!!")
		}

		features, labels, err := parseData(data)
		if err != nil {
			panic("Failed when parsing data.")
		}

		trainFeatures, trainLabels, testFeatures, testLabels := splitData(features, labels)
		fmt.Println("Training features: ")
		fmt.Println(trainFeatures)
		fmt.Println("Length of training features: ")
		fmt.Println(len(trainFeatures))

		fmt.Println("Training labels: ")
		fmt.Println(trainLabels)
		fmt.Println("Length of training labels: ")
		fmt.Println(len(trainLabels))

		fmt.Println("Testing features: ")
		fmt.Println(testFeatures)
		fmt.Println("Length of testing features: ")
		fmt.Println(len(testFeatures))

		fmt.Println("Testing labels: ")
		fmt.Println(testLabels)
		fmt.Println("Length of testing labels: ")
		fmt.Println(len(testLabels))
	*/
	nnConfig := &NeuralNetworkConfig{
		numInputNeurons:  4,
		numHiddenNeurons: 10,
		numOutputNeurons: 3,
		numEpochs:        1,
		learningRate:     0.1,
	}

	nn := NewNeuralNetwork(*nnConfig)
	fmt.Print(nn.wHidden.RawMatrix().Data)

}

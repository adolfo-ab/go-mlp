package main

import (
	"fmt"
	"gonum.org/v1/gonum/mat"
	"log"
)

func main() {
	data, err := loadData()
	if err != nil {
		panic("Oh no!!")
	}

	features, labels, err := parseData(data)
	if err != nil {
		panic("Failed when parsing data.")
	}

	trainFeatures, trainLabels, testFeatures, testLabels := splitData(features, labels)

	nnConfig := &NeuralNetworkConfig{
		numInputNeurons:  4,
		numHiddenNeurons: 10,
		numOutputNeurons: 3,
		numEpochs:        5000,
		learningRate:     0.5,
	}

	nn := NewNeuralNetwork(*nnConfig)
	fmt.Println("Before training: ")
	fmt.Println(nn.wHidden.RawMatrix().Data)

	nn.train(trainFeatures, trainLabels)
	fmt.Println("After training: ")
	fmt.Println(nn.wHidden.RawMatrix().Data)

	predictions, err := nn.predict(testFeatures)
	if err != nil {
		log.Fatal(err)
	}

	accuracy := calculateAccuracy(predictions, testLabels)
	fmt.Printf("Accuracy: %.2f%%\n", accuracy*100)

}

func calculateAccuracy(predictions, labels *mat.Dense) float64 {
	correctPredictions := 0
	numRows, _ := predictions.Dims()

	for i := 0; i < numRows; i++ {
		predRow := mat.Row(nil, i, predictions)
		labelRow := mat.Row(nil, i, labels)

		predLabel := argMax(predRow)
		trueLabel := argMax(labelRow)

		if predLabel == trueLabel {
			correctPredictions++
		}
	}

	return float64(correctPredictions) / float64(numRows)
}

func argMax(slice []float64) int {
	maxIndex := 0
	maxValue := slice[0]
	for i, value := range slice {
		if value > maxValue {
			maxIndex = i
			maxValue = value
		}
	}
	return maxIndex
}

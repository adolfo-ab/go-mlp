package main

import "fmt"

func main() {
	data, err := loadData()
	if err != nil {
		panic("Oh no!!")
	}

	parsedData, err := parseData(data)
	if err != nil {
		panic("Oh no!")
	}

	trainFeatures, trainLabels, testFeatures, testLabels := splitData(parsedData)

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
}

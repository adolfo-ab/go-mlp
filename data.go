package main

import (
	"bufio"
	"encoding/csv"
	"gonum.org/v1/gonum/mat"
	"math/rand"
	"os"
	"strconv"
)

const fileName = "./data/iris.data"
const trainSize = 0.8

func loadData() ([][]string, error) {
	csvFile, err := os.Open(fileName)
	if err != nil {
		return nil, err
	}
	defer csvFile.Close()

	reader := csv.NewReader(bufio.NewReader(csvFile))
	lines, err := reader.ReadAll()
	if err != nil {
		return nil, err
	}

	return lines, nil
}

func parseData(data [][]string) ([][]float64, [][]float64, error) {
	// Shuffle the data
	rand.Shuffle(len(data), func(i, j int) {
		data[i], data[j] = data[j], data[i]
	})

	// Parse data, splitting into features and one-hot encoded labels in the process
	features := make([][]float64, len(data))
	labels := make([][]float64, len(data))

	for i, row := range data {
		features[i] = make([]float64, len(row)-1)
		labels[i] = make([]float64, 3)

		for j, value := range row {
			if j == len(row)-1 {
				switch value {
				case "Iris-setosa":
					labels[i][0] = 1
				case "Iris-versicolor":
					labels[i][1] = 1
				case "Iris-virginica":
					labels[i][2] = 1
				}
			} else {
				if val, err := strconv.ParseFloat(value, 64); err != nil {
					return nil, nil, err
				} else {
					features[i][j] = val
				}
			}
		}
	}
	return features, labels, nil
}

func splitData(features [][]float64, labels [][]float64) (*mat.Dense, *mat.Dense, *mat.Dense, *mat.Dense) {
	// Split into training and testing sets
	splitIndex := int(float64(len(features)) * trainSize)
	trainFeatures, testFeatures := features[:splitIndex], features[splitIndex:]
	trainLabels, testLabels := labels[:splitIndex], labels[splitIndex:]

	return sliceToDense(trainFeatures), sliceToDense(trainLabels), sliceToDense(testFeatures), sliceToDense(testLabels)
}

func sliceToDense(data [][]float64) *mat.Dense {
	var total int
	rows := len(data)
	cols := 0
	if rows > 0 {
		cols = len(data[0])
		total = rows * cols
	}

	flat := make([]float64, 0, total)

	for _, row := range data {
		flat = append(flat, row...)
	}

	return mat.NewDense(rows, cols, flat)
}

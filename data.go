package main

import (
	"bufio"
	"encoding/csv"
	"fmt"
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

func parseData(data [][]string) ([][]float64, error) {
	parsedData := make([][]float64, len(data))
	speciesMap := map[string]float64{
		"Iris-setosa":     1.0,
		"Iris-versicolor": 2.0,
		"Iris-virginica":  3.0,
	}

	for i, row := range data {
		parsedRow := make([]float64, len(row))
		for j := range row {
			if j == len(row)-1 {
				species := row[j]
				if val, ok := speciesMap[species]; ok {
					parsedRow[j] = val
				} else {
					return nil, fmt.Errorf("unknown species: %s", species)
				}
			} else {
				val, err := strconv.ParseFloat(row[j], 64)
				if err != nil {
					return nil, err
				}
				parsedRow[j] = val
			}
		}
		parsedData[i] = parsedRow
	}

	return parsedData, nil
}

func splitData(data [][]float64) ([][]float64, []float64, [][]float64, []float64) {
	// Shuffle the data
	rand.Shuffle(len(data), func(i, j int) {
		data[i], data[j] = data[j], data[i]
	})

	// Separate features and labels
	features := make([][]float64, len(data))
	labels := make([]float64, len(data))

	for i, row := range data {
		features[i] = row[:len(row)-1]
		labels[i] = row[len(row)-1]
	}

	// Split into training and testing sets
	splitIndex := int(float64(len(data)) * trainSize)
	trainFeatures, testFeatures := features[:splitIndex], features[splitIndex:]
	trainLabels, testLabels := labels[:splitIndex], labels[splitIndex:]

	return trainFeatures, trainLabels, testFeatures, testLabels
}

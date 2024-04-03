package main

import (
	"bufio"
	"encoding/csv"
	"fmt"
	"os"
	"strconv"
)

const fileName = "./data/iris.data"

func loadData() ([][]string, error) {
	csvFile, err := os.Open(fileName)
	if err != nil {
		return nil, err
	}
	defer func(csvFile *os.File) {
		err := csvFile.Close()
		if err != nil {
			panic(`Couldn't close file`)
		}
	}(csvFile)

	reader := csv.NewReader(bufio.NewReader(csvFile))
	lines, err := reader.ReadAll()
	if err != nil {
		return nil, err
	}

	return lines, nil
}

func parseData(data [][]string) ([][]float64, error) {
	parsedData := make([][]float64, len(data))

	for i, row := range data {
		parsedRow := make([]float64, len(row))
		for j := 0; j < len(row)-1; j++ {
			val, err := strconv.ParseFloat(row[j], 64)
			if err != nil {
				return nil, err
			}
			parsedRow[j] = val
		}
		species := row[len(row)-1]

		switch species {
		case "Iris-setosa":
			parsedRow[len(row)-1] = 1.0
		case "Iris-versicolor":
			parsedRow[len(row)-1] = 2.0
		case "Iris-virginica":
			parsedRow[len(row)-1] = 3.0
		default:
			return nil, fmt.Errorf("unknown species: %s", species)
		}
		parsedData[i] = parsedRow
	}

	return parsedData, nil
}

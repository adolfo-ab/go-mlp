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

	fmt.Println(parsedData)
}

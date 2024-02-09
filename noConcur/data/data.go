package data

import (
	"encoding/csv"
	"os"
	"strconv"

	"gonum.org/v1/gonum/mat"
)

// LoadDataset reads the dataset from a CSV file and returns matrices for the selected predictors and the target variable.
func LoadDataset(filePath string, selectedColumns []int) (*mat.Dense, *mat.Dense, error) {
	file, err := os.Open(filePath)
	if err != nil {
		return nil, nil, err
	}
	defer file.Close()

	reader := csv.NewReader(file)
	rawCSVData, err := reader.ReadAll()
	if err != nil {
		return nil, nil, err
	}

	rows := len(rawCSVData) - 1 // Minus 1 to account for the header row
	cols := len(selectedColumns)

	predictorsData := make([]float64, rows*cols)
	targetData := make([]float64, rows)

	for i, record := range rawCSVData[1:] {
		for j, colIndex := range selectedColumns {
			val, err := strconv.ParseFloat(record[colIndex], 64)
			if err != nil {
				return nil, nil, err
			}
			predictorsData[i*cols+j] = val
		}
		// target variable is the last column
		targetVal, err := strconv.ParseFloat(record[len(record)-1], 64)
		if err != nil {
			return nil, nil, err
		}
		targetData[i] = targetVal
	}

	predictors := mat.NewDense(rows, cols, predictorsData)
	target := mat.NewDense(rows, 1, targetData)

	return predictors, target, nil
}

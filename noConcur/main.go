package main

import (
	"encoding/csv"
	"fmt"
	"log"
	"math"
	"math/rand"
	"os"
	"strconv"
	"time"

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

// split the data into training and testing sets
func splitDataset(X, y *mat.Dense, ratio float64) (*mat.Dense, *mat.Dense, *mat.Dense, *mat.Dense) {
	nRows, _ := X.Dims()
	nTrain := int(float64(nRows) * ratio)
	nTest := nRows - nTrain
	rand.Seed(time.Now().Unix())

	// Permute the data
	indices := rand.Perm(nRows)
	trainIdx, testIdx := indices[:nTrain], indices[nTrain:]

	trainX := mat.NewDense(nTrain, X.RawMatrix().Cols, nil)
	testX := mat.NewDense(nTest, X.RawMatrix().Cols, nil)
	trainY := mat.NewDense(nTrain, 1, nil)
	testY := mat.NewDense(nTest, 1, nil)

	for i, idx := range trainIdx {
		trainX.SetRow(i, X.RawRowView(idx))
		trainY.SetRow(i, y.RawRowView(idx))
	}
	for i, idx := range testIdx {
		testX.SetRow(i, X.RawRowView(idx))
		testY.SetRow(i, y.RawRowView(idx))
	}

	return trainX, trainY, testX, testY
}

// LinearRegression fits a linear model using the normal equation.
func LinearRegression(X, y *mat.Dense) *mat.Dense {
	var XTX mat.Dense
	XTX.Mul(X.T(), X)

	var XTXInv mat.Dense
	if err := XTXInv.Inverse(&XTX); err != nil {
		log.Fatal("Matrix is singular or near-singular: ", err)
	}

	var XTy mat.Dense
	XTy.Mul(X.T(), y)

	var theta mat.Dense
	theta.Mul(&XTXInv, &XTy)
	return &theta
}

// ComputeMSE calculates the mean squared error between the actual and predicted values.
func ComputeMSE(yActual, yPredicted *mat.Dense) float64 {
	var diff mat.Dense
	diff.Sub(yActual, yPredicted)

	// Create a new matrix to hold the result of the multiplication
	squaredDiff := new(mat.Dense)

	// Perform the element-wise multiplication
	squaredDiff.MulElem(&diff, &diff)

	// Calculate the mean squared error
	mse := mat.Sum(squaredDiff) / float64(yActual.RawMatrix().Rows)
	return mse
} // Missing closing brace was added here

// CalculateAIC calculates the Akaike Information Criterion for a model.
func CalculateAIC(mse float64, n, k int) float64 {
	aic := float64(n)*math.Log(mse) + 2*float64(k)
	return aic
}

// Predict uses the model coefficients to make predictions on new data.
func Predict(X *mat.Dense, theta *mat.Dense) *mat.Dense {
	// Number of rows in X
	rows, _ := X.Dims()

	// Create a matrix for the predictions
	predictions := mat.NewDense(rows, 1, nil)

	// Calculate predictions: predictions = X * theta
	predictions.Mul(X, theta)

	return predictions
}

func main() {
	filePath := "../boston.csv"
	// The indices for 'crim', 'nox', 'rooms', 'age', 'dis', 'rad', 'tax', 'ptratio' 'lstat'
	selectedColumns := []int{1, 5, 6, 7, 8, 9, 10, 11, 12}

	// Load the dataset
	predictors, target, err := LoadDataset(filePath, selectedColumns)
	if err != nil {
		log.Fatalf("Failed to load dataset: %v", err)
	}

	// Split the dataset into training and testing sets
	trainX, trainY, testX, testY := splitDataset(predictors, target, 0.8)

	// Fit a linear regression model
	theta := LinearRegression(trainX, trainY)

	// Predict on the testing set
	predictions := Predict(testX, theta)

	// Compute MSE
	mse := ComputeMSE(testY, predictions)

	// Compute AIC
	n, k := testY.Dims()
	aic := CalculateAIC(mse, n, k+1) // +1 for the intercept

	fmt.Printf("MSE: %v\n", mse)
	fmt.Printf("AIC: %v\n", aic)
}

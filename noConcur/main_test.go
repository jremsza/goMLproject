package main

import (
	"math"
	"testing"

	"gonum.org/v1/gonum/mat"
)

func TestLinearRegression(t *testing.T) {
	// Initialize some test data
	X := mat.NewDense(3, 2, []float64{1, 2, 3, 4, 5, 6})
	y := mat.NewDense(3, 1, []float64{1, 2, 3})

	// Call the function
	theta := LinearRegression(X, y)

	// Check the result
	if theta.At(0, 0) == 0 && theta.At(1, 0) == 0 {
		t.Errorf("Expected non-zero values in theta, got %v", theta)
	}
}

func TestComputeMSE(t *testing.T) {
	// Initialize the test
	yActual := mat.NewDense(4, 1, []float64{2, 3, 4, 5})
	yPredicted := mat.NewDense(4, 1, []float64{1, 2, 3, 4})
	mse := ComputeMSE(yActual, yPredicted)

	// Check the result
	if mse != 1.0 {
		t.Errorf("Expected 1.0, but got %v", mse)
	}
}
func TestComputeAIC(t *testing.T) {
	// Initialize the test
	mse := 1.0
	n := 100
	k := 2
	aic := CalculateAIC(mse, n, k)

	// Check the result
	expectedAIC := float64(n)*math.Log(mse) + 2*float64(k)
	if aic != expectedAIC {
		t.Errorf("Expected AIC to be %v, got %v", expectedAIC, aic)
	}
}

func TestPredict(t *testing.T) {
	// Initialize some test data
	X := mat.NewDense(3, 2, []float64{1, 2, 3, 4, 5, 6})
	theta := mat.NewDense(2, 1, []float64{1, 1})

	// Call the function
	predictions := Predict(X, theta)

	// Check the result
	if predictions.At(0, 0) != 3 || predictions.At(1, 0) != 7 || predictions.At(2, 0) != 11 {
		t.Errorf("Expected predictions to be [3 7 11], got %v", predictions)
	}
}
